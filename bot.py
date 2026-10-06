


import os
import sys
import csv
import json
import math
import time
import hmac
import hashlib
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone


import numpy as np
import pandas as pd
import requests
import ccxt




# =========================================================
# CONFIG  (override the important ones with env vars)
# =========================================================
def _ts(s: str) -> float:
   return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()




class Cfg:
   BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")


   # Competition window (UTC). END is mandatory to set correctly.
   END_TS = _ts(os.getenv("COMPETITION_END_UTC", "2026-10-19 12:00:00"))
   START_TS = _ts(os.getenv("COMPETITION_START_UTC")) if os.getenv("COMPETITION_START_UTC") else 0.0
   FLATTEN_BEFORE_MIN = 60      # go flat this many minutes before END (locks ratios)
   NO_ENTRY_BEFORE_H = 3        # no NEW entries in the last N hours


   # Universe (filtered against exchangeInfo at startup)
   UNIVERSE = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC", "FET"]
   DATA_EXCHANGES = ["binance", "binanceus", "kraken"]   # fallbacks if one is geo-blocked


   # Timing
   TIMEFRAME = "15m"
   CANDLE_LIMIT = 300
   POLL_SECONDS = 20            # stop-check cadence (one ticker call per poll)
   GRACE_SECONDS = 8            # wait after bar close before trusting the candle
   MIN_ORDER_GAP = 2.0          # seconds between orders. SET PER ROOSTOO RATE RULES.


   # Costs (per side)
   FEE_RATE = 0.001
   SLIPPAGE = 0.0005
   COST_RT = 2 * (FEE_RATE + SLIPPAGE)


   # Entry filters
   ADX_MIN = 20.0
   MAX_EXT_ATR = 3.0            # skip if price > EMA50 + 3*ATR (overextended)
   MOM_LOOKBACK = 96            # bars (24h on 15m) for ranking
   MIN_ATR_PCT = COST_RT * 1.0  # ENTRY-ONLY volatility floor (log ATR% to calibrate)


   # Position sizing
   MAX_POSITIONS = 3
   MAX_POS_PCT = 0.30           # max equity in one coin
   MAX_TOTAL_EXPOSURE = 0.90
   BASE_RISK = 0.015            # equity risked per trade at initial stop
   HIGH_CONV_MULT = 1.5         # risk multiplier when ADX strong
   ADX_HIGH = 30.0


   # ---- Stop / profit-capture profile --------------------------------------
   # Observed Roostoo fee is 0.1%/side, so a round trip costs ~0.2% (+slippage ~0.3%).
   # Any exit below ~+0.35% gross is a net loss.
   #   "wide"  (default): wider initial stop + noise grace, then lock to NO-LOSS quickly
   #   "tight": near-no-loss stop from the start (expect more, smaller stop-outs = more fees)
   STOP_MODE = os.getenv("STOP_MODE", "wide").lower()
   _P = {
       "wide":  dict(init_atr=2.5, min_stop=0.008, grace=3, lock_at=0.005, lock_floor=0.003,
                     trail_atr=1.5, min_trail=0.004, gb1_at=0.010, gb1=0.40, gb2_at=0.020, gb2=0.30,
                     tp1=0.008, tp2=0.016, fade_min=0.005),
       "tight": dict(init_atr=1.0, min_stop=0.004, grace=0, lock_at=0.004, lock_floor=0.0025,
                     trail_atr=1.0, min_trail=0.003, gb1_at=0.007, gb1=0.35, gb2_at=0.012, gb2=0.25,
                     tp1=0.005, tp2=0.010, fade_min=0.004),
   }[STOP_MODE if STOP_MODE in ("wide", "tight") else "wide"]


   INIT_STOP_ATR = _P["init_atr"]       # initial stop distance (ATR multiples)
   MIN_STOP_PCT = _P["min_stop"]        # ...but never closer than this fraction of price
   TREND_FAIL_GRACE = _P["grace"]       # bars before a trend-fail exit may fire at a loss
   LOCK_AT = _P["lock_at"]              # peak gain that arms the no-loss lock
   LOCK_FLOOR = _P["lock_floor"]        # stop jumps to entry*(1+this) (~breakeven after costs)
   TRAIL_ATR = _P["trail_atr"]          # trailing distance once armed
   MIN_TRAIL_PCT = _P["min_trail"]
   GB1_AT, GB1 = _P["gb1_at"], _P["gb1"]    # once peak >= GB1_AT, give back at most GB1 of peak gain
   GB2_AT, GB2 = _P["gb2_at"], _P["gb2"]
   TP1_PCT, TP2_PCT = _P["tp1"], _P["tp2"]  # take-profit tiers (gross %), also scaled by ATR
   TP_ATR_MULT = 1.5
   TP1_FRAC, TP2_FRAC = 0.5, 0.5        # TP1 sells 50% of position, TP2 50% of the remainder
   FADE_EXIT = True                     # exit when MACD rolls over while in profit
   FADE_MIN_PNL = _P["fade_min"]


   MAX_BARS_HELD = 12                   # 3h
   MIN_PROGRESS = 0.006                 # must be > +0.6% by then or rotate out
   COOLDOWN_LOSS_BARS = 4               # no re-entry after a losing exit
   COOLDOWN_WIN_BARS = 2                # ...and a shorter one after a winner (stops same-price re-buys)
   MAX_NEW_PER_BAR = 1                  # stagger entries; avoids a correlated cluster at once


   # Portfolio protection
   DD_SOFT = 0.04               # halve risk
   DD_HARD = 0.08               # flatten + pause
   BREAKER_PAUSE_H = 12


   ADOPT_ORPHANS = False        # adopt unmanaged coin balances (off: initial wallet may hold coins)
   DRY_RUN_START_USD = 100000.0


   STATE_FILE = "state.json"
   TRADES_CSV = "trades.csv"
   EQUITY_CSV = "equity.csv"




BAR_MS = int(ccxt.Exchange.parse_timeframe(Cfg.TIMEFRAME) * 1000)


logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s - %(levelname)s - %(message)s",
   handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
)
logger = logging.getLogger("RoostooBot")




# =========================================================
# HELPERS
# =========================================================
def floor_to(x: float, prec: int) -> float:
   f = 10 ** prec
   return math.floor(x * f + 1e-9) / f




def _get(d, *keys, default=None):
   """Case-insensitive dict getter tolerant of schema differences."""
   if not isinstance(d, dict):
       return default
   low = {str(k).lower(): v for k, v in d.items()}
   for k in keys:
       v = low.get(k.lower())
       if v is not None:
           return v
   return default




def append_csv(path, header, row):
   new = not os.path.exists(path)
   with open(path, "a", newline="") as f:
       w = csv.writer(f)
       if new:
           w.writerow(header)
       w.writerow(row)




@dataclass
class Fill:
   ok: bool
   qty: float = 0.0
   price: float = 0.0
   msg: str = ""
   uncertain: bool = False   # network ambiguity: order may or may not have executed




# =========================================================
# STATE (atomic writes)
# =========================================================
class StateManager:
   def __init__(self, filename=Cfg.STATE_FILE):
       self.filename = filename


   def load(self) -> dict:
       if os.path.exists(self.filename):
           try:
               with open(self.filename) as f:
                   return json.load(f)
           except Exception as e:
               logger.error(f"State load failed ({e}); starting fresh.")
       return {"positions": {}, "meta": {}}


   def save(self, data: dict):
       tmp = self.filename + ".tmp"
       try:
           with open(tmp, "w") as f:
               json.dump(data, f, indent=2)
               f.flush()
               os.fsync(f.fileno())
           os.replace(tmp, self.filename)
       except Exception as e:
           logger.error(f"State save failed: {e}")




# =========================================================
# ROOSTOO CLIENT
# =========================================================
class RoostooClient:
   def __init__(self, api_key: str, api_secret: str, dry_run: bool = False):
       self.key = api_key
       self.secret = api_secret.encode()
       self.dry_run = dry_run
       self.session = requests.Session()
       self._last_order = 0.0
       self.virtual = {"USD": Cfg.DRY_RUN_START_USD}


   # -- signing -------------------------------------------------
   def _signed(self, params: dict):
       p = dict(params)
       p["timestamp"] = int(time.time() * 1000)
       qs = "&".join(f"{k}={p[k]}" for k in sorted(p))
       sig = hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()
       headers = {"RST-API-KEY": self.key, "MSG-SIGNATURE": sig}
       return headers, qs


   def _get_json(self, path: str, params=None, attempts=3):
       for i in range(attempts):
           headers, qs = self._signed(params or {})
           try:
               r = self.session.get(f"{Cfg.BASE_URL}{path}?{qs}", headers=headers, timeout=10)
               if r.status_code == 200:
                   return r.json()
               logger.error(f"GET {path} -> HTTP {r.status_code}")
               if r.status_code < 500 and r.status_code != 429:
                   return None
           except (requests.RequestException, ValueError) as e:
               logger.error(f"GET {path} failed: {e}")
           time.sleep(1.5 * (i + 1))
       return None


   # -- market data / account -----------------------------------
   def exchange_info(self) -> dict:
       data = self._get_json("/v3/exchangeInfo")
       return _get(data, "TradePairs", default={}) or {}


   def get_tickers(self) -> dict:
       data = self._get_json("/v3/ticker")
       if not data or not _get(data, "Success", default=True):
           return {}
       out = {}
       for pair, v in (_get(data, "Data", default={}) or {}).items():
           px = _get(v, "LastPrice", "last", default=None)
           if px:
               out[pair] = float(px)
       return out


   def get_wallet(self):
       """Returns {asset: free} or None on failure."""
       if self.dry_run:
           return dict(self.virtual)
       data = self._get_json("/v3/balance")
       if not data or not _get(data, "Success", default=False):
           return None
       wallet = {}
       spot = _get(data, "SpotWallet")
       if isinstance(spot, dict):
           for a, v in spot.items():
               wallet[a] = float(_get(v, "Free", default=0) or 0)
       for item in _get(data, "balances", default=[]) or []:
           wallet[item.get("asset")] = float(item.get("free", 0) or 0)
       return wallet


   # -- orders --------------------------------------------------
   def _throttle(self):
       wait = Cfg.MIN_ORDER_GAP - (time.time() - self._last_order)
       if wait > 0:
           time.sleep(wait)
       self._last_order = time.time()


   def place_market_order(self, pair: str, side: str, qty: float, prec: int, ref_price: float) -> Fill:
       side = side.upper()
       self._throttle()
       qty_str = f"{qty:.{prec}f}"


       if self.dry_run:
           asset = pair.split("/")[0]
           px = ref_price * (1 + Cfg.SLIPPAGE if side == "BUY" else 1 - Cfg.SLIPPAGE)
           if side == "BUY":
               cost = qty * px * (1 + Cfg.FEE_RATE)
               if cost > self.virtual["USD"]:
                   return Fill(False, msg="dry-run: insufficient USD")
               self.virtual["USD"] -= cost
               self.virtual[asset] = self.virtual.get(asset, 0.0) + qty
           else:
               if qty > self.virtual.get(asset, 0.0) + 1e-12:
                   return Fill(False, msg="dry-run: insufficient coin")
               self.virtual[asset] -= qty
               self.virtual["USD"] += qty * px * (1 - Cfg.FEE_RATE)
           logger.info(f"[DRY] {side} {qty_str} {pair} @ {px:.4f}")
           return Fill(True, qty, px, "dry-run")


       headers, qs = self._signed(
           {"pair": pair, "side": side, "type": "MARKET", "quantity": qty_str}
       )
       headers["Content-Type"] = "application/x-www-form-urlencoded"
       try:
           r = self.session.post(f"{Cfg.BASE_URL}/v3/place_order", headers=headers, data=qs, timeout=10)
       except requests.RequestException as e:
           logger.error(f"Order network failure {side} {pair}: {e}")
           return Fill(False, msg=str(e), uncertain=True)   # DO NOT blindly retry


       if r.status_code >= 500:
           return Fill(False, msg=f"HTTP {r.status_code}", uncertain=True)
       if r.status_code != 200:
           return Fill(False, msg=f"HTTP {r.status_code}: {r.text[:200]}")
       try:
           data = r.json()
       except ValueError:
           return Fill(False, msg="bad JSON", uncertain=True)


       if not _get(data, "Success", default=False):
           msg = _get(data, "ErrMsg", "message", default="unknown")
           logger.error(f"Order rejected {side} {qty_str} {pair}: {msg}")
           return Fill(False, msg=str(msg))


       d = _get(data, "OrderDetail", default={}) or {}
       filled = float(_get(d, "FilledQuantity", default=qty) or 0)
       avg = float(_get(d, "FilledAverPrice", default=ref_price) or ref_price)
       if filled <= 0:
           filled = qty if str(_get(d, "Status", default="FILLED")).upper() == "FILLED" else 0.0
       if avg <= 0:
           avg = ref_price
       logger.info(f"[EXEC] {side} {filled} {pair} @ {avg:.4f} | id={_get(d, 'OrderID', default='?')}")
       return Fill(filled > 0, filled, avg, "ok")




# =========================================================
# MARKET DATA (closed candles only, with exchange fallbacks)
# =========================================================
class DataFeed:
   def __init__(self):
       self.exchanges = []
       for name in Cfg.DATA_EXCHANGES:
           try:
               self.exchanges.append(getattr(ccxt, name)({"enableRateLimit": True}))
           except Exception as e:
               logger.warning(f"Data exchange {name} unavailable: {e}")


   def candles(self, asset: str):
       now_ms = int(time.time() * 1000)
       for ex in self.exchanges:
           try:
               raw = ex.fetch_ohlcv(f"{asset}/USDT", Cfg.TIMEFRAME, limit=Cfg.CANDLE_LIMIT)
               df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"]).astype(float)
               df = df[df["ts"] + BAR_MS <= now_ms].reset_index(drop=True)   # closed bars only
               if len(df) >= 220:
                   return df
           except Exception as e:
               logger.debug(f"{ex.id} {asset} fetch failed: {e}")
       return None




def compute_signals(df: pd.DataFrame):
   c, h, l = df["close"], df["high"], df["low"]
   ema50 = c.ewm(span=50, adjust=False).mean()
   ema200 = c.ewm(span=200, adjust=False).mean()
   macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
   macd_sig = macd.ewm(span=9, adjust=False).mean()


   pc = c.shift()
   tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
   atr = tr.ewm(alpha=1 / 14, adjust=False).mean()


   up, dn = h.diff(), -l.diff()
   plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
   minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
   pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
   mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
   dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
   adx = dx.ewm(alpha=1 / 14, adjust=False).mean()
   roc = c / c.shift(Cfg.MOM_LOOKBACK) - 1


   i = -1
   s = {
       "ts": int(df["ts"].iloc[i]), "price": c.iloc[i], "atr": atr.iloc[i],
       "ema50": ema50.iloc[i], "ema200": ema200.iloc[i],
       "macd": macd.iloc[i], "signal": macd_sig.iloc[i],
       "adx": adx.iloc[i], "pdi": pdi.iloc[i], "mdi": mdi.iloc[i], "roc": roc.iloc[i],
   }
   if not all(np.isfinite(v) for v in s.values()) or s["price"] <= 0 or s["atr"] <= 0:
       return None
   s["ts"] = int(s["ts"])
   s["atr_pct"] = s["atr"] / s["price"]
   s["trend_up"] = (
       s["price"] > s["ema50"] and s["price"] > s["ema200"]
       and s["macd"] > s["signal"] and s["pdi"] > s["mdi"] and s["adx"] >= Cfg.ADX_MIN
   )
   s["extended"] = (s["price"] - s["ema50"]) > Cfg.MAX_EXT_ATR * s["atr"]
   s["trend_fail"] = s["price"] < s["ema50"] and s["macd"] < s["signal"]
   s["macd_down"] = s["macd"] < s["signal"]
   return s




# =========================================================
# ENGINE
# =========================================================
def new_pos(cooldown_until=0):
   return {"type": "NONE", "size": 0.0, "entry": 0.0, "stop": 0.0, "extremum": 0.0,
           "atr": 0.0, "risk": 0.0, "entry_bar": 0, "tp_done": 0,
           "cooldown_until": cooldown_until}




class Engine:
   def __init__(self, client: RoostooClient, feed: DataFeed, state: StateManager):
       self.c, self.feed, self.st = client, feed, state
       data = state.load()
       self.pos = data.get("positions", {})
       self.meta = {"peak_equity": 0.0, "last_bar": 0, "paused_until": 0.0}
       self.meta.update(data.get("meta", {}))
       self.rules = {}
       self.need_reconcile = True
       self._waiting_logged = False
       self._init_universe()


   def _init_universe(self):
       info = self.c.exchange_info()
       for asset in Cfg.UNIVERSE:
           pair = f"{asset}/USD"
           spec = info.get(pair)
           if info and not spec:
               logger.warning(f"{pair} not listed on Roostoo; skipped")
               continue
           if spec and not _get(spec, "CanTrade", default=True):
               logger.warning(f"{pair} not tradable; skipped")
               continue
           prec = int(_get(spec, "AmountPrecision", default=3)) if spec else 3
           mini = float(_get(spec, "MiniOrder", default=1.0)) if spec else 1.0
           self.rules[asset] = {"pair": pair, "prec": prec, "min": mini}
           self.pos[asset] = {**new_pos(), **self.pos.get(asset, {})}   # also migrates old state files
       if not info:
           logger.warning("exchangeInfo unavailable: using default precision/min order. VERIFY.")
       logger.info(f"Universe: {list(self.rules)}")


   # -- bookkeeping ---------------------------------------------
   def _save(self):
       self.st.save({"positions": self.pos, "meta": self.meta})


   def _held(self):
       return [a for a, p in self.pos.items() if p["type"] == "LONG" and a in self.rules]


   def _reset(self, asset, cooldown_until=None):
       cd = self.pos[asset]["cooldown_until"] if cooldown_until is None else cooldown_until
       self.pos[asset] = new_pos(cd)


   def equity(self, wallet: dict, prices: dict) -> float:
       eq = wallet.get("USD", 0.0)
       for a, r in self.rules.items():
           eq += wallet.get(a, 0.0) * prices.get(r["pair"], 0.0)
       return eq


   def _journal(self, action, asset, reason, qty, price, pnl=""):
       append_csv(Cfg.TRADES_CSV, ["utc", "action", "asset", "reason", "qty", "price", "pnl_pct"],
                  [datetime.now(timezone.utc).isoformat(timespec="seconds"), action, asset,
                   reason, qty, round(price, 6), pnl])


   # -- reconciliation ------------------------------------------
   def reconcile(self, wallet: dict, prices: dict):
       for asset, rule in self.rules.items():
           price = prices.get(rule["pair"])
           if not price:
               continue
           free = wallet.get(asset, 0.0)
           pos = self.pos[asset]
           value = free * price
           if pos["type"] == "LONG":
               if value < rule["min"]:
                   logger.warning(f"[{asset}] state LONG but wallet empty -> resetting")
                   self._reset(asset)
               elif free < pos["size"] * 0.995:      # fees taken in coin, partial fills etc.
                   pos["size"] = floor_to(free, rule["prec"])
           elif value >= rule["min"] * 1.5:
               if Cfg.ADOPT_ORPHANS:
                   logger.warning(f"[{asset}] adopting orphan balance {free}")
                   pos.update({"type": "LONG", "size": floor_to(free, rule["prec"]), "entry": price,
                               "extremum": price, "atr": price * 0.006, "risk": price * 0.012,
                               "stop": price * 0.988, "entry_bar": 0, "tp_done": 0})
               else:
                   logger.warning(f"[{asset}] unmanaged balance {free} (set ADOPT_ORPHANS to manage)")
       self.need_reconcile = False


   # -- order wrappers ------------------------------------------
   def _sell(self, asset, qty, price):
       rule = self.rules[asset]
       qty = floor_to(qty, rule["prec"])
       if qty <= 0:
           return None
       fill = self.c.place_market_order(rule["pair"], "SELL", qty, rule["prec"], price)
       if fill.uncertain:
           self.need_reconcile = True
           return None
       if not fill.ok:   # likely insufficient balance (fees in coin): retry once with real balance
           wallet = self.c.get_wallet() or {}
           free = floor_to(min(qty, wallet.get(asset, 0.0)), rule["prec"])
           if 0 < free < qty:
               fill = self.c.place_market_order(rule["pair"], "SELL", free, rule["prec"], price)
       return fill if fill.ok else None


   def close_position(self, asset, price, reason) -> bool:
       pos, rule = self.pos[asset], self.rules[asset]
       if pos["size"] * price < rule["min"]:         # dust: nothing sellable
           self._reset(asset)
           return True
       fill = self._sell(asset, pos["size"], price)
       if fill is None:
           logger.error(f"[{asset}] exit '{reason}' FAILED; will retry")
           return False
       pnl = fill.price / pos["entry"] - 1 if pos["entry"] else 0.0
       self._journal("SELL", asset, reason, fill.qty, fill.price, round(pnl * 100, 3))
       logger.info(f"[{asset}] CLOSED ({reason}) pnl={pnl * 100:+.2f}%")
       if fill.qty < pos["size"] * 0.98:             # partial fill: keep remainder, retry next tick
           pos["size"] = floor_to(pos["size"] - fill.qty, rule["prec"])
           return False
       if reason in ("endgame", "dd_breaker"):
           cd = 0
       else:
           cd = time.time() + (Cfg.COOLDOWN_LOSS_BARS if pnl < 0 else Cfg.COOLDOWN_WIN_BARS) * BAR_MS / 1000
       self._reset(asset, cd)
       return True


   def flatten_all(self, prices, reason) -> bool:
       ok = True
       for a in self._held():                         # exits first, in priority
           ok &= self.close_position(a, prices.get(self.rules[a]["pair"], self.pos[a]["entry"]), reason)
       return ok


   # -- fast loop: stops + profit capture on live ticker ---------
   def manage_tick(self, prices):
       for asset in self._held():
           pos, rule = self.pos[asset], self.rules[asset]
           price = prices.get(rule["pair"])
           if not price or pos["entry"] <= 0:
               continue
           entry = pos["entry"]
           pos["extremum"] = max(pos["extremum"], price)
           peak = pos["extremum"] / entry - 1
           gain = price / entry - 1
           atr_pct = pos["atr"] / entry


           # 1) Stop = ATR chandelier -> no-loss lock -> profit give-back cap (only ever ratchets up)
           armed = peak >= Cfg.LOCK_AT
           mult = Cfg.TRAIL_ATR if armed else Cfg.INIT_STOP_ATR
           floor_pct = Cfg.MIN_TRAIL_PCT if armed else Cfg.MIN_STOP_PCT
           trail = max(pos["atr"] * mult, pos["extremum"] * floor_pct)
           stop = max(pos["stop"], pos["extremum"] - trail)
           if armed:
               stop = max(stop, entry * (1 + Cfg.LOCK_FLOOR))
           gb = Cfg.GB2 if peak >= Cfg.GB2_AT else (Cfg.GB1 if peak >= Cfg.GB1_AT else None)
           if gb is not None:
               stop = max(stop, pos["extremum"] - gb * (pos["extremum"] - entry))
           pos["stop"] = stop
           if price <= stop:
               self.close_position(asset, price, "stop" if gain < 0 else "profit_stop")
               continue


           # 2) Take-profit ladder (sell into strength, even for small gains)
           tier = pos["tp_done"]
           if tier < 2:
               thr = max(Cfg.TP1_PCT if tier == 0 else Cfg.TP2_PCT,
                         Cfg.TP_ATR_MULT * atr_pct * (1 if tier == 0 else 2))
               if gain >= thr:
                   frac = Cfg.TP1_FRAC if tier == 0 else Cfg.TP2_FRAC
                   qty = floor_to(pos["size"] * frac, rule["prec"])
                   rest = pos["size"] - qty
                   if qty * price >= rule["min"] * 1.1 and rest * price >= rule["min"] * 1.1:
                       fill = self._sell(asset, qty, price)
                       if fill:
                           pos["size"] = floor_to(pos["size"] - fill.qty, rule["prec"])
                           pos["tp_done"] = tier + 1
                           pos["stop"] = max(pos["stop"], entry * (1 + Cfg.LOCK_FLOOR))
                           pnl = fill.price / entry - 1
                           self._journal("SELL", asset, f"tp{tier + 1}", fill.qty, fill.price, round(pnl * 100, 3))
                           logger.info(f"[{asset}] TP{tier + 1} {pnl * 100:+.2f}% | remaining {pos['size']}")
                   else:
                       pos["tp_done"] = 2    # too small to split


   # -- slow loop: once per closed candle -----------------------
   def candle_cycle(self, prices):
       now_ms = int(time.time() * 1000)
       bar_open = now_ms // BAR_MS * BAR_MS
       if bar_open <= self.meta["last_bar"] or now_ms < bar_open + Cfg.GRACE_SECONDS * 1000:
           return
       wallet = self.c.get_wallet()
       if wallet is None:
           logger.warning("Wallet fetch failed; retrying next poll")
           return
       self.reconcile(wallet, prices)
       equity = self.equity(wallet, prices)
       self.meta["peak_equity"] = max(self.meta["peak_equity"], equity)
       dd = 1 - equity / self.meta["peak_equity"] if self.meta["peak_equity"] > 0 else 0.0
       self.meta["last_bar"] = bar_open
       append_csv(Cfg.EQUITY_CSV, ["utc", "equity", "usd", "dd"],
                  [datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   round(equity, 2), round(wallet.get("USD", 0.0), 2), round(dd, 4)])
       logger.info(f"Bar {datetime.fromtimestamp(bar_open / 1000, timezone.utc):%m-%d %H:%M} | "
                   f"equity={equity:,.2f} dd={dd * 100:.2f}% held={self._held()}")


       # Portfolio circuit breaker
       if dd >= Cfg.DD_HARD:
           logger.warning(f"🚨 DD {dd * 100:.1f}% >= {Cfg.DD_HARD * 100:.0f}%: flatten + pause {Cfg.BREAKER_PAUSE_H}h")
           self.flatten_all(prices, "dd_breaker")
           self.meta["paused_until"] = time.time() + Cfg.BREAKER_PAUSE_H * 3600
           self.meta["peak_equity"] = 0.0     # re-anchor so the breaker does not re-fire at once
           return


       # Fresh signals
       sigs = {}
       for asset in self.rules:
           df = self.feed.candles(asset)
           s = compute_signals(df) if df is not None else None
           if s and s["ts"] == bar_open - BAR_MS:     # must be the just-closed bar
               sigs[asset] = s
           elif s:
               logger.debug(f"[{asset}] stale candle; skipped this cycle")


       # Manage held positions on candle close
       exited = False
       for asset in self._held():
           s, pos = sigs.get(asset), self.pos[asset]
           if not s:
               continue
           px = prices.get(self.rules[asset]["pair"], s["price"])
           pos["atr"] = s["atr"]
           bars = (s["ts"] - pos["entry_bar"]) // BAR_MS if pos["entry_bar"] else 0
           pnl = px / pos["entry"] - 1
           if Cfg.FADE_EXIT and s["macd_down"] and pnl >= Cfg.FADE_MIN_PNL:
               exited |= self.close_position(asset, px, "momentum_fade")        # bank the small profit
           elif s["trend_fail"] and (bars >= Cfg.TREND_FAIL_GRACE or pnl >= Cfg.COST_RT):
               exited |= self.close_position(asset, px, "trend_fail")
           elif bars >= Cfg.MAX_BARS_HELD and pnl < Cfg.MIN_PROGRESS:
               exited |= self.close_position(asset, px, "time")


       if exited:
           wallet = self.c.get_wallet() or wallet
           equity = self.equity(wallet, prices)


       # Entries
       now = time.time()
       if now < self.meta["paused_until"]:
           logger.info("Entries paused (breaker cooldown)")
           return
       if now >= Cfg.END_TS - Cfg.NO_ENTRY_BEFORE_H * 3600:
           return
       slots = Cfg.MAX_POSITIONS - len(self._held())
       if slots <= 0:
           return


       cands = []
       for asset, s in sigs.items():
           pos = self.pos[asset]
           if pos["type"] != "NONE" or now < pos["cooldown_until"]:
               continue
           logger.debug(f"[{asset}] atr%={s['atr_pct'] * 100:.3f} adx={s['adx']:.1f} roc={s['roc'] * 100:.2f}%")
           if s["trend_up"] and not s["extended"] and s["roc"] > 0 and s["atr_pct"] >= Cfg.MIN_ATR_PCT:
               cands.append((s["roc"] / s["atr_pct"], asset, s))
       cands.sort(key=lambda x: x[0], reverse=True)


       usd = wallet.get("USD", 0.0)
       exposure = sum(self.pos[a]["size"] * prices.get(self.rules[a]["pair"], 0) for a in self._held())
       for score, asset, s in cands[:min(slots, Cfg.MAX_NEW_PER_BAR)]:
           rule, price = self.rules[asset], prices.get(self.rules[asset]["pair"])
           if not price:
               continue
           qty = self._entry_size(rule, s, price, equity, usd, exposure, dd)
           if qty <= 0:
               continue
           fill = self.c.place_market_order(rule["pair"], "BUY", qty, rule["prec"], price)
           if fill.uncertain:
               self.need_reconcile = True
               continue
           if not fill.ok:
               continue
           stop_dist = self._stop_dist(s, fill.price)
           self.pos[asset].update({
               "type": "LONG", "size": fill.qty, "entry": fill.price, "extremum": fill.price,
               "atr": s["atr"], "risk": stop_dist, "stop": fill.price - stop_dist,
               "entry_bar": s["ts"], "tp_done": 0,
           })
           usd -= fill.qty * fill.price * (1 + Cfg.FEE_RATE)
           exposure += fill.qty * fill.price
           self._journal("BUY", asset, f"score={score:.2f} adx={s['adx']:.0f}", fill.qty, fill.price)
           logger.info(f"[{asset}] LONG {fill.qty} @ {fill.price:.4f} stop={self.pos[asset]['stop']:.4f}")


   @staticmethod
   def _stop_dist(s, price) -> float:
       return max(s["atr"] * Cfg.INIT_STOP_ATR, price * Cfg.MIN_STOP_PCT)


   def _entry_size(self, rule, s, price, equity, usd, exposure, dd) -> float:
       risk_pct = Cfg.BASE_RISK
       if dd >= Cfg.DD_SOFT:
           risk_pct *= 0.5
       elif s["adx"] >= Cfg.ADX_HIGH:
           risk_pct *= Cfg.HIGH_CONV_MULT
       stop_dist = self._stop_dist(s, price)
       notional = equity * risk_pct / stop_dist * price
       cap = min(equity * Cfg.MAX_POS_PCT,
                 equity * Cfg.MAX_TOTAL_EXPOSURE - exposure,
                 usd / (1 + Cfg.FEE_RATE + Cfg.SLIPPAGE) * 0.98)
       notional = min(notional, cap)
       if notional <= 0:
           return 0.0
       qty = floor_to(notional / price, rule["prec"])
       return qty if qty * price >= rule["min"] * 1.1 else 0.0


   # -- main tick -----------------------------------------------
   def tick(self):
       now = time.time()
       if Cfg.START_TS and now < Cfg.START_TS:
           if not self._waiting_logged:
               logger.info("Waiting for competition start...")
               self._waiting_logged = True
           return
       prices = self.c.get_tickers()
       if not prices:
           logger.warning("No ticker data this poll")
           return


       if self.need_reconcile:
           wallet = self.c.get_wallet()
           if wallet is not None:
               self.reconcile(wallet, prices)


       if now >= Cfg.END_TS - Cfg.FLATTEN_BEFORE_MIN * 60:
           if self._held():
               logger.warning("🏁 ENDGAME: flattening to lock ratios")
               self.flatten_all(prices, "endgame")
           self._save()
           return


       self.manage_tick(prices)
       self.candle_cycle(prices)
       self._save()




# =========================================================
# REPORT (approximate scoring metrics from equity.csv)
# =========================================================
def report(path=Cfg.EQUITY_CSV):
   if not os.path.exists(path):
       print("No equity.csv yet.")
       return
   eq = pd.read_csv(path)["equity"].astype(float)
   if len(eq) < 3:
       print("Not enough data.")
       return
   ret = eq.pct_change().dropna()
   per_year = 365 * 24 * 3600 / (BAR_MS / 1000)
   total = eq.iloc[-1] / eq.iloc[0] - 1
   mdd = (1 - eq / eq.cummax()).max()
   sharpe = ret.mean() / ret.std() * math.sqrt(per_year) if ret.std() > 0 else float("nan")
   dd_dev = math.sqrt((np.minimum(ret, 0) ** 2).mean())
   sortino = ret.mean() / dd_dev * math.sqrt(per_year) if dd_dev > 0 else float("nan")
   calmar = total / mdd if mdd > 0 else float("nan")
   print(f"Return {total * 100:+.2f}% | MaxDD {mdd * 100:.2f}% | "
         f"Sharpe~{sharpe:.2f} | Sortino~{sortino:.2f} | Calmar~{calmar:.2f}")
   print("(Approximate: the competition's own sampling/annualisation may differ.)")




# =========================================================
# ENTRYPOINT
# =========================================================
def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("--dry-run", action="store_true", help="virtual wallet, no orders sent")
   ap.add_argument("--report", action="store_true")
   args = ap.parse_args()
   if args.report:
       return report()


   key, secret = os.getenv("ROOSTOO_API_KEY"), os.getenv("ROOSTOO_API_SECRET")
   if not (key and secret) and not args.dry_run:
       sys.exit("Set ROOSTOO_API_KEY and ROOSTOO_API_SECRET")
   client = RoostooClient(key or "dry", secret or "dry", dry_run=args.dry_run)
   engine = Engine(client, DataFeed(), StateManager())
   logger.info(f"Bot started | dry_run={args.dry_run} | tf={Cfg.TIMEFRAME} | stop_mode={Cfg.STOP_MODE} | "
               f"end={datetime.fromtimestamp(Cfg.END_TS, timezone.utc):%Y-%m-%d %H:%M} UTC")


   while True:
       try:
           engine.tick()
       except KeyboardInterrupt:
           logger.info("Stopped by user (positions left open; state saved).")
           engine._save()
           break
       except Exception:
           logger.exception("Loop error")
       time.sleep(Cfg.POLL_SECONDS)




if __name__ == "__main__":
   main()
