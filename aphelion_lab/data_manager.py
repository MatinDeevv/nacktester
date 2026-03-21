"""
Aphelion Lab — Data Manager
MT5 multithreaded historical data download + local Parquet cache.
"""

import os, json, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Callable

import pandas as pd
import numpy as np

logger = logging.getLogger("aphelion.data")

MT5_TF = {}

def _init_mt5_tf():
    global MT5_TF
    try:
        import MetaTrader5 as mt5
        MT5_TF = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        }
    except ImportError:
        pass

SYMBOLS = ["XAUUSD","XAGUSD","EURUSD","GBPUSD","USDJPY","US500","XTIUSD","BTCUSD"]
TIMEFRAMES = ["M1","M5","M15","M30","H1","H4","D1"]
TF_YEARS = {"M1":2,"M5":5,"M15":10,"M30":10,"H1":10,"H4":10,"D1":10}
TF_MINUTES = {"M1":1,"M5":5,"M15":15,"M30":30,"H1":60,"H4":240,"D1":1440,"W1":10080}

@dataclass
class CacheEntry:
    symbol: str; timeframe: str; start: str; end: str
    bars: int; file: str; updated: str

class DataManager:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index: dict[str, CacheEntry] = {}
        self._load_index()
        self._mt5_ok = False
        self._symbol_map: dict[str, str] = {}

    def _load_index(self):
        f = self.cache_dir / "index.json"
        if f.exists():
            try:
                raw = json.loads(f.read_text())
                self.index = {k: CacheEntry(**v) for k, v in raw.items()}
            except: self.index = {}

    def _save_index(self):
        raw = {k: v.__dict__ for k, v in self.index.items()}
        (self.cache_dir / "index.json").write_text(json.dumps(raw, indent=2))

    def _key(self, s, tf): return f"{s}_{tf}"
    def _path(self, s, tf): return self.cache_dir / f"{s}_{tf}.parquet"

    def init_mt5(self) -> bool:
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            _init_mt5_tf()
            self._mt5_ok = True
            info = mt5.account_info()
            logger.info(f"MT5 OK: {info.server} #{info.login}" if info else "MT5 connected")
            return True
        except ImportError:
            logger.error("MetaTrader5 not installed")
            return False

    def _resolve_symbol(self, requested_symbol: str) -> Optional[str]:
        """Resolve requested symbol to an actual broker symbol in Market Watch."""
        import MetaTrader5 as mt5

        if requested_symbol in self._symbol_map:
            return self._symbol_map[requested_symbol]

        # Exact match first.
        info = mt5.symbol_info(requested_symbol)
        if info is not None:
            if not info.visible:
                if not mt5.symbol_select(requested_symbol, True):
                    logger.warning(f"Could not select symbol in Market Watch: {requested_symbol}")
            self._symbol_map[requested_symbol] = requested_symbol
            return requested_symbol

        symbols = mt5.symbols_get()
        if not symbols:
            logger.error(f"MT5 symbols_get failed while resolving {requested_symbol}: {mt5.last_error()}")
            return None

        names = [s.name for s in symbols]
        target = requested_symbol.upper()

        # Try common broker suffix/prefix forms (e.g., XAUUSDm, .XAUUSD, XAUUSD.r).
        candidates = [n for n in names if n.upper() == target]
        if not candidates:
            candidates = [n for n in names if n.upper().startswith(target)]
        if not candidates:
            candidates = [n for n in names if target in n.upper()]

        if not candidates:
            logger.error(f"Symbol {requested_symbol} not found in broker symbols")
            return None

        resolved = sorted(candidates, key=len)[0]
        info = mt5.symbol_info(resolved)
        if info is not None and not info.visible:
            if not mt5.symbol_select(resolved, True):
                logger.warning(f"Could not select resolved symbol in Market Watch: {resolved}")
        self._symbol_map[requested_symbol] = resolved
        logger.info(f"Resolved symbol {requested_symbol} -> {resolved}")
        return resolved

    def _download_one(self, sym: str, tf: str, start: datetime, end: datetime):
        import MetaTrader5 as mt5
        tf_c = MT5_TF.get(tf)
        if tf_c is None:
            logger.error(f"Unsupported timeframe: {tf}")
            return None

        resolved_sym = self._resolve_symbol(sym)
        if resolved_sym is None:
            return None

        # MT5 Python API expects naive UTC datetime values.
        start_utc = start.astimezone(timezone.utc).replace(tzinfo=None) if start.tzinfo else start
        end_utc = end.astimezone(timezone.utc).replace(tzinfo=None) if end.tzinfo else end

        rates = mt5.copy_rates_range(resolved_sym, tf_c, start_utc, end_utc)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            # Some MT5 terminals return "Invalid params" for range queries.
            # Fallback to position-based fetch and then filter by date locally.
            if err and err[0] == -2:
                minutes = TF_MINUTES.get(tf, 60)
                total_minutes = int((end_utc - start_utc).total_seconds() / 60)
                estimated = int(total_minutes / max(minutes, 1))
                term_info = mt5.terminal_info()
                maxbars = max(2000, int(getattr(term_info, "maxbars", 100000) or 100000))
                initial = min(maxbars - 1, max(2000, estimated))
                retry_counts = [initial, 50000, 20000, 10000, 5000, 2000]

                for count in dict.fromkeys(c for c in retry_counts if c > 0):
                    rates = mt5.copy_rates_from_pos(resolved_sym, tf_c, 0, count)
                    if rates is None or len(rates) == 0:
                        continue

                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    df = df.rename(columns={"time": "timestamp", "tick_volume": "volume"})
                    df = df.set_index("timestamp").sort_index()
                    df = df[~df.index.duplicated(keep="first")]
                    start_ts = pd.Timestamp(start_utc, tz="UTC")
                    end_ts = pd.Timestamp(end_utc, tz="UTC")
                    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
                    if len(df) > 0:
                        return df

                err = mt5.last_error()

            logger.error(
                f"No rates for {sym} ({resolved_sym}) {tf} {start_utc.date()}->{end_utc.date()} | mt5.last_error={err}"
            )
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(columns={"time":"timestamp","tick_volume":"volume"})
        df = df.set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        return df

    def download_all(self, symbols=None, timeframes=None, workers=4, on_progress=None):
        if not self._mt5_ok and not self.init_mt5(): return {}
        symbols = symbols or SYMBOLS
        timeframes = timeframes or TIMEFRAMES
        now = datetime.utcnow()
        jobs = []
        for s in symbols:
            for tf in timeframes:
                yrs = TF_YEARS.get(tf, 10)
                jobs.append((s, tf, now - timedelta(days=yrs*365), now))
        total = len(jobs)
        done = 0
        results = {}
        if on_progress: on_progress(0, total, "Starting...")

        # MetaTrader5 Python API is not reliably thread-safe for concurrent reads.
        workers = 1

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self._download_one, s, tf, st, en): (s, tf) for s, tf, st, en in jobs}
            for fut in as_completed(futs):
                s, tf = futs[fut]
                k = self._key(s, tf)
                done += 1
                try:
                    df = fut.result()
                    if df is not None and len(df) > 0:
                        p = self._path(s, tf)
                        df.to_parquet(p, engine="pyarrow")
                        self.index[k] = CacheEntry(
                            symbol=s, timeframe=tf,
                            start=str(df.index[0].date()), end=str(df.index[-1].date()),
                            bars=len(df), file=str(p),
                            updated=str(datetime.now(timezone.utc)),
                        )
                        results[k] = len(df)
                    else:
                        results[k] = 0
                except Exception as e:
                    logger.error(f"{k}: {e}")
                    results[k] = 0
                if on_progress:
                    on_progress(done, total, f"{s} {tf}: {results.get(k,0)} bars")

        self._save_index()
        return results

    def load(self, symbol: str, timeframe: str, start=None, end=None) -> Optional[pd.DataFrame]:
        p = self._path(symbol, timeframe)
        if not p.exists(): return None
        df = pd.read_parquet(p)
        if start: df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end: df = df[df.index <= pd.Timestamp(end, tz="UTC")]
        return df if len(df) > 0 else None

    def get_cached(self) -> list[CacheEntry]:
        return list(self.index.values())

    def get_symbols(self) -> list[str]:
        return sorted(set(e.symbol for e in self.index.values()))

    def get_timeframes(self, symbol: str) -> list[str]:
        return [e.timeframe for e in self.index.values() if e.symbol == symbol]

    def is_cached(self, s, tf) -> bool: return self._path(s, tf).exists()

    def cache_size_mb(self) -> float:
        return sum(f.stat().st_size for f in self.cache_dir.glob("*.parquet")) / 1048576
