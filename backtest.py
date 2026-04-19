# Sticky Expectations Quality Tilt — backtest w/ vol-management twist
# pip install pandas numpy matplotlib yfinance && python backtest.py
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import ssl, urllib.request, zipfile

import matplotlib.pyplot as plt, matplotlib.dates as mdates
import numpy as np, pandas as pd, yfinance as yf

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
TICKERS = ["QUAL", "SPY", "BIL"]
OUT = Path(__file__).resolve().parent / "outputs"

@dataclass(frozen=True)
class Cfg:
    start: str = "2014-01-01"
    train_end: str = "2020-12-31"
    qw_range: tuple[float, ...] = tuple(np.round(np.arange(0.50, 1.01, 0.05), 2))
    lookbacks: tuple[int, ...] = (3, 6, 9, 12)

# --- metrics ---
def _ar(r, n=12): return (1+r).prod()**(n/len(r))-1 if len(r) else np.nan
def _av(r, n=12): return r.std()*np.sqrt(n) if len(r) else np.nan
def _sh(r, n=12): v=_av(r,n); return (r.mean()*n)/v if v and not np.isnan(v) else np.nan
def _md(r): eq=(1+r).cumprod(); return (eq/eq.cummax()-1).min() if len(r) else np.nan
def _cal(r): d=_md(r); return _ar(r)/abs(d) if d and not np.isnan(d) else np.nan
def _row(r, name): return {"Strategy":name,"Return":f"{_ar(r):+.2%}","Vol":f"{_av(r):.2%}","Sharpe":f"{_sh(r):.2f}","Max DD":f"{_md(r):.2%}","Calmar":f"{_cal(r):.2f}"}

# --- data ---
def get_ff5():
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(FF5_URL, context=ctx) as resp: payload = resp.read()
    with zipfile.ZipFile(BytesIO(payload)) as zf: raw = zf.read(zf.namelist()[0]).decode()
    lines = raw.splitlines()
    hdr = next(i for i,l in enumerate(lines) if l.startswith(",Mkt-RF"))
    rows = [lines[hdr]]
    for l in lines[hdr+1:]:
        c = l.split(",",1)[0].strip()
        if c.isdigit() and len(c)==6: rows.append(l)
        elif not l.strip() and len(rows)>1: break
    df = pd.read_csv(StringIO("\n".join(rows))).rename(columns={pd.read_csv(StringIO("\n".join(rows))).columns[0]:"date"})
    df["date"] = pd.PeriodIndex(df["date"].astype(str), freq="M").to_timestamp("M")
    df = df.set_index("date").astype(float)/100
    df["Market"] = df["Mkt-RF"]+df["RF"]
    return df

def get_prices(start):
    p = yf.download(TICKERS, start=start, auto_adjust=True, progress=False)["Close"]
    return p.sort_index().dropna(how="all")

def mret(prices): return prices.resample("ME").last().dropna(how="all").pct_change().dropna(how="all")

# --- strategies ---
def base(prices, qw, lb):
    # QUAL>MA -> blend QUAL+SPY, else BIL
    mp = prices.resample("ME").last().dropna(how="all")
    ret = mp.pct_change().dropna(how="all")
    on = (mp["QUAL"]>mp["QUAL"].rolling(lb).mean()).shift(1).reindex(ret.index).astype("boolean").fillna(False).astype(bool)
    s = pd.Series(index=ret.index, dtype=float)
    s[on] = qw*ret.loc[on,"QUAL"]+(1-qw)*ret.loc[on,"SPY"]
    s[~on] = ret.loc[~on,"BIL"]
    return s.dropna()

def vol_managed(prices, qw, lb):
    # same signal + inverse-vol position sizing (Moreira & Muir 2017)
    # high vol = attention elevated = mispricing corrects faster -> size down
    mp = prices.resample("ME").last().dropna(how="all")
    ret = mp.pct_change().dropna(how="all")
    on = (mp["QUAL"]>mp["QUAL"].rolling(lb).mean()).shift(1).reindex(ret.index).astype("boolean").fillna(False).astype(bool)
    rv = ret["QUAL"].rolling(lb).std()
    w = (rv.expanding().median().shift(1)/rv.shift(1)).clip(upper=1.0).reindex(ret.index).fillna(1.0)
    eq = qw*ret["QUAL"]+(1-qw)*ret["SPY"]
    s = pd.Series(index=ret.index, dtype=float)
    s[on] = w[on]*eq[on]+(1-w[on])*ret.loc[on,"BIL"]
    s[~on] = ret.loc[~on,"BIL"]
    return s.dropna(), w

# --- optimize ---
def optimize(prices, cfg):
    best, bqw, blb = -np.inf, 0.5, 9
    for qw in cfg.qw_range:
        for lb in cfg.lookbacks:
            s,_ = vol_managed(prices,qw,lb)
            sh = _sh(s.loc[:cfg.train_end])
            if sh > best: best, bqw, blb = sh, qw, lb
    return bqw, blb

# --- chart ---
CL = {"vm":"#e63946","base":"#1a1a2e","qual":"#4361ee","spy":"#7209b7","bil":"#adb5bd"}

def _ax(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2,ls="--"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y")); ax.xaxis.set_major_locator(mdates.YearLocator(2))

def plot(twist, b, bench, w, path):
    fig, axes = plt.subplots(3,1,figsize=(14,11),gridspec_kw={"height_ratios":[3,1,1]})
    # equity curves
    ax=axes[0]
    for nm,r,c,lw in [("Vol-Managed",twist,CL["vm"],2.5),("Base",b,CL["base"],1.8),("QUAL",bench["QUAL"],CL["qual"],1.3),("SPY",bench["SPY"],CL["spy"],1.3),("BIL",bench["BIL"],CL["bil"],1.0)]:
        eq=(1+r).cumprod(); ax.plot(eq.index,eq.values,label=nm,lw=lw,color=c,alpha=0.5 if nm=="BIL" else 0.85)
    ax.set_title("Sticky Expectations — Volatility-Managed Quality Tilt",fontsize=14,fontweight="bold",pad=12)
    ax.set_ylabel("Growth of $1"); _ax(ax); ax.legend(loc="upper left",fontsize=9)
    # drawdown
    ax2=axes[1]
    for nm,r,c in [("Vol-Managed",twist,CL["vm"]),("Base",b,CL["base"]),("QUAL",bench["QUAL"],CL["qual"]),("SPY",bench["SPY"],CL["spy"])]:
        eq=(1+r).cumprod(); dd=eq/eq.cummax()-1; ax2.plot(dd.index,dd.values,color=c,lw=1.2,alpha=0.7,label=nm)
        if nm in ("Vol-Managed","Base"): ax2.fill_between(dd.index,dd.values,0,alpha=0.2,color=c)
    ax2.set_ylabel("Drawdown"); ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}")); _ax(ax2); ax2.legend(loc="lower left",fontsize=8,ncol=2)
    # equity weight
    ax3=axes[2]; ws=w.reindex(twist.index).fillna(1.0)
    ax3.fill_between(ws.index,ws.values,0,alpha=0.35,color=CL["vm"]); ax3.plot(ws.index,ws.values,color=CL["vm"],lw=1)
    ax3.set_ylabel("Equity Weight"); ax3.set_ylim(0,1.05); ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.0%}"))
    ax3.set_title("Equity exposure scales down when vol spikes",fontsize=10,pad=8); _ax(ax3)
    fig.tight_layout(); fig.savefig(path,dpi=200,bbox_inches="tight"); plt.close(fig)

def plot_factor(ff5, path):
    fig,ax=plt.subplots(figsize=(12,5))
    ax.plot((1+ff5["RMW"]).cumprod(),label="RMW (profitability)",lw=2,color=CL["qual"])
    ax.plot((1+ff5["Market"]).cumprod(),label="US market",lw=1.8,alpha=0.7,color=CL["spy"])
    ax.set_title("Fama-French Profitability Factor",fontsize=13,fontweight="bold",pad=12)
    ax.set_ylabel("Growth of $1"); _ax(ax); ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(path,dpi=200,bbox_inches="tight"); plt.close(fig)

# --- main ---
if __name__ == "__main__":
    OUT.mkdir(exist_ok=True); cfg=Cfg()
    print("Downloading data..."); ff5=get_ff5(); prices=get_prices(cfg.start)
    print("Optimizing..."); qw,lb=optimize(prices,cfg); print(f"  best: qw={qw:.2f} lb={lb}")
    twist,w=vol_managed(prices,qw,lb); b=base(prices,qw,lb); bench=mret(prices)[list(TICKERS)]
    print(pd.DataFrame([_row(twist,"Vol-Managed"),_row(b,"Base"),_row(bench["QUAL"],"QUAL"),_row(bench["SPY"],"SPY"),_row(bench["BIL"],"BIL")]).set_index("Strategy").to_string())
    plot(twist,b,bench,w,OUT/"backtest.png"); plot_factor(ff5,OUT/"factor_evidence.png")
    print("Done. Charts in outputs/")
