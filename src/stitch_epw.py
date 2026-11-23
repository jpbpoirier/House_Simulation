"""Stitch a contiguous 365-day EPW from several EPWs, by priority (first file that has an hour wins);
gaps are filled from the typical-year file (CWEC) with its year replaced. Kiva foundations and the
water-mains correlation need a full year of weather, so every run uses a 365-day file.
    python3 src/stitch_epw.py 2026-01-01 out.epw first.epw second.epw ...
"""
import sys, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; W = ROOT / "weather"

def read(p):
    lines = Path(p).read_text().splitlines()
    return lines[:8], {(int(l.split(",")[0]),) + tuple(int(v) for v in l.split(",")[1:4]): l
                       for l in lines[8:] if l.strip()}

def stitch(start, out, sources, typical="Ottawa_CWEC_2020.epw", ndays=365):
    """start: first day (date or ISO string). Returns {source: hours used}."""
    start = dt.date.fromisoformat(str(start))
    hdr, _ = read(W / typical)
    recs = [read(W / s if not Path(s).is_absolute() else s)[1] for s in sources]
    typ = {k[1:]: v for k, v in read(W / typical)[1].items()}
    rows, origin, d, n = [], {}, start, 0
    while n < ndays:
        if not (d.month == 2 and d.day == 29):
            for h in range(1, 25):
                key = (d.year, d.month, d.day, h); line = None
                for s, r in zip(sources, recs):
                    if key in r:
                        line = r[key]; origin[Path(s).name] = origin.get(Path(s).name, 0) + 1; break
                if line is None:
                    f = typ[(d.month, d.day, h)].split(","); f[0] = str(d.year); line = ",".join(f)
                    origin[typical] = origin.get(typical, 0) + 1
                rows.append(line)
            n += 1
        d += dt.timedelta(1)
    last = d - dt.timedelta(1)
    hdr[0] = "LOCATION,KANATA (stitched),ON,CAN,ActualYear,716280,45.31,-75.90,-5.0,100.0"
    hdr[5] = "COMMENTS 1,Stitched " + "; ".join(f"{k}: {v} h" for k, v in origin.items())
    hdr[7] = f"DATA PERIODS,1,1,Data,{start.strftime('%A')}, {start.month}/{start.day}/{start.year}, {last.month}/{last.day}/{last.year}"
    out = W / out if not Path(out).is_absolute() else Path(out)
    out.write_text("\n".join(hdr + rows) + "\n")
    return origin

if __name__ == "__main__":
    a = sys.argv[1:]
    print(stitch(a[0], a[1], a[2:]))
