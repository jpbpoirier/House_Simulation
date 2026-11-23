"""All-in electricity (Hydro Ottawa, RPP Time-of-Use) and natural gas (Enbridge Rate 1) costs.

Sources (checked 2026-11-22):
- TOU prices/periods: OEB RPP, effective Nov 1 2026 (Hydro One / insauga reports; hydroottawa.com/residential/time-of-use)
  and Nov 1 2025 term (J-P's Hydro Ottawa bill 2026-10-23: 9.8 / 15.7 / 20.3 c/kWh).
- Hydro Ottawa delivery & regulatory (hydroottawa.com/residential/time-of-use, daily charges since Jun 1 2026).
- OER: 23.5% (Nov 1 2025 - Oct 31 2026), 15.7% from Nov 1 2026. HST 13% on charges before the OER.
- Enbridge Rate 1: J-P's bill for Sep 26 - Oct 27 2026 (rates from Oct 1 2026). Delivery block rates are FITTED
  from invoice history (not read from the rate sheet) -> approximate.
All timestamps in local standard time (EST); TOU periods follow local clock time, so DST is applied here.
"""
import datetime as dt
import pandas as pd

HST = 0.13

# ---------------- electricity
ELEC_TERMS = [  # (start date, {tier: $/kWh}, OER)
    (dt.date(2025, 11, 1), {"off": 0.098, "mid": 0.157, "on": 0.203}, 0.235),
    (dt.date(2026, 11, 1), {"off": 0.091, "mid": 0.146, "on": 0.189}, 0.157),
]
HO = dict(transmission=0.0216, low_voltage=0.00007, other_rpp=-0.0003, line_loss=0.0332,
          regulatory=0.0053,                                   # wholesale system admin & reliability, $/kWh
          fixed_day=1.2888 - 0.0391 + 0.05918 + 0.0138 + 0.0082)  # service, temp, forgone rev, SME, admin fee

def _nth_weekday(y, m, wd, n):
    d = dt.date(y, m, 1); d += dt.timedelta((wd - d.weekday()) % 7); return d + dt.timedelta(7 * (n - 1))

def _easter(y):  # anonymous Gregorian algorithm
    a, b, c = y % 19, y // 100, y % 100; d, e = b // 4, b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30; i, k = c // 4, c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451; mo = (h + l - 7 * m + 114) // 31; da = (h + l - 7 * m + 114) % 31 + 1
    return dt.date(y, mo, da)

def tou_holidays(y):
    """Ontario TOU off-peak holidays (observed dates move to the next weekday when they fall on a weekend)."""
    def obs(d):
        while d.weekday() >= 5: d += dt.timedelta(1)
        return d
    vic = dt.date(y, 5, 24); vic -= dt.timedelta((vic.weekday() - 0) % 7)
    xmas, box = obs(dt.date(y, 12, 25)), dt.date(y, 12, 26)
    box = obs(box if box != xmas else box + dt.timedelta(1)); box = box if box != xmas else box + dt.timedelta(1)
    return {obs(dt.date(y, 1, 1)), _nth_weekday(y, 2, 0, 3), _easter(y) - dt.timedelta(2), vic,
            obs(dt.date(y, 7, 1)), _nth_weekday(y, 8, 0, 1), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 10, 0, 2),
            xmas, box}

def tou_tier(ts_lst):
    """TOU tier for an hour starting at ts_lst (EST). Periods are in local clock time."""
    t = pd.Timestamp(ts_lst).tz_localize("Etc/GMT+5").tz_convert("America/Toronto")
    d, h = t.date(), t.hour
    if d.weekday() >= 5 or d in tou_holidays(d.year) or h < 7 or h >= 19:
        return "off"
    winter = d.month >= 11 or d.month <= 4
    peak = (7 <= h < 11) or (17 <= h < 19)
    return ("on" if peak else "mid") if winter else ("mid" if peak else "on")

def _term(d):
    t = [x for x in ELEC_TERMS if x[0] <= d]
    return t[-1] if t else ELEC_TERMS[0]

def elec_cost(kwh):
    """kwh: Series of hourly kWh indexed by hour-start (EST). Returns DataFrame with tier, commodity and all-in $."""
    df = pd.DataFrame({"kwh": kwh})
    df["tier"] = [tou_tier(t) for t in df.index]
    terms = [_term(t.date()) for t in df.index]
    df["price"] = [p[1][tier] for p, tier in zip(terms, df.tier)]
    df["oer"] = [p[2] for p in terms]
    df["commodity"] = df.kwh * df.price
    var = df.commodity * (1 + HO["line_loss"]) + df.kwh * (HO["transmission"] + HO["low_voltage"] + HO["other_rpp"] + HO["regulatory"])
    fixed = HO["fixed_day"] / 24
    df["allin"] = (var + fixed) * (1 + HST - df.oer)          # HST on pre-rebate charges, OER credited before tax
    return df

def elec_marginal_cents(tier, d=dt.date(2026, 11, 1)):
    p, oer = _term(d)[1][tier], _term(d)[2]
    v = p * (1 + HO["line_loss"]) + HO["transmission"] + HO["low_voltage"] + HO["other_rpp"] + HO["regulatory"]
    return 100 * v * (1 + HST - oer)

# ---------------- natural gas (Enbridge Rate 1, from Oct 1 2026)
GAS = dict(customer_month=27.69, supply=0.094741, pef=1.00830, transport=4.38 / 80, cost_adj=0.38 / 80,
           # delivery $/m3 by monthly block (0-30, 30-85, 85-170, >170), fitted to invoices; approximate
           delivery_blocks=[(30, 0.1440), (85, 0.1350), (170, 0.1300), (1e9, 0.1260)])

def gas_month_cost(m3, months=1.0):
    """All-in Enbridge charge for one billing period (incl. HST). Customer charge is per bill (monthly)."""
    left, prev, deliv = m3, 0, 0.0
    for top, rate in GAS["delivery_blocks"]:
        q = max(0.0, min(left, top - prev)); deliv += q * rate; left -= q; prev = top
    var = m3 * (GAS["supply"] * GAS["pef"] + GAS["transport"] + GAS["cost_adj"]) + deliv
    return (var + GAS["customer_month"] * months) * (1 + HST)

def gas_marginal_cents(monthly_m3=150):
    rate = next(r for top, r in GAS["delivery_blocks"] if monthly_m3 <= top)
    return 100 * (rate + GAS["supply"] * GAS["pef"] + GAS["transport"] + GAS["cost_adj"]) * (1 + HST)

if __name__ == "__main__":
    hh = pd.read_csv("data/clean/hydro_hourly.csv", index_col=0, parse_dates=True).kwh
    # Hydro Ottawa bill 2026-10-23: Sep 8 - Oct 8 2026 (meter reads are local dates; use LST hours)
    b = elec_cost(hh.loc["2026-09-08":"2026-10-07 23:00"])
    print("Bill check Sep 8-Oct 8 2026: kWh %.1f (bill 379.12), commodity $%.2f (bill 47.62), all-in $%.2f (bill 89.16)"
          % (b.kwh.sum(), b.commodity.sum(), b.allin.sum()))
    print("  by tier kWh:", b.groupby("tier").kwh.sum().round(1).to_dict(), "(bill off 247.77, mid 72.24, on 59.11)")
    print("Enbridge check Sep 26-Oct 27 2026, 80 m3: $%.2f (bill 57.76 before the one-off rate adjustment)" % gas_month_cost(80))
    for t in ["off", "mid", "on"]:
        print(f"Marginal all-in electricity from Nov 1 2026, {t}: {elec_marginal_cents(t):.2f} c/kWh")
    print("Fixed electricity: $%.3f/day all-in" % (HO['fixed_day'] * (1 + HST - 0.157)))
    print("Marginal all-in gas (winter month >170 m3): %.1f c/m3; customer charge $%.2f/month all-in" % (gas_marginal_cents(200), GAS['customer_month'] * 1.13))
