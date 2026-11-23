"""Pull hourly meters / variables from an EnergyPlus SQLite output into a DataFrame (LST)."""
import sqlite3, pandas as pd

def hourly(sql_path, names):
    con = sqlite3.connect(sql_path)
    q = """SELECT t.Year, t.Month, t.Day, t.Hour, d.Name, d.KeyValue, r.Value
           FROM ReportData r JOIN ReportDataDictionary d USING(ReportDataDictionaryIndex)
           JOIN Time t USING(TimeIndex) WHERE d.ReportingFrequency='Hourly' AND t.EnvironmentPeriodIndex=
           (SELECT MAX(EnvironmentPeriodIndex) FROM EnvironmentPeriods) AND d.Name IN (%s)""" % ",".join("?" * len(names))
    df = pd.read_sql(q, con, params=names); con.close()
    yr = df.Year.where(df.Year > 0, 2025)
    df["ts"] = pd.to_datetime(dict(year=yr, month=df.Month, day=df.Day)) + pd.to_timedelta(df.Hour - 1, "h")
    df["col"] = df.Name + df.KeyValue.map(lambda k: "" if (k is None or k == "" or k != k) else f"[{k}]")
    return df.pivot_table(index="ts", columns="col", values="Value", aggfunc="sum")
