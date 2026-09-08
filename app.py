import sqlite3
import os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from datetime import timedelta, datetime
from zoneinfo import ZoneInfo

# Page Configuration
st.set_page_config(
    page_title="Smartwatch Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Personal Health Dashboard")
st.markdown("Automated metrics extracted directly from your Gadgetbridge database with advanced sleep and HRV analysis.")

DB_FILE = "Gadgetbridge"


def _db_mtime():
    """st.cache_data has no idea when the underlying sqlite file changes - without this,
    a re-sync of the band (new sleep/activity data) would keep showing old cached results
    for the lifetime of the running Streamlit process. Passing the file's mtime into the
    cached functions below makes Streamlit re-read the DB automatically whenever it changes."""
    try:
        return os.path.getmtime(DB_FILE)
    except OSError:
        return 0


def get_local_tz(tz_name):
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("UTC")


def to_local(series_or_ts, unit, tz):
    """Parse epoch values as UTC, then convert to `tz` (DST-aware) and drop back to a
    naive datetime so the rest of the app's plain datetime comparisons keep working.
    `tz` is always passed explicitly (never read from a global) so that cached functions
    correctly invalidate when the timezone setting changes."""
    dt_utc = pd.to_datetime(series_or_ts, unit=unit, errors='coerce', utc=True)
    return dt_utc.dt.tz_convert(tz).dt.tz_localize(None)


# Session-state defaults. Pre-seeded here (rather than set via widget defaults) so their
# values are available before the widgets that control them are actually rendered further
# down the sidebar - several are needed early, to load the data the date list depends on.
_DEFAULTS = {
    "tz_name": "Europe/London",
    "max_hr_override": 190,
    "resting_hr_override": 0,
    "strain_sensitivity": 1.0,
    "sleep_table_override": "Auto-Detect",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

LOCAL_TZ = get_local_tz(st.session_state.tz_name)


# 1. Database Schema Inspector
@st.cache_data
def get_db_schema(mtime):
    schema = {}
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            for t in tables:
                cursor.execute(f'PRAGMA table_info("{t}")')
                schema[t] = [row[1].upper() for row in cursor.fetchall()]
    except Exception:
        pass
    return schema


schema = get_db_schema(_db_mtime())

if not schema:
    st.error(f"Could not connect to '{DB_FILE}'. Ensure your Gadgetbridge database file is in the folder!")
    st.stop()


# 2. Intelligent Table Detection
def find_best_activity_table():
    candidates = []
    for t, cols in schema.items():
        score = 0
        if any(c in cols for c in ["HEART_RATE", "HR"]):
            score += 15
        if any(c in cols for c in ["STEPS", "STEP"]):
            score += 15
        if "SAMPLE" in t.upper() and "SUMMARY" not in t.upper() and "SLEEP" not in t.upper():
            score += 10
        if score > 0:
            try:
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'SELECT COUNT(*) FROM "{t}"')
                    cnt = cursor.fetchone()[0]
                    if cnt > 0:
                        candidates.append((t, score + min(cnt, 1000)))
            except Exception:
                pass
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return list(schema.keys())[0]


auto_detected_table = find_best_activity_table()
all_tables = list(schema.keys())
if "activity_table_override" not in st.session_state:
    st.session_state.activity_table_override = auto_detected_table
if st.session_state.activity_table_override not in all_tables:
    st.session_state.activity_table_override = auto_detected_table
activity_table = st.session_state.activity_table_override

sleep_tables = [t for t in schema.keys() if "SLEEP" in t.upper()]
auto_detected_sleep_table = next((t for t in sleep_tables if "STAGE" in t.upper()), sleep_tables[0] if sleep_tables else None)
if st.session_state.sleep_table_override != "Auto-Detect":
    active_sleep_table = st.session_state.sleep_table_override
else:
    active_sleep_table = auto_detected_sleep_table


# Load Data Helper
@st.cache_data
def load_table(table_name, mtime):
    try:
        with sqlite3.connect(DB_FILE) as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            df.columns = df.columns.str.upper()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data
def get_cleaned_activity_df(table_name, mtime, tz_name):
    """Load + standardize the activity table once, cached. Used both for the main dashboard
    and (internally) by compute_sleep_and_hrv for every date in the sidebar list, so this only
    actually re-runs when the table, the file, or the timezone changes."""
    tz = get_local_tz(tz_name)
    df = load_table(table_name, mtime) if table_name else pd.DataFrame()
    if df.empty:
        return df

    time_col = next((c for c in ["TIMESTAMP", "TIME", "DATE", "RAW_TIMESTAMP", "START_TIME"] if c in df.columns), None)
    if time_col:
        if pd.api.types.is_numeric_dtype(df[time_col]):
            df['Datetime'] = to_local(df[time_col], unit='ms' if df[time_col].max() > 2e10 else 's', tz=tz)
        else:
            df['Datetime'] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=['Datetime']).sort_values('Datetime').reset_index(drop=True)
        df['Date'] = df['Datetime'].dt.date

    hr_col = next((c for c in ["HEART_RATE", "HR", "RATE"] if c in df.columns), None)
    if hr_col:
        df['HEART_RATE'] = pd.to_numeric(df[hr_col], errors='coerce')
        df.loc[(df['HEART_RATE'] >= 255) | (df['HEART_RATE'] <= 30), 'HEART_RATE'] = np.nan
    else:
        df['HEART_RATE'] = np.nan

    step_col = next((c for c in df.columns if any(k in c for k in ["STEPS", "STEP"]) and "SLEEP" not in c), None)
    if step_col:
        df['STEPS'] = pd.to_numeric(df[step_col], errors='coerce')
        # Some firmwares use 65535/-1/255 as a "no data" sentinel for this minute rather than 0 steps.
        df.loc[(df['STEPS'] > 500) | (df['STEPS'] < 0), 'STEPS'] = np.nan
        df['STEPS'] = df['STEPS'].fillna(0)
    else:
        df['STEPS'] = 0

    return df


df_activity = get_cleaned_activity_df(activity_table, _db_mtime(), st.session_state.tz_name) if activity_table else pd.DataFrame()


# Sleep + HRV engine. Runs per-date so it can power both the main dashboard and the
# sidebar's per-day HRV list.

MIN_SLEEP_BLOCK_HOURS = 2.0


def detect_sleep_from_raw_kind(activity_df, w_start, w_end, baseline_hr):
    """Find the sleep session overnight from per-minute activity-kind codes. Far more reliable
    on Huami/Xiaomi devices than a raw 'SLEEP' column, which can be a noisy counter rather than
    a boolean. Doesn't hardcode which code means sleep - it varies by device.

    A single longest-contiguous-block approach undercounts real sleep: a brief nighttime
    awakening (bathroom trip, checking the time) splits one sleep session into two blocks, and
    picking only the longer one silently drops the other, reporting the wrong start or end time.
    This instead finds the best "core" block, then merges in any other resting-HR block within
    a short gap of it, regardless of its exact code, extending the session outward."""
    if 'RAW_KIND' not in activity_df.columns or 'Datetime' not in activity_df.columns:
        return None
    window_df = activity_df[(activity_df['Datetime'] >= w_start) & (activity_df['Datetime'] <= w_end)].sort_values('Datetime').reset_index(drop=True)
    if window_df.empty:
        return None

    block_ids = (window_df['RAW_KIND'] != window_df['RAW_KIND'].shift()).cumsum()
    blocks = []
    for _, g in window_df.groupby(block_ids):
        gaps = g['Datetime'].diff().dt.total_seconds().dropna() / 60.0
        sample_width = gaps.median() if not gaps.empty else 1.0
        span_min = (g['Datetime'].max() - g['Datetime'].min()).total_seconds() / 60.0 + (sample_width if pd.notna(sample_width) else 1.0)
        avg_hr = g['HEART_RATE'].mean()
        is_resting = pd.isna(baseline_hr) or pd.isna(avg_hr) or avg_hr <= baseline_hr
        blocks.append({"raw_kind": g['RAW_KIND'].iloc[0], "start": g['Datetime'].min(), "end": g['Datetime'].max(),
                        "span_min": span_min, "avg_hr": avg_hr, "n": len(g), "resting": is_resting})

    candidates = [b for b in blocks if b["span_min"] >= MIN_SLEEP_BLOCK_HOURS * 60 and b["resting"]]
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c["span_min"])
    core = candidates[0]

    MERGE_GAP_MINUTES = 60          # brief awakenings within this gap get folded into the same session
    MAX_FOREIGN_BLOCK_MINUTES = 15  # a different-code block only merges if it's itself this short -
                                     # a brief transitional blip, not a separate resting-but-awake period
    merged_start, merged_end = core["start"], core["end"]
    used = {id(core)}
    changed = True
    while changed:
        changed = False
        for b in blocks:
            if id(b) in used or not b["resting"]:
                continue
            if b["raw_kind"] != core["raw_kind"] and b["span_min"] > MAX_FOREIGN_BLOCK_MINUTES:
                continue
            gap_before = (merged_start - b["end"]).total_seconds() / 60.0
            gap_after = (b["start"] - merged_end).total_seconds() / 60.0
            if 0 <= gap_before <= MERGE_GAP_MINUTES:
                merged_start = min(merged_start, b["start"])
                used.add(id(b))
                changed = True
            elif 0 <= gap_after <= MERGE_GAP_MINUTES:
                merged_end = max(merged_end, b["end"])
                used.add(id(b))
                changed = True

    merged_rows = window_df[(window_df['Datetime'] >= merged_start) & (window_df['Datetime'] <= merged_end)]
    return {"raw_kind": core["raw_kind"], "start": merged_start, "end": merged_end,
            "span_min": (merged_end - merged_start).total_seconds() / 60.0,
            "avg_hr": merged_rows['HEART_RATE'].mean() if not merged_rows.empty else core["avg_hr"],
            "n": len(merged_rows)}


@st.cache_data
def compute_sleep_and_hrv(target_date, activity_table, active_sleep_table, mtime, tz_name):
    """Everything needed to report sleep + HRV for a single date. Cached per (date, settings)
    so the sidebar can call this once per day in the history list without recomputing on every
    rerun, and the main dashboard reuses the exact same cached result for the selected date."""
    tz = get_local_tz(tz_name)
    activity_df = get_cleaned_activity_df(activity_table, mtime, tz_name) if activity_table else pd.DataFrame()

    result = {
        "sleep_duration_mins": 0,
        "sleep_start_str": "N/A", "sleep_end_str": "N/A",
        "sleep_window_start": None, "sleep_window_end": None,
        "sleep_hr_median": np.nan,
        "hrv_val": np.nan, "hrv_source": "none",
        "sleep_diag": {"source": "none", "detail": "No sleep table configured."},
        "window_start": None, "window_end": None,
    }

    window_start = datetime.combine(target_date - timedelta(days=1), datetime.min.time()) + timedelta(hours=16)
    window_end = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=15)
    result["window_start"], result["window_end"] = window_start, window_end

    sleep_diag = result["sleep_diag"]
    sleep_hr_data = pd.Series(dtype=float)

    if active_sleep_table:
        df_sleep = load_table(active_sleep_table, mtime)
        if not df_sleep.empty:
            df_sleep = df_sleep.copy()
            df_sleep.columns = df_sleep.columns.str.upper()
            time_c = next((c for c in df_sleep.columns if any(k in c for k in ["TIMESTAMP", "TIME", "START", "DATE", "FROM"])), None)
            sleep_diag["table"] = active_sleep_table
            sleep_diag["total_rows"] = len(df_sleep)

            if time_c:
                try:
                    max_v = pd.to_numeric(df_sleep[time_c], errors='coerce').max()
                    unit = 'ms' if max_v > 2e10 else 's'
                    df_sleep['Sleep_DT'] = to_local(df_sleep[time_c], unit=unit, tz=tz)
                    sleep_diag["table_min_dt"] = str(df_sleep['Sleep_DT'].min())
                    sleep_diag["table_max_dt"] = str(df_sleep['Sleep_DT'].max())

                    day_sleep = df_sleep[(df_sleep['Sleep_DT'] >= window_start) & (df_sleep['Sleep_DT'] <= window_end)]
                    if day_sleep.empty:
                        latest_ts = df_sleep['Sleep_DT'].max()
                        if pd.notna(latest_ts):
                            day_sleep = df_sleep[(df_sleep['Sleep_DT'] >= latest_ts - timedelta(hours=24)) & (df_sleep['Sleep_DT'] <= latest_ts)]
                            if not day_sleep.empty:
                                sleep_diag["note"] = "Selected day had no rows in this table; showing the most recent night on record instead."

                    if not day_sleep.empty:
                        day_sleep = day_sleep.sort_values('Sleep_DT').reset_index(drop=True)
                        result["sleep_start_str"] = day_sleep['Sleep_DT'].min().strftime('%H:%M')
                        result["sleep_end_str"] = day_sleep['Sleep_DT'].max().strftime('%H:%M')

                        ts = day_sleep['Sleep_DT']
                        gaps_min = (ts.shift(-1) - ts).dt.total_seconds() / 60.0
                        valid_gaps = gaps_min.dropna()
                        valid_gaps = valid_gaps[valid_gaps > 0]
                        median_gap = valid_gaps.median() if not valid_gaps.empty else 1.0
                        if pd.isna(median_gap) or median_gap <= 0:
                            median_gap = 1.0
                        gaps_min = gaps_min.fillna(median_gap).clip(lower=0, upper=120)
                        result["sleep_duration_mins"] = float(gaps_min.sum())

                        s_min, e_max = day_sleep['Sleep_DT'].min(), day_sleep['Sleep_DT'].max()
                        result["sleep_window_start"], result["sleep_window_end"] = s_min, e_max
                        if not activity_df.empty and 'Datetime' in activity_df.columns and 'HEART_RATE' in activity_df.columns:
                            sleep_hr_data = activity_df[(activity_df['Datetime'] >= s_min) & (activity_df['Datetime'] <= e_max)]['HEART_RATE'].dropna()
                            if not sleep_hr_data.empty:
                                result["hrv_source"] = "sleep"
                        sleep_diag["source"] = "stage_table"
                        sleep_diag["rows_used"] = len(day_sleep)
                    else:
                        sleep_diag["detail"] = f"'{active_sleep_table}' has {len(df_sleep)} total rows, but none fall in the overnight window or the most recent 24h."
                except Exception as e:
                    sleep_diag["detail"] = f"Error parsing '{active_sleep_table}': {e}"
        else:
            sleep_diag["detail"] = f"'{active_sleep_table}' returned 0 rows."

    if result["sleep_duration_mins"] == 0 and not activity_df.empty and 'Datetime' in activity_df.columns:
        day_baseline_hr = activity_df['HEART_RATE'].mean() if 'HEART_RATE' in activity_df.columns else np.nan
        block = detect_sleep_from_raw_kind(activity_df, window_start, window_end, day_baseline_hr)
        if block:
            result["sleep_duration_mins"] = block["span_min"]
            result["sleep_start_str"] = block["start"].strftime('%H:%M')
            result["sleep_end_str"] = block["end"].strftime('%H:%M')
            result["sleep_window_start"], result["sleep_window_end"] = block["start"], block["end"]
            window_df_block = activity_df[(activity_df['Datetime'] >= block["start"]) & (activity_df['Datetime'] <= block["end"])]
            if 'HEART_RATE' in window_df_block.columns:
                sleep_hr_data = window_df_block['HEART_RATE'].dropna()
                if not sleep_hr_data.empty:
                    result["hrv_source"] = "sleep"
            sleep_diag["source"] = "activity_table_raw_kind"
            sleep_diag["raw_kind_detected"] = int(block["raw_kind"]) if pd.notna(block["raw_kind"]) else None
            sleep_diag["raw_kind_block_avg_hr"] = round(float(block["avg_hr"]), 1) if pd.notna(block["avg_hr"]) else None
            sleep_diag["note"] = (f"Dedicated sleep-stage table had no data for this night, so this comes from the "
                                   f"RAW_KIND={sleep_diag['raw_kind_detected']} activity-kind block on the activity table "
                                   f"(avg {sleep_diag['raw_kind_block_avg_hr']} bpm vs {day_baseline_hr:.0f} bpm day average).")
        else:
            general_col = next((c for c in activity_df.columns if c == "SLEEP"), None)
            if general_col:
                window_df = activity_df[(activity_df['Datetime'] >= window_start) & (activity_df['Datetime'] <= window_end)]
                if not window_df.empty:
                    flag_numeric = pd.to_numeric(window_df[general_col], errors='coerce').fillna(0)
                    asleep = window_df[flag_numeric != 0]
                    sleep_diag["activity_sleep_flag_unique_values"] = sorted(flag_numeric.unique().tolist())
                    if not asleep.empty:
                        span_hours = (asleep['Datetime'].max() - asleep['Datetime'].min()).total_seconds() / 3600.0
                        flagged_minutes = len(asleep)
                        coverage_pct = round(100 * flagged_minutes / max(1, len(window_df)), 1)
                        sleep_diag["fallback_span_hours"] = round(span_hours, 1)
                        sleep_diag["fallback_flagged_minutes"] = flagged_minutes
                        sleep_diag["fallback_coverage_pct_of_window"] = coverage_pct
                        if span_hours > 14 or coverage_pct < 80:
                            sleep_diag["source"] = "activity_table_fallback_rejected"
                            sleep_diag["note"] = (f"No clean RAW_KIND sleep block found, and the SLEEP flag was set across a "
                                                   f"{span_hours:.1f}-hour span at only {coverage_pct}% coverage - too gappy to trust.")
                        else:
                            result["sleep_duration_mins"] = flagged_minutes
                            result["sleep_start_str"] = asleep['Datetime'].min().strftime('%H:%M')
                            result["sleep_end_str"] = asleep['Datetime'].max().strftime('%H:%M')
                            result["sleep_window_start"], result["sleep_window_end"] = asleep['Datetime'].min(), asleep['Datetime'].max()
                            if 'HEART_RATE' in asleep.columns:
                                sleep_hr_data = asleep['HEART_RATE'].dropna()
                                if not sleep_hr_data.empty:
                                    result["hrv_source"] = "sleep"
                            sleep_diag["source"] = "activity_table_fallback_total_only"
                            sleep_diag["note"] = "Total-only estimate from the activity table's SLEEP flag - treat as approximate."

    # HRV (and RHR, and therefore Recovery Score) are only ever computed from a real detected
    # sleep window. No daytime resting-proxy fallback: a day with no sleep detected should show
    # "N/A", not a number quietly estimated from arbitrary low daytime readings.
    if not sleep_hr_data.empty:
        result["sleep_hr_median"] = float(sleep_hr_data.median())

    if not sleep_hr_data.empty and len(sleep_hr_data) > 5:
        rr_intervals = 60000.0 / sleep_hr_data
        diffs = rr_intervals.diff().dropna()
        if not diffs.empty:
            raw_rmssd = np.sqrt(np.mean(diffs ** 2))
            result["hrv_val"] = raw_rmssd * 1.35  # wrist optical-to-ECG calibration scale factor

    result["sleep_diag"] = sleep_diag
    return result


STRAIN_ACTIVITY_GATE = 15  # bpm above resting before a minute counts toward strain load


@st.cache_data
def compute_strain_for_date(target_date, activity_table, active_sleep_table, mtime, tz_name,
                             max_hr_override, resting_hr_override, sensitivity):
    """Day Strain (0-21): duration-weighted heart-rate-reserve load via Banister TRIMP.
    Whoop's own algorithm is proprietary and unpublished. Resting HR defaults to the night's
    actual overnight HR; minutes are smoothed and gated so ordinary daytime drift doesn't count
    as load; and "today" starts at wake time rather than midnight, matching how Whoop defines a
    day (sleep-to-sleep), since evening activity before falling asleep otherwise bleeds into the
    next day's score.

    The raw TRIMP total is mapped onto the 0-21 scale with a saturating exponential
    (21*(1-e^-trimp/T)) rather than a log curve: this stays close to proportional through
    low-to-moderate effort, so a light day and a heavy day are actually distinguishable, and
    only compresses hard as it nears the ceiling - reaching 20+ still takes a genuinely
    exceptional day, not just "somewhat more than a 10.\""""
    sleep_info = compute_sleep_and_hrv(target_date, activity_table, active_sleep_table, mtime, tz_name)
    activity_df = get_cleaned_activity_df(activity_table, mtime, tz_name) if activity_table else pd.DataFrame()
    day_df = activity_df[activity_df['Date'] == target_date] if not activity_df.empty and 'Date' in activity_df.columns else pd.DataFrame()

    sleep_window_end = sleep_info["sleep_window_end"]
    if sleep_window_end is not None and pd.Timestamp(sleep_window_end).date() <= target_date:
        floor = sleep_window_end
    else:
        floor = datetime.combine(target_date, datetime.min.time())

    if not day_df.empty and 'Datetime' in day_df.columns:
        src = day_df[day_df['Datetime'] >= floor]
    else:
        src = day_df
    hr = src['HEART_RATE'].dropna() if not src.empty and 'HEART_RATE' in src.columns else pd.Series(dtype=float)

    strain = 0.0
    if not hr.empty:
        auto_resting = sleep_info["sleep_hr_median"] if pd.notna(sleep_info["sleep_hr_median"]) else hr.quantile(0.10)
        resting_hr = resting_hr_override if resting_hr_override > 0 else max(40.0, auto_resting)
        hr_reserve = max(1.0, max_hr_override - resting_hr)
        smoothed = hr.sort_index().rolling(5, min_periods=1, center=True).median()
        active = smoothed[smoothed >= (resting_hr + STRAIN_ACTIVITY_GATE)]
        frac = ((active - resting_hr) / hr_reserve).clip(lower=0, upper=1)
        trimp = (frac * 0.64 * np.exp(1.92 * frac)).sum()
        STRAIN_SATURATION_T = 240.0  # lower = curve saturates faster (higher scores for the same trimp)
        effective_t = STRAIN_SATURATION_T / max(0.01, sensitivity)
        strain = min(21.0, round(21.0 * (1.0 - np.exp(-trimp / effective_t)), 1))

    total_steps = day_df['STEPS'].sum() if not day_df.empty and 'STEPS' in day_df.columns else 0
    avg_hr = day_df['HEART_RATE'].mean() if not day_df.empty and 'HEART_RATE' in day_df.columns else np.nan

    return {"strain_score": strain, "total_steps": total_steps, "avg_hr": avg_hr,
            "hrv_val": sleep_info["hrv_val"], "sleep_duration_mins": sleep_info["sleep_duration_mins"]}


# Global Metric Hunter for SpO2 & Stress
def fetch_global_metric(keywords, target_date):
    for t, cols in schema.items():
        val_col = next((c for c in cols if any(k in c for k in keywords)), None)
        if not val_col and any(k in t.upper() for k in keywords):
            val_col = next((c for c in cols if c in ["LEVEL", "VALUE", "SPO2", "STRESS", "INTENSITY"]), None)
        if val_col:
            time_c = next((c for c in cols if c in ["TIMESTAMP", "TIME", "DATE", "RAW_TIMESTAMP", "START_TIME"]), None)
            if time_c:
                try:
                    with sqlite3.connect(DB_FILE) as conn:
                        temp = pd.read_sql_query(f'SELECT "{time_c}", "{val_col}" FROM "{t}"', conn)
                        temp.columns = [time_c, val_col]
                        max_v = temp[time_c].max()
                        dates = to_local(temp[time_c], unit='ms' if max_v > 2e10 else 's', tz=LOCAL_TZ).dt.date
                        day_data = pd.to_numeric(temp[dates == target_date][val_col], errors='coerce')
                        day_data = day_data.replace([0, 255], np.nan).dropna()
                        if not day_data.empty:
                            return day_data.mean()
                except Exception:
                    continue
    return np.nan


# --- Recovery Score (0-100) ---------------------------------------------------------------
# Rough population reference points, not clinically validated percentiles - real population
# RHR/HRV vary substantially by age, sex, and measurement method. Good enough for a relative
# "roughly where does this sit" read, not a medical claim.
POPULATION_RHR_REF = 50.0        # bpm - below this scores ~100 on the population axis
POPULATION_HRV_LOW, POPULATION_HRV_HIGH = 20.0, 100.0  # ms - population "low" to "elite" range
RECOVERY_WEIGHTS = {"Sleep Duration": 0.40, "Resting Heart Rate": 0.20, "HRV": 0.20,
                     "Stress (previous day)": 0.10, "Strain (previous day)": 0.10}


def _sleep_hours_subscore(hours):
    """Peaks at 9h, falls off in both directions - matches Whoop's convention that both under-
    and oversleeping cost points, rather than treating 'more sleep' as unconditionally better.
    Sleep carries the largest single weight (40%) of any component here."""
    if hours is None or pd.isna(hours) or hours <= 0:
        return None
    diff = abs(hours - 9.0)
    return max(0.0, 100.0 - 5.0 * (diff ** 1.5))


def _sleep_debt_cap(hours):
    """A hard ceiling only for genuinely severe sleep deprivation - under 4 hours caps the
    overall score at 65 regardless of how RHR/HRV happen to read that morning. Sleep's 40%
    weight in the blend already does most of the work for moderate shortfalls; this only
    steps in for the extreme end."""
    if hours is None or pd.isna(hours) or hours <= 0:
        return 100.0
    if hours < 4.0:
        return 65.0
    return 100.0


def _rhr_subscore(today_rhr, personal_avg_rhr):
    """Blends a population-normed score with a personal-relative one (50 = exactly your own
    average). This is the key difference from Whoop's purely self-relative approach: someone
    with consistently excellent RHR who has a day merely 'average for them' still scores well
    here, because the population half of the blend recognizes that's still good in absolute
    terms - a pure self-relative score would flag it as a dip regardless of how good it is."""
    if today_rhr is None or pd.isna(today_rhr):
        return None
    pop_score = max(0.0, min(100.0, 100.0 - (today_rhr - POPULATION_RHR_REF) * 2.0))
    if personal_avg_rhr is not None and pd.notna(personal_avg_rhr):
        personal_score = max(0.0, min(100.0, 50.0 + (personal_avg_rhr - today_rhr) * 5.0))
        return 0.5 * pop_score + 0.5 * personal_score
    return pop_score


def _hrv_subscore(today_hrv, personal_avg_hrv):
    """Same population/personal blend as RHR, using percentage deviation for the personal half
    since baseline HRV varies hugely between individuals - a 10ms drop means very different
    things to someone with a 30ms baseline versus a 100ms one."""
    if today_hrv is None or pd.isna(today_hrv):
        return None
    pop_score = max(0.0, min(100.0, (today_hrv - POPULATION_HRV_LOW) * 100.0 / (POPULATION_HRV_HIGH - POPULATION_HRV_LOW)))
    if personal_avg_hrv is not None and pd.notna(personal_avg_hrv) and personal_avg_hrv > 0:
        pct_diff = (today_hrv - personal_avg_hrv) / personal_avg_hrv * 100.0
        personal_score = max(0.0, min(100.0, 50.0 + pct_diff * 2.0))
        return 0.5 * pop_score + 0.5 * personal_score
    return pop_score


def _stress_subscore(stress_val):
    if stress_val is None or pd.isna(stress_val):
        return None
    return max(0.0, min(100.0, 100.0 - stress_val))


def _strain_subscore(strain_val):
    """A hard previous day costs some points but never dominates - it's a minor input here,
    not the main signal, since HRV/RHR already partly reflect accumulated training stress."""
    if strain_val is None or pd.isna(strain_val):
        return None
    return max(0.0, min(100.0, 100.0 - (strain_val / 21.0) * 40.0))


@st.cache_data
def compute_recovery_score(target_date, activity_table, active_sleep_table, mtime, tz_name,
                            max_hr_override, resting_hr_override, sensitivity, available_dates_tuple, prev_stress):
    """Recovery score (0-100): sleep duration (peak at 9h, 40% weight), resting HR and HRV
    (each blended population-vs-personal, see subscore functions above), plus previous day's
    stress and strain as smaller inputs. Any missing component is dropped entirely and the
    remaining weights renormalize, rather than treating missing data as zero or failing."""
    sleep_info = compute_sleep_and_hrv(target_date, activity_table, active_sleep_table, mtime, tz_name)
    today_rhr = sleep_info["sleep_hr_median"]
    today_hrv = sleep_info["hrv_val"]
    sleep_hours = sleep_info["sleep_duration_mins"] / 60.0 if sleep_info["sleep_duration_mins"] > 0 else None

    if sleep_hours is None:
        # No real sleep detected for this night - RHR/HRV are already unavailable without it (no
        # daytime resting-proxy fallback), but Stress+Strain alone could still renormalize into a
        # confident-looking score from very thin evidence. Require sleep explicitly instead.
        return {"score": None, "components": {}, "personal_avg_rhr": None, "personal_avg_hrv": None,
                "today_rhr": today_rhr, "today_hrv": today_hrv, "sleep_hours": None,
                "raw_score_before_sleep_cap": None, "sleep_debt_cap": None}

    prior_dates = sorted([d for d in available_dates_tuple if d < target_date])[-30:]
    prior_rhrs, prior_hrvs = [], []
    for d in prior_dates:
        pi = compute_sleep_and_hrv(d, activity_table, active_sleep_table, mtime, tz_name)
        if pd.notna(pi["sleep_hr_median"]):
            prior_rhrs.append(pi["sleep_hr_median"])
        if pd.notna(pi["hrv_val"]):
            prior_hrvs.append(pi["hrv_val"])
    personal_avg_rhr = float(np.mean(prior_rhrs)) if prior_rhrs else None
    personal_avg_hrv = float(np.mean(prior_hrvs)) if prior_hrvs else None

    prev_date = target_date - timedelta(days=1)
    prev_strain = None
    if prev_date in available_dates_tuple:
        prev_strain_info = compute_strain_for_date(prev_date, activity_table, active_sleep_table, mtime, tz_name,
                                                     max_hr_override, resting_hr_override, sensitivity)
        prev_strain = prev_strain_info["strain_score"]

    components = {
        "Sleep Duration": (_sleep_hours_subscore(sleep_hours), RECOVERY_WEIGHTS["Sleep Duration"]),
        "Resting Heart Rate": (_rhr_subscore(today_rhr, personal_avg_rhr), RECOVERY_WEIGHTS["Resting Heart Rate"]),
        "HRV": (_hrv_subscore(today_hrv, personal_avg_hrv), RECOVERY_WEIGHTS["HRV"]),
        "Stress (previous day)": (_stress_subscore(prev_stress), RECOVERY_WEIGHTS["Stress (previous day)"]),
        "Strain (previous day)": (_strain_subscore(prev_strain), RECOVERY_WEIGHTS["Strain (previous day)"]),
    }
    available = {k: (v, w) for k, (v, w) in components.items() if v is not None}
    if not available:
        return {"score": None, "components": {}, "personal_avg_rhr": personal_avg_rhr, "personal_avg_hrv": personal_avg_hrv,
                "today_rhr": today_rhr, "today_hrv": today_hrv, "sleep_hours": sleep_hours,
                "raw_score_before_sleep_cap": None, "sleep_debt_cap": None}

    total_weight = sum(w for _, w in available.values())
    weighted_sum = sum(v * w for v, w in available.values())
    raw_score = weighted_sum / total_weight
    cap = _sleep_debt_cap(sleep_hours)
    final_score = min(raw_score, cap)
    return {"score": round(final_score), "components": {k: round(v, 1) for k, (v, _) in available.items()},
            "personal_avg_rhr": personal_avg_rhr, "personal_avg_hrv": personal_avg_hrv,
            "today_rhr": today_rhr, "today_hrv": today_hrv, "sleep_hours": sleep_hours,
            "raw_score_before_sleep_cap": round(raw_score), "sleep_debt_cap": round(cap) if cap < 100 else None}


def classify_recovery(score):
    if score is None:
        return None, None
    if score >= 67:
        return "🟢 Well Recovered", "#2ecc71"
    elif score >= 34:
        return "🟡 Adequate Recovery", "#f1c40f"
    else:
        return "🔴 Low Recovery", "#e74c3c"


# --- Daily Journal: simple per-day txt logs + trend comparisons ----------------------------
JOURNAL_DIR = "journal"
JOURNAL_TAGS_FILE = os.path.join(JOURNAL_DIR, "_tags.txt")
DEFAULT_JOURNAL_TAGS = ["Alcohol", "Caffeine (evening)", "Drugs/Medication", "Fish", "Late Meal",
                        "Screen Before Bed", "Stretching/Yoga", "Travel", "Feeling Sick", "High Stress Day"]


def load_journal_tags():
    if os.path.exists(JOURNAL_TAGS_FILE):
        with open(JOURNAL_TAGS_FILE, "r", encoding="utf-8") as f:
            tags = [line.strip() for line in f.read().split("\n") if line.strip()]
        if tags:
            return tags
    save_journal_tags(DEFAULT_JOURNAL_TAGS)
    return list(DEFAULT_JOURNAL_TAGS)


def save_journal_tags(tags):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    with open(JOURNAL_TAGS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(tags))


def _journal_path(d):
    return os.path.join(JOURNAL_DIR, f"{d.isoformat()}.txt")


def load_journal_entry(d):
    path = _journal_path(d)
    tags, notes, workouts = {}, "", {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        mode = "tags"
        notes_lines = []
        for line in lines:
            if mode == "notes":
                notes_lines.append(line)
                continue
            stripped = line.strip()
            if stripped.startswith("Workouts:"):
                mode = "workouts"
                continue
            if stripped.startswith("Notes:"):
                mode = "notes"
                rest = line.split(":", 1)[1].strip()
                if rest:
                    notes_lines.append(rest)
                continue
            if mode == "workouts":
                parts = line.split("|")
                if len(parts) == 3:
                    start_str, end_str, label = parts
                    workouts[start_str.strip()] = {"end": end_str.strip(), "label": label.strip()}
                continue
            if mode == "tags" and ":" in line:
                tag, val = line.split(":", 1)
                tags[tag.strip()] = val.strip().lower() in ("yes", "true", "y", "1")
        notes = "\n".join(notes_lines).strip()
    return tags, notes, workouts


def save_journal_entry(d, tag_values, notes, workout_labels=None):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    lines = [f"{tag}: {'Yes' if val else 'No'}" for tag, val in tag_values.items()]
    if workout_labels:
        lines.append("Workouts:")
        for start_str, info in workout_labels.items():
            lines.append(f"{start_str}|{info['end']}|{info['label']}")
    lines.append("Notes:")
    if notes:
        lines.append(notes)
    with open(_journal_path(d), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def has_journal_entry(d):
    return os.path.exists(_journal_path(d))


COMMON_SPORTS = ["Workout", "Run", "Walk", "Cycling", "Weights/Strength", "HIIT", "Yoga/Stretching",
                  "Swimming", "Sports/Game", "Hiking", "Other"]
WORKOUT_HRR_THRESHOLD = 0.40   # heart-rate-reserve fraction that counts as "working out", not just active
WORKOUT_MIN_MINUTES = 20       # shorter sustained elevations are treated as daily activity, not a session
WORKOUT_MIN_STRAIN = 5.0       # filters out trivial blips that technically qualify but barely register
WORKOUT_MERGE_GAP_MINUTES = 5  # brief dips (a red light, a rest between sets) don't split one session


def detect_workout_sessions(day_df, resting_hr, max_hr_override):
    """Find sustained elevated-HR sessions during waking hours - same block-detection idea as
    sleep, but looking for sustained HIGH heart-rate-reserve instead of low. Doesn't try to name
    the sport (steps-per-minute is too weak a signal to reliably tell cycling from rowing from
    an elliptical) - just a rough starting suggestion the person can correct."""
    if day_df.empty or 'HEART_RATE' not in day_df.columns or 'Datetime' not in day_df.columns:
        return []
    d = day_df.dropna(subset=['HEART_RATE']).sort_values('Datetime').reset_index(drop=True)
    if d.empty or pd.isna(resting_hr):
        return []
    hr_reserve = max(1.0, max_hr_override - resting_hr)
    smoothed = d['HEART_RATE'].rolling(3, min_periods=1, center=True).median()
    hrr_frac_all = ((smoothed - resting_hr) / hr_reserve).clip(lower=0, upper=1)
    is_elevated = hrr_frac_all >= WORKOUT_HRR_THRESHOLD

    block_id = (is_elevated != is_elevated.shift()).cumsum()
    raw_blocks = []
    for _, g in d.groupby(block_id):
        if not is_elevated.loc[g.index[0]]:
            continue
        gaps = g['Datetime'].diff().dt.total_seconds().dropna() / 60.0
        sample_width = gaps.median() if not gaps.empty else 1.0
        span_min = (g['Datetime'].max() - g['Datetime'].min()).total_seconds() / 60.0 + (sample_width if pd.notna(sample_width) else 1.0)
        raw_blocks.append({"start": g['Datetime'].min(), "end": g['Datetime'].max(), "span_min": span_min})

    merged = []
    for b in sorted(raw_blocks, key=lambda x: x["start"]):
        if merged and (b["start"] - merged[-1]["end"]).total_seconds() / 60.0 <= WORKOUT_MERGE_GAP_MINUTES:
            merged[-1]["end"] = max(merged[-1]["end"], b["end"])
        else:
            merged.append({"start": b["start"], "end": b["end"]})

    sessions = []
    for m in merged:
        span_min = (m["end"] - m["start"]).total_seconds() / 60.0
        if span_min < WORKOUT_MIN_MINUTES:
            continue
        seg = d[(d['Datetime'] >= m["start"]) & (d['Datetime'] <= m["end"])]
        seg_hr = seg['HEART_RATE'].dropna()
        if seg_hr.empty:
            continue
        frac = ((seg_hr - resting_hr) / hr_reserve).clip(lower=0, upper=1)
        trimp = (frac * 0.64 * np.exp(1.92 * frac)).sum()
        session_strain = min(21.0, round(21.0 * (1.0 - np.exp(-trimp / 240.0)), 1))
        if session_strain < WORKOUT_MIN_STRAIN:
            continue
        total_steps = seg['STEPS'].sum() if 'STEPS' in seg.columns else 0
        steps_per_min = total_steps / max(1.0, span_min)
        suggested = "Run" if steps_per_min >= 80 else ("Walk" if steps_per_min >= 25 else "Workout")
        sessions.append({"start": m["start"], "end": m["end"], "duration_min": span_min,
                          "avg_hr": seg_hr.mean(), "peak_hr": seg_hr.max(), "total_steps": int(total_steps),
                          "session_strain": session_strain, "suggested_label": suggested})
    return sessions


if not df_activity.empty and 'Date' in df_activity.columns:
    available_dates = sorted(df_activity['Date'].dropna().unique(), reverse=True)
else:
    available_dates = []

SIDEBAR_PAGE_DAYS = 14

if "week_offset" not in st.session_state:
    st.session_state.week_offset = 0

if available_dates:
    most_recent = available_dates[0]
    earliest = available_dates[-1]

    week_end = most_recent - timedelta(days=SIDEBAR_PAGE_DAYS * st.session_state.week_offset)
    week_dates = [week_end - timedelta(days=i) for i in range(SIDEBAR_PAGE_DAYS)]

    can_go_back = week_dates[-1] > earliest
    can_go_forward = st.session_state.week_offset > 0

    nav_prev, nav_label, nav_next = st.sidebar.columns([1, 3, 1])
    with nav_prev:
        if st.button("◀", key="week_prev", disabled=not can_go_back):
            st.session_state.week_offset += 1
            st.rerun()
    with nav_label:
        st.markdown(f"<p style='text-align:center;margin:0;padding-top:5px;font-size:0.8em;color:#888;'>"
                   f"{week_dates[-1].strftime('%b %d')}–{week_dates[0].strftime('%b %d')}</p>", unsafe_allow_html=True)
    with nav_next:
        if st.button("▶", key="week_next", disabled=not can_go_forward):
            st.session_state.week_offset -= 1
            st.rerun()

    day_info_by_date = {}
    with st.spinner("Loading history..."):
        for d in week_dates:
            strain_info_d = compute_strain_for_date(d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                                      st.session_state.max_hr_override, st.session_state.resting_hr_override,
                                                      st.session_state.strain_sensitivity)
            stress_prev_d = fetch_global_metric(["STRESS"], d - timedelta(days=1))
            recovery_d = compute_recovery_score(d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                                 st.session_state.max_hr_override, st.session_state.resting_hr_override,
                                                 st.session_state.strain_sensitivity, tuple(available_dates), stress_prev_d)
            day_info_by_date[d] = {"strain_score": strain_info_d["strain_score"], "hrv_val": strain_info_d["hrv_val"],
                                    "recovery_score": recovery_d["score"]}

    st.sidebar.caption("🗓️ Date — Recovery / Strain / HRV")

    def _fmt_date(d):
        info = day_info_by_date.get(d, {})
        strain, hrv, recovery = info.get("strain_score"), info.get("hrv_val"), info.get("recovery_score")
        strain_str = f"{strain:.1f}" if pd.notna(strain) else "—"
        hrv_str = f"{int(hrv)}ms" if pd.notna(hrv) else "—"
        recovery_str = str(int(recovery)) if recovery is not None else "—"
        return f"{d.strftime('%a %d %b')}  —  🔋{recovery_str} 🔥{strain_str} ⚡{hrv_str}"

    selected_date = st.sidebar.radio("Select a day to analyze", week_dates, format_func=_fmt_date,
                                      label_visibility="collapsed", key=f"date_radio_{st.session_state.week_offset}")
else:
    selected_date = datetime.now().date()

# All other settings, collapsed by default.
with st.sidebar.expander("⚙️ Advanced Settings"):
    if st.button("🔄 Force-refresh data"):
        st.cache_data.clear()
        st.rerun()

    st.text_input("Timezone (IANA name)", key="tz_name",
                   help="Used to convert the database's UTC timestamps to your local time. "
                        "Change this if you're not in the UK - e.g. 'America/New_York', 'Asia/Tokyo'.")
    if st.session_state.tz_name and st.session_state.tz_name not in ("", None):
        try:
            ZoneInfo(st.session_state.tz_name)
        except Exception:
            st.error(f"Unknown timezone '{st.session_state.tz_name}', falling back to UTC.")

    st.selectbox("🛠️ Override Activity Table:", all_tables, key="activity_table_override")
    st.selectbox("💤 Override Sleep Table:", ["Auto-Detect"] + sleep_tables, key="sleep_table_override")

    st.divider()
    st.caption("⚙️ Strain Calibration (optional)")
    st.number_input("Max Heart Rate", min_value=120, max_value=230, step=1, key="max_hr_override",
                     help="Used to scale Day Strain. Use your real tested max HR if you know it (e.g. 220-age as a rough estimate).")
    st.number_input("Resting Heart Rate (0 = auto)", min_value=0, max_value=100, step=1, key="resting_hr_override",
                     help="Leave at 0 to auto-estimate resting HR from overnight readings.")
    st.number_input("Strain Sensitivity (1.0 = default)", min_value=0.3, max_value=2.0, step=0.1, key="strain_sensitivity",
                     help="Turn down if Strain still runs high for how intense your days actually feel; turn up if it runs low.")

    st.divider()
    st.caption(f"Active Activity Source: `{activity_table}`")
    if active_sleep_table:
        st.caption(f"Sleep Source: `{active_sleep_table}`")

max_hr_override = st.session_state.max_hr_override
resting_hr_override = st.session_state.resting_hr_override

target_df = df_activity[df_activity['Date'] == selected_date] if not df_activity.empty and 'Date' in df_activity.columns else pd.DataFrame()
last_7_days = df_activity[(df_activity['Date'] > (selected_date - timedelta(days=7))) & (df_activity['Date'] <= selected_date)] if not df_activity.empty and 'Date' in df_activity.columns else pd.DataFrame()

sleep_info = compute_sleep_and_hrv(selected_date, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name)
sleep_duration_mins = sleep_info["sleep_duration_mins"]
sleep_start_str = sleep_info["sleep_start_str"]
sleep_end_str = sleep_info["sleep_end_str"]
sleep_window_start = sleep_info["sleep_window_start"]
sleep_window_end = sleep_info["sleep_window_end"]
hrv_val = sleep_info["hrv_val"]
hrv_source = sleep_info["hrv_source"]
sleep_diag = sleep_info["sleep_diag"]
window_start = sleep_info["window_start"]
window_end = sleep_info["window_end"]

# --- General Metrics ---
total_steps = target_df['STEPS'].sum() if not target_df.empty and "STEPS" in target_df.columns else 0
hr_data = target_df['HEART_RATE'].dropna() if not target_df.empty and "HEART_RATE" in target_df.columns else pd.Series()
avg_hr = hr_data.mean() if not hr_data.empty else np.nan
peak_hr = hr_data.max() if not hr_data.empty else np.nan

strain_info = compute_strain_for_date(selected_date, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                       max_hr_override, resting_hr_override, st.session_state.strain_sensitivity)
strain_score = strain_info["strain_score"]

spo2_val = fetch_global_metric(["SPO2", "OXYGEN"], selected_date)
stress_val = fetch_global_metric(["STRESS"], selected_date)
prev_stress_val = fetch_global_metric(["STRESS"], selected_date - timedelta(days=1))

def classify_stress(val):
    """Xiaomi/Huami's stress score follows the same 0-100 HRV-derived scale used across the
    Zepp ecosystem (0 = fully relaxed, 100 = high stress). There's no single official Xiaomi-
    published cutoff table, but the standard bands used across these devices are roughly:
    0-25 relaxed, 26-50 normal, 51-75 medium, 76-100 high."""
    if pd.isna(val):
        return None, None
    if val <= 25:
        return "🟢 Relaxed - low stress", "This is a good, low-stress reading."
    elif val <= 50:
        return "🟡 Normal", "Typical day-to-day range, nothing to be concerned about."
    elif val <= 75:
        return "🟠 Medium - elevated", "Higher than resting - could be exercise, caffeine, or genuine stress."
    else:
        return "🔴 High", "Significantly elevated - worth noting if it persists through the day."

recovery_info = compute_recovery_score(selected_date, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                        max_hr_override, resting_hr_override, st.session_state.strain_sensitivity,
                                        tuple(available_dates), prev_stress_val)

# --- Dashboard UI Layout ---
st.subheader(f"📊 Daily Report: {selected_date.strftime('%A, %B %d, %Y')}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="🔥 Day Strain (0-21)", value=f"{strain_score:.1f}")
with col2:
    hrv_caption = "from overnight HR" if hrv_source == "sleep" else ""
    st.metric(label="⚡ HRV (RMSSD)", value=f"{int(hrv_val)} ms" if pd.notna(hrv_val) else "N/A", delta=hrv_caption if hrv_caption else None, delta_color="off")
with col3:
    st.metric(label="❤️ Avg Heart Rate", value=f"{avg_hr:.1f} bpm" if pd.notna(avg_hr) else "N/A")
with col4:
    st.metric(label="👟 Total Steps", value=f"{int(total_steps):,}")

st.divider()

col5, col6, col7, col8 = st.columns(4)
with col5:
    hours = int(sleep_duration_mins // 60)
    mins = int(sleep_duration_mins % 60)
    if sleep_duration_mins > 0:
        sd_value, sd_delta = f"{hours}h {mins}m", (f"From {sleep_start_str} to {sleep_end_str}" if sleep_start_str != "N/A" else None)
    elif sleep_diag.get("source") == "activity_table_fallback_rejected":
        sd_value, sd_delta = "No reliable data", "fallback rejected as implausible, see diagnostics"
    else:
        sd_value, sd_delta = "No Sleep Logged", None
    st.metric(label="💤 Sleep Duration", value=sd_value, delta=sd_delta, delta_color="off")
with col6:
    if recovery_info["score"] is not None:
        label, _ = classify_recovery(recovery_info["score"])
        st.metric(label="🔋 Recovery Score", value=f"{recovery_info['score']}/100", delta=label, delta_color="off")
    else:
        st.metric(label="🔋 Recovery Score", value="N/A", delta="not enough data yet", delta_color="off")
with col7:
    st.metric(label="🩸 Blood Oxygen (SpO2)", value=f"{spo2_val:.1f}%" if pd.notna(spo2_val) else "No Sensor Data")
with col8:
    st.metric(label="🧠 Stress Score", value=f"{stress_val:.0f}" if pd.notna(stress_val) else "No Sensor Data")
    if pd.notna(stress_val):
        label, note = classify_stress(stress_val)
        st.caption(f"{label} — {note}")

st.divider()

st.subheader("📈 Trends")
trend_period = st.radio("Period", ["7 Days", "30 Days", "1 Year", "All Time"], horizontal=True, index=0,
                         key="trend_period", label_visibility="collapsed")

if trend_period == "7 Days":
    range_start = selected_date - timedelta(days=6)
elif trend_period == "30 Days":
    range_start = selected_date - timedelta(days=29)
elif trend_period == "1 Year":
    range_start = selected_date - timedelta(days=364)
else:
    range_start = available_dates[-1] if available_dates else selected_date
if available_dates:
    range_start = max(range_start, available_dates[-1])

range_dates = [range_start + timedelta(days=i) for i in range((selected_date - range_start).days + 1)]

trend_rows = []
with st.spinner(f"Loading {trend_period.lower()}..."):
    for d in range_dates:
        s_info = compute_strain_for_date(d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                          max_hr_override, resting_hr_override, st.session_state.strain_sensitivity)
        stress_prev_d = fetch_global_metric(["STRESS"], d - timedelta(days=1))
        r_info = compute_recovery_score(d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                         max_hr_override, resting_hr_override, st.session_state.strain_sensitivity,
                                         tuple(available_dates), stress_prev_d)
        trend_rows.append({"Date": pd.Timestamp(d), "Steps": s_info["total_steps"], "Sleep_min": s_info["sleep_duration_mins"],
                            "Strain": s_info["strain_score"], "Recovery": r_info["score"], "HRV": s_info["hrv_val"]})
trend_df = pd.DataFrame(trend_rows)


def render_trend_chart(df, col, color, y_title, height=240):
    sub = df[["Date", col]].dropna(subset=[col])
    if sub.empty or (sub[col] == 0).all():
        st.info(f"No {y_title.lower()} data available for this period.")
        return
    use_bar = len(sub) <= 31
    if use_bar:
        # Ordinal day labels avoid Altair's temporal axis auto-inserting sub-day ticks
        # (e.g. "12 PM") when there's only one data point per day.
        sub = sub.copy()
        sub['Day'] = sub['Date'].dt.strftime('%a %d')
        chart = alt.Chart(sub).mark_bar(color=color).encode(
            x=alt.X('Day:O', sort=None, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f'{col}:Q', title=y_title),
            tooltip=[alt.Tooltip('Day:N', title='Day'), alt.Tooltip(f'{col}:Q', title=y_title)],
        ).properties(height=height)
    else:
        chart = alt.Chart(sub).mark_line(color=color, point=len(sub) <= 60).encode(
            x=alt.X('Date:T', title=None, axis=alt.Axis(format='%b %d', tickCount=8)),
            y=alt.Y(f'{col}:Q', title=y_title),
            tooltip=[alt.Tooltip('Date:T', format='%b %d, %Y'), alt.Tooltip(f'{col}:Q', title=y_title)],
        ).properties(height=height)
    st.altair_chart(chart, width='stretch')


@st.dialog("📈 Trend Detail", width="large")
def show_trend_dialog(title, col_key, color, y_title):
    st.subheader(title)
    render_trend_chart(trend_df, col_key, color, y_title, height=450)


TREND_METRICS = [
    ("👟 Steps", "Steps", "#29b5e8", "Steps"),
    ("💤 Sleep", "Sleep_min", "#9b59b6", "Minutes"),
    ("🔥 Strain", "Strain", "#e67e22", "Strain"),
    ("🔋 Recovery", "Recovery", "#2ecc71", "Recovery"),
    ("⚡ HRV", "HRV", "#3498db", "ms"),
]

grid_cols = st.columns(2)
for i, (title, col_key, color, y_title) in enumerate(TREND_METRICS):
    with grid_cols[i % 2]:
        st.markdown(f"**{title}**")
        render_trend_chart(trend_df, col_key, color, y_title, height=180)
        if st.button("🔍 Inspect closer", key=f"inspect_{col_key}", width='stretch'):
            show_trend_dialog(title, col_key, color, y_title)

st.divider()

st.subheader("📔 Daily Journal")
journal_tags = load_journal_tags()
existing_tags, existing_notes, existing_workouts = load_journal_entry(selected_date)

tag_values = {}
n_cols = 4
cols = st.columns(n_cols)
for i, tag in enumerate(journal_tags):
    with cols[i % n_cols]:
        tag_values[tag] = st.checkbox(tag, value=existing_tags.get(tag, False),
                                       key=f"journal_{tag}_{selected_date.isoformat()}")
with cols[len(journal_tags) % n_cols]:
    with st.popover("🏷️ Manage tags"):
        st.caption("Add a new tag")
        new_tag_input = st.text_input("New tag name", key="new_journal_tag_input", label_visibility="collapsed",
                                       placeholder="e.g. Cold Plunge")
        if st.button("Add", key="add_journal_tag_btn") and new_tag_input.strip():
            if new_tag_input.strip() not in journal_tags:
                save_journal_tags(journal_tags + [new_tag_input.strip()])
                st.rerun()

        if journal_tags:
            st.divider()
            st.caption("Remove a tag - not relevant to you? Drop it, e.g. Alcohol if you never drink")
            for t in journal_tags:
                rc1, rc2 = st.columns([4, 1])
                with rc1:
                    st.write(t)
                with rc2:
                    if st.button("✕", key=f"remove_tag_{t}"):
                        save_journal_tags([x for x in journal_tags if x != t])
                        st.rerun()
        st.caption("Removing a tag only stops tracking it going forward - past logged entries keep their data.")

if not journal_tags:
    st.caption("No journal tags yet - use 🏷️ Manage tags above to add one.")

notes_input = st.text_area("Notes", value=existing_notes, key=f"journal_notes_{selected_date.isoformat()}",
                            placeholder="Anything else worth noting about today...")

jc1, jc2 = st.columns([1, 4])
with jc1:
    if st.button("💾 Save Entry", key=f"journal_save_{selected_date.isoformat()}"):
        save_journal_entry(selected_date, tag_values, notes_input)
        st.rerun()
with jc2:
    if has_journal_entry(selected_date):
        st.caption(f"✅ Entry saved for {selected_date.strftime('%b %d')} — stored at `journal/{selected_date.isoformat()}.txt`")
    else:
        st.caption("No entry saved yet for this day.")

with st.expander("📈 Journal Insights: how your trends compare"):
    journaled_dates = [d for d in available_dates if has_journal_entry(d)]
    if len(journaled_dates) < 2:
        st.caption("Log at least a couple of days (ideally with some variation - a tag marked Yes on some days, No on others) to start seeing comparisons here.")
    else:
        rows = []
        for d in journaled_dates:
            tags, _, _ = load_journal_entry(d)
            s_info = compute_strain_for_date(d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                              max_hr_override, resting_hr_override, st.session_state.strain_sensitivity)
            # Strain is wake-anchored and accumulates through day d, so it reflects day d itself.
            # HRV/Sleep/Recovery for "day d" on the dashboard are actually from the night ENDING
            # that morning - i.e. they reflect what happened the evening before. What a tag logged
            # for day d (say, alcohol that evening) actually affects is the night that FOLLOWS d,
            # reported as day d+1's numbers. Pull those instead so the comparison lines up right.
            next_day_info = compute_strain_for_date(d + timedelta(days=1), activity_table, active_sleep_table, _db_mtime(),
                                                      st.session_state.tz_name, max_hr_override, resting_hr_override,
                                                      st.session_state.strain_sensitivity)
            stress_on_d = fetch_global_metric(["STRESS"], d)
            next_day_recovery = compute_recovery_score(d + timedelta(days=1), activity_table, active_sleep_table, _db_mtime(),
                                                         st.session_state.tz_name, max_hr_override, resting_hr_override,
                                                         st.session_state.strain_sensitivity, tuple(available_dates), stress_on_d)
            rows.append({"tags": tags, "strain": s_info["strain_score"], "hrv": next_day_info["hrv_val"],
                         "sleep_mins": next_day_info["sleep_duration_mins"], "recovery": next_day_recovery["score"]})

        all_tags = sorted({t for r in rows for t in r["tags"].keys()})

        def _avg(key, rs):
            vals = [r[key] for r in rs if pd.notna(r[key]) and r[key] != 0]
            return np.mean(vals) if vals else np.nan

        # Tags rarely happen in isolation - if Alcohol and Meditate almost always co-occur,
        # a plain "Meditate Yes vs No" comparison is really measuring both at once. This finds,
        # for each tag, any other tag it's strongly correlated with across journaled days, so
        # that can be flagged rather than silently baked into the number.
        def _tag_correlation(tag_a, tag_b):
            pairs = [(r["tags"][tag_a], r["tags"][tag_b]) for r in rows if tag_a in r["tags"] and tag_b in r["tags"]]
            if len(pairs) < 3:
                return None
            xs = np.array([1.0 if p[0] else 0.0 for p in pairs])
            ys = np.array([1.0 if p[1] else 0.0 for p in pairs])
            if xs.std() == 0 or ys.std() == 0:
                return None
            return float(np.corrcoef(xs, ys)[0, 1])

        CONFOUND_THRESHOLD = 0.5
        confounds = {}
        for tag in all_tags:
            found = [(other, c) for other in all_tags if other != tag
                     for c in [_tag_correlation(tag, other)] if c is not None and abs(c) >= CONFOUND_THRESHOLD]
            confounds[tag] = sorted(found, key=lambda x: -abs(x[1]))

        comparison_rows = []
        for tag in all_tags:
            yes_rows = [r for r in rows if r["tags"].get(tag) is True]
            no_rows = [r for r in rows if r["tags"].get(tag) is False]
            if not yes_rows or not no_rows:
                continue
            correlated = ", ".join(f"{o} ({'+' if c > 0 else '−'}{abs(c):.1f})" for o, c in confounds.get(tag, [])[:2])
            comparison_rows.append({
                "Tag": tag, "n (Yes/No)": f"{len(yes_rows)}/{len(no_rows)}",
                "Strain that day (Yes)": _avg("strain", yes_rows), "Strain that day (No)": _avg("strain", no_rows),
                "Recovery next morning (Yes)": _avg("recovery", yes_rows), "Recovery next morning (No)": _avg("recovery", no_rows),
                "HRV that night (Yes)": _avg("hrv", yes_rows), "HRV that night (No)": _avg("hrv", no_rows),
                "Sleep that night (Yes)": _avg("sleep_mins", yes_rows), "Sleep that night (No)": _avg("sleep_mins", no_rows),
                "Correlated With": correlated or "—",
            })

        if not comparison_rows:
            st.caption("Not enough variation yet - each tag needs at least one 'Yes' day and one 'No' day to compare.")
        else:
            comp_df = pd.DataFrame(comparison_rows)
            for col in ["Sleep that night (Yes)", "Sleep that night (No)"]:
                comp_df[col] = comp_df[col].apply(lambda m: f"{int(m // 60)}h {int(m % 60)}m" if pd.notna(m) else "—")
            for col in ["Strain that day (Yes)", "Strain that day (No)"]:
                comp_df[col] = comp_df[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            for col in ["HRV that night (Yes)", "HRV that night (No)"]:
                comp_df[col] = comp_df[col].apply(lambda v: f"{int(v)} ms" if pd.notna(v) else "—")
            for col in ["Recovery next morning (Yes)", "Recovery next morning (No)"]:
                comp_df[col] = comp_df[col].apply(lambda v: f"{int(v)}/100" if pd.notna(v) else "—")
            st.dataframe(comp_df, hide_index=True, width='stretch')
            st.caption("Strain reflects the day itself (from when you woke up). Recovery, HRV, and Sleep reflect the night "
                       "that *followed* that day, since that's the sleep your logged behavior would actually have affected - "
                       "e.g. Alcohol logged for the 5th is compared against your Recovery Score on the morning of the 6th, "
                       "not the morning of the 5th.")
            st.caption("\"Correlated With\" flags tags that tend to happen together in your logs (+ means together, − means opposite), "
                       "so a strong number there might really belong to the correlated tag, not this one. See below to help tell them apart.")

            st.markdown("**🔍 Controlled comparisons**")
            st.caption("Holding a correlated tag fixed at 'No' isolates the other tag's effect on those days specifically - "
                       "e.g. does Alcohol still look bad on days you also didn't meditate?")
            shown_pairs = set()
            any_controlled = False
            for tag in all_tags:
                for other, corr in confounds.get(tag, []):
                    pair_key = (tag, other)
                    if pair_key in shown_pairs:
                        continue
                    stratum = [r for r in rows if r["tags"].get(other) is False]
                    yes_s = [r for r in stratum if r["tags"].get(tag) is True]
                    no_s = [r for r in stratum if r["tags"].get(tag) is False]
                    if not yes_s or not no_s:
                        continue
                    shown_pairs.add(pair_key)
                    any_controlled = True
                    s_yes, s_no = _avg("strain", yes_s), _avg("strain", no_s)
                    h_yes, h_no = _avg("hrv", yes_s), _avg("hrv", no_s)
                    r_yes, r_no = _avg("recovery", yes_s), _avg("recovery", no_s)
                    line = f"**{tag}** vs no {tag}, on days without **{other}** (n={len(yes_s)}/{len(no_s)}): "
                    parts = []
                    if pd.notna(s_yes) and pd.notna(s_no):
                        parts.append(f"Strain {s_yes:.1f} vs {s_no:.1f}")
                    if pd.notna(r_yes) and pd.notna(r_no):
                        parts.append(f"Recovery {int(r_yes)} vs {int(r_no)}")
                    if pd.notna(h_yes) and pd.notna(h_no):
                        parts.append(f"HRV {int(h_yes)}ms vs {int(h_no)}ms")
                    st.markdown(line + ", ".join(parts) if parts else line + "not enough data")
            if not any_controlled:
                st.caption("No controlled comparisons available yet - need days where a correlated tag stays constant while the other one varies.")

st.divider()

st.subheader(f"⏳ Intraday Heart Rate ({selected_date.strftime('%b %d')})")
if not hr_data.empty:
    st.line_chart(target_df.set_index('Datetime')['HEART_RATE'], color="#ff4b4b")
else:
    st.info("No intraday heart rate data available for this specific date.")

st.subheader("🏋️ Workouts Today")
resting_hr_for_workouts = sleep_info["sleep_hr_median"] if pd.notna(sleep_info["sleep_hr_median"]) else (hr_data.quantile(0.10) if not hr_data.empty else np.nan)
workout_sessions = detect_workout_sessions(target_df, resting_hr_for_workouts, max_hr_override)

if workout_sessions:
    st.caption("Sustained heart rate well above resting - detected automatically, but the sport itself has to be picked or typed, since HR/steps alone can't reliably tell cycling from rowing from an elliptical.")
    new_workout_labels = {}
    for sess in workout_sessions:
        start_key = sess['start'].strftime('%H:%M')
        end_str = sess['end'].strftime('%H:%M')
        existing = existing_workouts.get(start_key, {})
        default_label = existing.get('label', sess['suggested_label'])

        with st.container(border=True):
            st.markdown(f"**{start_key}–{end_str}** ({int(sess['duration_min'])} min)")
            wc1, wc2, wc3 = st.columns(3)
            wc1.metric("Avg HR", f"{sess['avg_hr']:.0f} bpm")
            wc2.metric("Peak HR", f"{sess['peak_hr']:.0f} bpm")
            wc3.metric("Session Strain", f"{sess['session_strain']:.1f}")

            options = COMMON_SPORTS.copy()
            preset_default = default_label if default_label in options else "Other"
            sc1, sc2 = st.columns([1, 1])
            with sc1:
                sport_choice = st.selectbox("Sport", options, index=options.index(preset_default),
                                             key=f"workout_sport_{selected_date.isoformat()}_{start_key}")
            final_label = sport_choice
            if sport_choice == "Other":
                with sc2:
                    custom = st.text_input("Specify", value=default_label if default_label not in COMMON_SPORTS else "",
                                            key=f"workout_sport_custom_{selected_date.isoformat()}_{start_key}")
                    final_label = custom.strip() if custom.strip() else "Other"
            new_workout_labels[start_key] = {"end": end_str, "label": final_label}

    if st.button("💾 Save Workout Labels", key=f"save_workouts_{selected_date.isoformat()}"):
        cur_tags, cur_notes, _ = load_journal_entry(selected_date)
        save_journal_entry(selected_date, cur_tags, cur_notes, new_workout_labels)
        st.rerun()
else:
    st.caption("No workout-level sessions detected today (sustained heart rate meaningfully above resting).")

with st.expander("⚙️ Database Transparency & Sleep Table Inspector"):
    st.write(f"Active Activity Table: `{activity_table}` | Active Sleep Table: `{active_sleep_table if active_sleep_table else 'None'}`")

    st.markdown("**All tables in this database**")
    st.caption("If your sleep-stage table is empty, Gadgetbridge's own app may be reading a different table for its sleep report - scan this list for anything else sleep-related (e.g. a *_SUMMARY table).")
    st.code(", ".join(sorted(schema.keys())), language=None)

    st.markdown("**Sleep resolution diagnostics**")
    st.json(sleep_diag, expanded=False)
    if window_start is not None:
        st.caption(f"Overnight window checked: {window_start.strftime('%Y-%m-%d %H:%M')} → {window_end.strftime('%Y-%m-%d %H:%M')}")
    if sleep_window_start is not None:
        st.caption(f"Sleep window excluded from Strain: {sleep_window_start.strftime('%Y-%m-%d %H:%M')} → {sleep_window_end.strftime('%Y-%m-%d %H:%M')}")

    if active_sleep_table:
        st.write("Sleep Table Preview:")
        st.dataframe(load_table(active_sleep_table, _db_mtime()).head(50))

    st.markdown("**Recovery Score breakdown**")
    if recovery_info["score"] is not None:
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Today's resting HR", f"{recovery_info['today_rhr']:.0f} bpm" if pd.notna(recovery_info['today_rhr']) else "—")
        rc2.metric("Your average RHR", f"{recovery_info['personal_avg_rhr']:.0f} bpm" if recovery_info['personal_avg_rhr'] else "no history yet")
        rc3.metric("Sleep", f"{recovery_info['sleep_hours']:.1f}h" if recovery_info['sleep_hours'] else "—")
        rc4, rc5, _ = st.columns(3)
        rc4.metric("Today's HRV", f"{recovery_info['today_hrv']:.0f} ms" if pd.notna(recovery_info['today_hrv']) else "—")
        rc5.metric("Your average HRV", f"{recovery_info['personal_avg_hrv']:.0f} ms" if recovery_info['personal_avg_hrv'] else "no history yet")
        st.write("Component subscores (each 0-100, weighted and averaged into the final score):")
        st.json(recovery_info["components"], expanded=True)
        st.caption("RHR and HRV subscores each blend a population-normed estimate with how today compares to your own "
                   "rolling average (up to the last 30 days before this one) - so a day that's merely average for you, "
                   "but still strong by general standards, doesn't get penalized the way a purely self-relative score would.")
        if recovery_info["sleep_debt_cap"] is not None:
            st.caption(f"⚠️ Sleep-debt cap applied: the weighted blend of components alone would have scored "
                       f"{recovery_info['raw_score_before_sleep_cap']}, but under 7h of sleep puts a ceiling of "
                       f"{recovery_info['sleep_debt_cap']} on the overall score regardless of how RHR/HRV read that morning.")
    else:
        st.caption("Not enough data yet to compute a Recovery Score for this day.")

    st.markdown("**Activity/Steps diagnostics for the selected day**")
    if not target_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows this day", len(target_df))
        c2.metric("Rows with STEPS > 0", int((target_df['STEPS'] > 0).sum()) if 'STEPS' in target_df.columns else 0)
        c3.metric("Rows with HR reading", int(target_df['HEART_RATE'].notna().sum()) if 'HEART_RATE' in target_df.columns else 0)
        st.caption("If HR rows vastly outnumber STEPS>0 rows, the device is likely syncing continuous background HR faster/more reliably than its step-history batch, or the STEPS field for this firmware needs a different column/scale - compare against the raw preview below.")
    st.write("Activity Table Preview:")
    st.dataframe(target_df.head(50))