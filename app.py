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
    page_title="Whoop-Style Health Dashboard",
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
    "deep_codes": "3",
    "light_codes": "2",
    "rem_codes": "4",
    "awake_codes": "",
    "sleep_table_override": "Auto-Detect",
    "journal_tags": "Alcohol, Caffeine (evening), Drugs/Medication, Fish, Late Meal, Screen Before Bed, Stretching/Yoga, Travel, Feeling Sick, High Stress Day",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if st.session_state.get("_pending_new_journal_tag"):
    _new_tag = st.session_state.pop("_pending_new_journal_tag").strip()
    _existing = [t.strip() for t in st.session_state.journal_tags.replace("\n", ",").split(",") if t.strip()]
    if _new_tag and _new_tag not in _existing:
        _existing.append(_new_tag)
        st.session_state.journal_tags = ", ".join(_existing)
    if "new_journal_tag_input" in st.session_state:
        del st.session_state["new_journal_tag_input"]

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

def classify_stage(raw, deep_codes, light_codes, rem_codes, awake_codes):
    """Map a raw stage code/label to a bucket using the sidebar-configured code sets
    (exact match - Xiaomi's numeric encoding isn't publicly documented and a loose
    substring match like '3' in '13' would misfire). Also accepts textual labels."""
    s = str(raw).strip().upper()
    if s in deep_codes or "DEEP" in s:
        return "deep"
    if s in light_codes or "LIGHT" in s:
        return "light"
    if s in rem_codes or "REM" in s:
        return "rem"
    if s in awake_codes or "AWAKE" in s or "WAKE" in s:
        return "awake"
    return "other"


def _parse_codes(s):
    return {c.strip().upper() for c in s.split(",") if c.strip()}


MIN_SLEEP_BLOCK_HOURS = 2.0


def detect_sleep_from_raw_kind(activity_df, w_start, w_end, baseline_hr):
    """Find the longest contiguous same-RAW_KIND run overnight with resting-level HR. Far more
    reliable on Huami/Xiaomi devices than a raw 'SLEEP' column, which can be a noisy counter
    rather than a boolean. Doesn't hardcode which code means sleep - it varies by device."""
    if 'RAW_KIND' not in activity_df.columns or 'Datetime' not in activity_df.columns:
        return None
    window_df = activity_df[(activity_df['Datetime'] >= w_start) & (activity_df['Datetime'] <= w_end)].sort_values('Datetime')
    if window_df.empty:
        return None
    blocks = (window_df['RAW_KIND'] != window_df['RAW_KIND'].shift()).cumsum()
    candidates = []
    for _, g in window_df.groupby(blocks):
        if len(g) < 30:
            continue
        gaps = g['Datetime'].diff().dt.total_seconds().dropna() / 60.0
        sample_width = gaps.median() if not gaps.empty else 1.0
        span_min = (g['Datetime'].max() - g['Datetime'].min()).total_seconds() / 60.0 + (sample_width if pd.notna(sample_width) else 1.0)
        if span_min < MIN_SLEEP_BLOCK_HOURS * 60:
            continue
        avg_hr = g['HEART_RATE'].mean()
        candidates.append({"raw_kind": g['RAW_KIND'].iloc[0], "start": g['Datetime'].min(), "end": g['Datetime'].max(),
                            "span_min": span_min, "avg_hr": avg_hr, "n": len(g)})
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c["span_min"])
    best = candidates[0]
    if pd.notna(baseline_hr) and pd.notna(best["avg_hr"]) and best["avg_hr"] > baseline_hr:
        return None
    return best


@st.cache_data
def compute_sleep_and_hrv(target_date, activity_table, active_sleep_table, mtime, tz_name,
                           deep_codes_str, light_codes_str, rem_codes_str, awake_codes_str):
    """Everything needed to report sleep + HRV for a single date. Cached per (date, settings)
    so the sidebar can call this once per day in the history list without recomputing on every
    rerun, and the main dashboard reuses the exact same cached result for the selected date."""
    tz = get_local_tz(tz_name)
    activity_df = get_cleaned_activity_df(activity_table, mtime, tz_name) if activity_table else pd.DataFrame()
    deep_codes, light_codes, rem_codes, awake_codes = (
        _parse_codes(deep_codes_str), _parse_codes(light_codes_str), _parse_codes(rem_codes_str), _parse_codes(awake_codes_str)
    )

    result = {
        "sleep_duration_mins": 0, "deep_sleep_mins": 0, "light_sleep_mins": 0, "rem_sleep_mins": 0,
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
            stage_col = next((c for c in df_sleep.columns if any(k in c for k in ["STAGE", "MODE", "KIND", "VALUE", "STATE", "TYPE"])), None)
            sleep_diag["table"] = active_sleep_table
            sleep_diag["total_rows"] = len(df_sleep)
            sleep_diag["stage_col"] = stage_col

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

                        stage_durations = {}
                        for d_val, raw_stage in zip(gaps_min, day_sleep[stage_col] if stage_col else [None] * len(day_sleep)):
                            code_label = str(raw_stage) if stage_col and pd.notna(raw_stage) else "(none)"
                            stage_durations[code_label] = stage_durations.get(code_label, 0.0) + d_val
                            bucket = classify_stage(raw_stage, deep_codes, light_codes, rem_codes, awake_codes) if stage_col and pd.notna(raw_stage) else "other"
                            if bucket == "deep":
                                result["deep_sleep_mins"] += d_val
                                result["sleep_duration_mins"] += d_val
                            elif bucket == "light":
                                result["light_sleep_mins"] += d_val
                                result["sleep_duration_mins"] += d_val
                            elif bucket == "rem":
                                result["rem_sleep_mins"] += d_val
                                result["sleep_duration_mins"] += d_val
                            elif bucket == "awake":
                                pass
                            else:
                                result["sleep_duration_mins"] += d_val

                        sleep_diag["minutes_per_raw_code"] = {k: round(v, 1) for k, v in sorted(stage_durations.items(), key=lambda kv: -kv[1])}

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
            sleep_diag["source"] = "activity_table_raw_kind_total_only"
            sleep_diag["raw_kind_detected"] = int(block["raw_kind"]) if pd.notna(block["raw_kind"]) else None
            sleep_diag["raw_kind_block_avg_hr"] = round(float(block["avg_hr"]), 1) if pd.notna(block["avg_hr"]) else None
            sleep_diag["note"] = (f"Dedicated sleep-stage table had no data for this night, so this total comes from the longest "
                                   f"contiguous RAW_KIND={sleep_diag['raw_kind_detected']} block on the activity table "
                                   f"(avg {sleep_diag['raw_kind_block_avg_hr']} bpm vs {day_baseline_hr:.0f} bpm day average). "
                                   "Deep/Light/REM breakdown isn't available from this source, total duration only.")
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

    # Resting-proxy HRV fallback if no real sleep HR was found at all
    if sleep_hr_data.empty and not activity_df.empty and 'Date' in activity_df.columns:
        day_df = activity_df[activity_df['Date'] == target_date]
        hr_clean = day_df['HEART_RATE'].dropna() if 'HEART_RATE' in day_df.columns else pd.Series(dtype=float)
        if not hr_clean.empty:
            sleep_hr_data = hr_clean[hr_clean <= hr_clean.quantile(0.25)]
            if not sleep_hr_data.empty:
                result["hrv_source"] = "resting_proxy"

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
                             deep_codes_str, light_codes_str, rem_codes_str, awake_codes_str,
                             max_hr_override, resting_hr_override, sensitivity):
    """Day Strain (0-21): duration-weighted heart-rate-reserve load via Banister TRIMP.
    Whoop's own algorithm is proprietary and unpublished. Resting HR defaults to the night's
    actual overnight HR; minutes are smoothed and gated so ordinary daytime drift doesn't count
    as load; and "today" starts at wake time rather than midnight, matching how Whoop defines a
    day (sleep-to-sleep), since evening activity before falling asleep otherwise bleeds into the
    next day's score."""
    sleep_info = compute_sleep_and_hrv(target_date, activity_table, active_sleep_table, mtime, tz_name,
                                        deep_codes_str, light_codes_str, rem_codes_str, awake_codes_str)
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
        strain = min(21.0, round(sensitivity * 1.9 * np.log1p(trimp / 0.5), 1))

    total_steps = day_df['STEPS'].sum() if not day_df.empty and 'STEPS' in day_df.columns else 0
    avg_hr = day_df['HEART_RATE'].mean() if not day_df.empty and 'HEART_RATE' in day_df.columns else np.nan

    return {"strain_score": strain, "total_steps": total_steps, "avg_hr": avg_hr,
            "hrv_val": sleep_info["hrv_val"], "sleep_duration_mins": sleep_info["sleep_duration_mins"]}


# --- Daily Journal: simple per-day txt logs + trend comparisons ----------------------------
JOURNAL_DIR = "journal"


def _journal_path(d):
    return os.path.join(JOURNAL_DIR, f"{d.isoformat()}.txt")


def load_journal_entry(d):
    path = _journal_path(d)
    tags, notes = {}, ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        in_notes = False
        notes_lines = []
        for line in lines:
            if in_notes:
                notes_lines.append(line)
                continue
            if line.strip().startswith("Notes:"):
                in_notes = True
                rest = line.split(":", 1)[1].strip()
                if rest:
                    notes_lines.append(rest)
                continue
            if ":" in line:
                tag, val = line.split(":", 1)
                tags[tag.strip()] = val.strip().lower() in ("yes", "true", "y", "1")
        notes = "\n".join(notes_lines).strip()
    return tags, notes


def save_journal_entry(d, tag_values, notes):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    lines = [f"{tag}: {'Yes' if val else 'No'}" for tag, val in tag_values.items()]
    lines.append("Notes:")
    if notes:
        lines.append(notes)
    with open(_journal_path(d), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def has_journal_entry(d):
    return os.path.exists(_journal_path(d))



if not df_activity.empty and 'Date' in df_activity.columns:
    available_dates = sorted(df_activity['Date'].dropna().unique(), reverse=True)
else:
    available_dates = []

if available_dates:
    day_info_by_date = {}
    with st.spinner("Loading history..."):
        for d in available_dates:
            day_info_by_date[d] = compute_strain_for_date(
                d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                st.session_state.deep_codes, st.session_state.light_codes,
                st.session_state.rem_codes, st.session_state.awake_codes,
                st.session_state.max_hr_override, st.session_state.resting_hr_override, st.session_state.strain_sensitivity)

    st.sidebar.caption("🗓️ Date — Strain / HRV")

    def _fmt_date(d):
        info = day_info_by_date.get(d, {})
        strain, hrv = info.get("strain_score"), info.get("hrv_val")
        strain_str = f"{strain:.1f}" if pd.notna(strain) else "—"
        hrv_str = f"{int(hrv)}ms" if pd.notna(hrv) else "—"
        return f"{d.strftime('%a %d %b')}  —  🔥{strain_str}  ⚡{hrv_str}"

    selected_date = st.sidebar.radio("Select a day to analyze", available_dates, format_func=_fmt_date,
                                      label_visibility="collapsed")
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
    st.caption("🌙 Sleep Stage Codes (optional)")
    st.caption("Xiaomi's raw stage codes aren't publicly documented and vary by device/firmware. Check '⚙️ Database Transparency' below for a duration-per-code breakdown, compare against the Gadgetbridge app's own sleep report, and adjust these if they don't match.")
    st.text_input("Deep sleep code(s)", key="deep_codes")
    st.text_input("Light sleep code(s)", key="light_codes")
    st.text_input("REM sleep code(s)", key="rem_codes")
    st.text_input("Awake code(s), comma-separated (blank = none excluded)", key="awake_codes")

    st.divider()
    st.caption("📔 Journal Tags (optional)")
    st.text_area("Comma-separated list of yes/no tags to track each day", key="journal_tags", height=80)

    st.divider()
    st.caption(f"Active Activity Source: `{activity_table}`")
    if active_sleep_table:
        st.caption(f"Sleep Source: `{active_sleep_table}`")

max_hr_override = st.session_state.max_hr_override
resting_hr_override = st.session_state.resting_hr_override

target_df = df_activity[df_activity['Date'] == selected_date] if not df_activity.empty and 'Date' in df_activity.columns else pd.DataFrame()
last_7_days = df_activity[(df_activity['Date'] > (selected_date - timedelta(days=7))) & (df_activity['Date'] <= selected_date)] if not df_activity.empty and 'Date' in df_activity.columns else pd.DataFrame()

sleep_info = compute_sleep_and_hrv(selected_date, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                    st.session_state.deep_codes, st.session_state.light_codes,
                                    st.session_state.rem_codes, st.session_state.awake_codes)
sleep_duration_mins = sleep_info["sleep_duration_mins"]
deep_sleep_mins = sleep_info["deep_sleep_mins"]
light_sleep_mins = sleep_info["light_sleep_mins"]
rem_sleep_mins = sleep_info["rem_sleep_mins"]
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
                                       st.session_state.deep_codes, st.session_state.light_codes,
                                       st.session_state.rem_codes, st.session_state.awake_codes,
                                       max_hr_override, resting_hr_override, st.session_state.strain_sensitivity)
strain_score = strain_info["strain_score"]

# Global Metric Hunter for SpO2 & Stress
def fetch_global_metric(keywords):
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
                        day_data = pd.to_numeric(temp[dates == selected_date][val_col], errors='coerce')
                        day_data = day_data.replace([0, 255], np.nan).dropna()
                        if not day_data.empty:
                            return day_data.mean()
                except Exception:
                    continue
    return np.nan

spo2_val = fetch_global_metric(["SPO2", "OXYGEN"])
stress_val = fetch_global_metric(["STRESS"])

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

# --- Dashboard UI Layout ---
st.subheader(f"📊 Daily Report: {selected_date.strftime('%A, %B %d, %Y')}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="🔥 Day Strain (0-21)", value=f"{strain_score:.1f}")
with col2:
    hrv_caption = {"sleep": "from overnight HR", "resting_proxy": "resting-proxy, not real sleep HRV", "none": ""}[hrv_source]
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
    if deep_sleep_mins > 0 or light_sleep_mins > 0:
        d_h, d_m = int(deep_sleep_mins // 60), int(deep_sleep_mins % 60)
        l_h, l_m = int(light_sleep_mins // 60), int(light_sleep_mins % 60)
        r_h, r_m = int(rem_sleep_mins // 60), int(rem_sleep_mins % 60)
        stage_text = f"Deep: {d_h}h {d_m}m | Light: {l_h}h {l_m}m"
        if rem_sleep_mins > 0:
            stage_text += f" | REM: {r_h}h {r_m}m"
        st.metric(label="🌊 Sleep Stages", value=stage_text)
    elif sleep_diag.get("source") in ("activity_table_fallback_total_only", "activity_table_raw_kind_total_only"):
        st.metric(label="🌊 Sleep Stages", value="Unavailable", delta="total-only source, see diagnostics", delta_color="off")
    else:
        st.metric(label="🌊 Deep Sleep / Cycles", value="N/A")
with col7:
    st.metric(label="🩸 Blood Oxygen (SpO2)", value=f"{spo2_val:.1f}%" if pd.notna(spo2_val) else "No Sensor Data")
with col8:
    st.metric(label="🧠 Stress Score", value=f"{stress_val:.0f}" if pd.notna(stress_val) else "No Sensor Data")
    if pd.notna(stress_val):
        label, note = classify_stress(stress_val)
        st.caption(f"{label} — {note}")

st.divider()

st.subheader(f"⏳ Intraday Heart Rate ({selected_date.strftime('%b %d')})")
if not hr_data.empty:
    st.line_chart(target_df.set_index('Datetime')['HEART_RATE'], color="#ff4b4b")
else:
    st.info("No intraday heart rate data available for this specific date.")

st.subheader("👟 7-Day Step Trend")
if not last_7_days.empty and "STEPS" in last_7_days.columns and last_7_days['STEPS'].sum() > 0:
    daily_steps = last_7_days.groupby(last_7_days['Date'])['STEPS'].sum().reset_index()
    daily_steps['Day'] = daily_steps['Date'].apply(lambda d: d.strftime('%a %d'))
    step_chart = alt.Chart(daily_steps).mark_bar(color="#29b5e8").encode(
        x=alt.X('Day:O', sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('STEPS:Q', title='Steps'),
        tooltip=[alt.Tooltip('Day:N', title='Day'), alt.Tooltip('STEPS:Q', title='Steps')],
    ).properties(height=280)
    st.altair_chart(step_chart, width='stretch')
else:
    st.info("No step data available.")

st.divider()

st.subheader("📔 Daily Journal")
journal_tags = [t.strip() for t in st.session_state.journal_tags.replace("\n", ",").split(",") if t.strip()]
existing_tags, existing_notes = load_journal_entry(selected_date)

tag_values = {}
n_cols = 4
cols = st.columns(n_cols)
for i, tag in enumerate(journal_tags):
    with cols[i % n_cols]:
        tag_values[tag] = st.checkbox(tag, value=existing_tags.get(tag, False),
                                       key=f"journal_{tag}_{selected_date.isoformat()}")
with cols[len(journal_tags) % n_cols]:
    with st.popover("➕ Add tag"):
        new_tag_input = st.text_input("New tag name", key="new_journal_tag_input")
        if st.button("Add", key="add_journal_tag_btn") and new_tag_input.strip():
            st.session_state["_pending_new_journal_tag"] = new_tag_input.strip()
            st.rerun()

if not journal_tags:
    st.caption("No journal tags yet - use the ➕ button above to add one, or set a list in ⚙️ Advanced Settings.")

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
            tags, _ = load_journal_entry(d)
            s_info = compute_strain_for_date(d, activity_table, active_sleep_table, _db_mtime(), st.session_state.tz_name,
                                              st.session_state.deep_codes, st.session_state.light_codes,
                                              st.session_state.rem_codes, st.session_state.awake_codes,
                                              max_hr_override, resting_hr_override, st.session_state.strain_sensitivity)
            rows.append({"tags": tags, "strain": s_info["strain_score"], "hrv": s_info["hrv_val"],
                         "sleep_mins": s_info["sleep_duration_mins"]})

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
                "Strain (Yes)": _avg("strain", yes_rows), "Strain (No)": _avg("strain", no_rows),
                "HRV (Yes)": _avg("hrv", yes_rows), "HRV (No)": _avg("hrv", no_rows),
                "Sleep (Yes)": _avg("sleep_mins", yes_rows), "Sleep (No)": _avg("sleep_mins", no_rows),
                "Correlated With": correlated or "—",
            })

        if not comparison_rows:
            st.caption("Not enough variation yet - each tag needs at least one 'Yes' day and one 'No' day to compare.")
        else:
            comp_df = pd.DataFrame(comparison_rows)
            for col in ["Sleep (Yes)", "Sleep (No)"]:
                comp_df[col] = comp_df[col].apply(lambda m: f"{int(m // 60)}h {int(m % 60)}m" if pd.notna(m) else "—")
            for col in ["Strain (Yes)", "Strain (No)"]:
                comp_df[col] = comp_df[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
            for col in ["HRV (Yes)", "HRV (No)"]:
                comp_df[col] = comp_df[col].apply(lambda v: f"{int(v)} ms" if pd.notna(v) else "—")
            st.dataframe(comp_df, hide_index=True, width='stretch')
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
                    line = f"**{tag}** vs no {tag}, on days without **{other}** (n={len(yes_s)}/{len(no_s)}): "
                    parts = []
                    if pd.notna(s_yes) and pd.notna(s_no):
                        parts.append(f"Strain {s_yes:.1f} vs {s_no:.1f}")
                    if pd.notna(h_yes) and pd.notna(h_no):
                        parts.append(f"HRV {int(h_yes)}ms vs {int(h_no)}ms")
                    st.markdown(line + ", ".join(parts) if parts else line + "not enough data")
            if not any_controlled:
                st.caption("No controlled comparisons available yet - need days where a correlated tag stays constant while the other one varies.")



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

    st.markdown("**Activity/Steps diagnostics for the selected day**")
    if not target_df.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows this day", len(target_df))
        c2.metric("Rows with STEPS > 0", int((target_df['STEPS'] > 0).sum()) if 'STEPS' in target_df.columns else 0)
        c3.metric("Rows with HR reading", int(target_df['HEART_RATE'].notna().sum()) if 'HEART_RATE' in target_df.columns else 0)
        st.caption("If HR rows vastly outnumber STEPS>0 rows, the device is likely syncing continuous background HR faster/more reliably than its step-history batch, or the STEPS field for this firmware needs a different column/scale - compare against the raw preview below.")
    st.write("Activity Table Preview:")
    st.dataframe(target_df.head(50))