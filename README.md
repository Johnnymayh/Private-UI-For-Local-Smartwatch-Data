# ⚡ Personal Health Dashboard

A Whoop-style recovery/strain dashboard that runs entirely locally against your own
[Gadgetbridge](https://gadgetbridge.org/) database, no cloud, no account, no data leaving
your machine. Built for Xiaomi/Huami wearables synced through Gadgetbridge.

## Features

- **Day Strain (0–21)**  a Banister TRIMP-based load score, using your actual overnight
  heart rate as the resting baseline and anchoring "today" to your wake time rather than
  midnight, so evening activity before you fell asleep doesn't bleed into the next day's score.
- **Sleep duration & stages**  parses your dedicated sleep-stage table when your device
  populates one; falls back to detecting the longest contiguous "sleep" block from the raw
  activity table's per-minute activity-kind codes when it doesn't (common on several
  Xiaomi/Huami device + firmware combinations).
- **HRV (RMSSD)**  computed from real overnight heart rate when available, clearly labeled
  as a same-day resting-proxy estimate when it isn't.
- **Stress score**  with a plain-language "relaxed / normal / medium / high" note.
- **Timezone-aware**  all timestamps in the database are UTC; this is converted to your
  local time (configurable, DST-aware) rather than displayed raw.
- **Date history sidebar**  scroll back through every day in your database, with that
  night's HRV shown alongside each date.
- **Full diagnostics panel**  every table in your database, per-night raw sleep-code
  durations, and per-day activity stats, so you can see exactly why a number looks the way
  it does and recalibrate the sidebar settings to match your specific device.
  **Private Customisable Journal** you can make a journal text document with your own personal
  customisable yes/no inputs. You save the text file locally and input it with context for each
  day, so that the trends of how these inputs affect your data can be calculated. The text file 
  should be in the same directory as your GadgetBridge dataset and the dashboard files.

## Setup

1. **Install Python 3.10+** if you don't already have it.
2. Clone or download this repo.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Export your database from Gadgetbridge (**Settings → Data management → Export
   database**) and place the file  named exactly `Gadgetbridge`, no extension  in this
   same folder.
5. Run it:
   - **Windows:** double-click `run_dashboard.bat`.
   - **Any OS:** `streamlit run app.py` from this folder.

Your browser will open automatically. Re-export and drop in a fresh `Gadgetbridge` file any
time you want to pull in newer data  the app detects the file change and refreshes on its
own (there's also a "Force-refresh data" button in the sidebar's Advanced Settings).

## Calibrating it to your device

Xiaomi/Huami's raw sleep-stage and activity codes aren't publicly documented and vary by
device and firmware. Defaults here were reverse-engineered against a Xiaomi Smart Band 7 
they may need adjusting for other devices. Open **⚙️ Advanced Settings** in the sidebar for:

- **Timezone**  IANA name (e.g. `Europe/London`, `America/New_York`).
- **Max/Resting Heart Rate** and **Strain Sensitivity**  tune Day Strain to match how hard
  your days actually feel.
- **Sleep Stage Codes**  if Deep/Light/REM don't look right, expand **"⚙️ Database
  Transparency"** on the main page for a duration-per-raw-code breakdown, compare it against
  Gadgetbridge's own sleep report, and enter the correct codes here.


## License

MIT  do whatever you like with it.
