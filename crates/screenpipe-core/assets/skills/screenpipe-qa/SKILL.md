---
name: screenpipe-qa
description: Run pre-release QA tests on screenpipe using desktop automation (BigBrother CLI) and API checks. Use when the user asks to test screenpipe, run QA, verify the pipeline works, or check for regressions before a release.
---

# Screenpipe QA Testing

Automated pre-release QA for screenpipe. Uses BigBrother CLI (`bb`) for desktop automation and screenpipe APIs for pipeline verification.

The screenpipe API runs at `http://localhost:3030`.

## Shell

- **macOS/Linux** → `bash`, `curl`
- **Windows** → `powershell`, `curl.exe`

All examples below use bash.

## BigBrother CLI (bb)

`bb` is a desktop automation tool. Binary location: `$HOME/Documents/bigbrother/target/release/bb`

If `bb` is not in PATH, use the full path. If the binary is not found, ask the user where it is installed.

### bb Commands

| Command | Description | Example |
|---------|-------------|---------|
| `bb open <URL>` | Open a URL in default browser | `bb open "https://youtube.com/watch?v=..."` |
| `bb apps` | List running applications | `bb apps` |
| `bb activate <APP>` | Bring app to front | `bb activate "screenpipe"` |
| `bb screenshot -o <path>` | Take a screenshot | `bb screenshot -o /tmp/test.png` |
| `bb scrape --app <APP>` | Scrape text from app window | `bb scrape --app "screenpipe"` |
| `bb find <SELECTOR> --app <APP>` | Find UI element | `bb find "button:Settings" --app "screenpipe"` |
| `bb click <SELECTOR> --app <APP>` | Click UI element | `bb click "button:Settings" --app "screenpipe"` |
| `bb type <TEXT> --app <APP>` | Type text | `bb type "hello world" --app "screenpipe"` |
| `bb shortcut <KEY> --modifiers <MODS>` | Press keyboard shortcut | `bb shortcut "space" --modifiers "alt"` |
| `bb wait --idle <SECS>` | Wait for system idle | `bb wait --idle 3` |
| `bb tree --app <APP> --depth <N>` | Dump accessibility tree | `bb tree --app "screenpipe" --depth 3` |
| `bb scroll --direction <DIR> --app <APP>` | Scroll | `bb scroll --direction down --pages 2 --app "screenpipe"` |
| `bb press <KEY>` | Press a key | `bb press "escape"` |

## QA Test Phases

Run these phases in order. Each phase builds on the previous one.

### Phase 1: Health & Prerequisites

Verify screenpipe is running and healthy before doing anything else.

```bash
# 1. Check screenpipe process is running
pgrep -f screenpipe-app || echo "FAIL: screenpipe not running"

# 2. Check API is responding
curl -sf http://localhost:3030/health | head -c 500

# 3. Check port is bound
lsof -i :3030 -t 2>/dev/null || echo "WARN: lsof may hang, skip if no output in 5s"
```

**Pass criteria from /health response:**
- `status` should NOT be `"unhealthy"`
- `frame_status` should be `"ok"` (not `"stale"` or `"no data"`)
- `audio_status` should be `"ok"`
- `frame_drop_rate` should be < 0.1 (10%)
- `pipeline_stall_count` should be 0
- `transcription_error_count` should be low (< 5 in normal operation)
- `timestamps_status.is_healthy` should be `true`

```bash
# Parse key health metrics
HEALTH=$(curl -sf http://localhost:3030/health)
echo "$HEALTH" | python3 -c "
import json, sys
h = json.load(sys.stdin)
checks = [
    ('status', h.get('status') != 'unhealthy'),
    ('frame_status', h.get('frame_status') == 'ok'),
    ('audio_status', h.get('audio_status') == 'ok'),
    ('frame_drop_rate', (h.get('frame_drop_rate') or 0) < 0.1),
    ('pipeline_stalls', (h.get('pipeline_stall_count') or 0) == 0),
    ('transcription_errors', (h.get('transcription_error_count') or 0) < 5),
]
for name, ok in checks:
    print(f'  {\"PASS\" if ok else \"FAIL\"}: {name}')
"
```

### Phase 2: Pipeline E2E Test (YouTube Video)

Play a known video to test the full capture pipeline: screen → OCR → frames + audio → transcription.

**Test video:** `https://www.youtube.com/watch?v=zlDmYkeQpVQ` (Elon Musk motivational speech — contains clear speech and on-screen text)

```bash
# 1. Record the current time (UTC) as baseline
START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# 2. Open the YouTube video
bb open "https://www.youtube.com/watch?v=zlDmYkeQpVQ"

# 3. Wait for page to load and video to start playing
sleep 10

# 4. Take a screenshot to verify video is visible
bb screenshot -o /tmp/qa-youtube-before.png

# 5. Let the video play for 60 seconds to capture enough data
sleep 60

# 6. Take another screenshot
bb screenshot -o /tmp/qa-youtube-after.png

# 7. Record end time
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
```

Now verify data was captured:

```bash
# Check OCR captured YouTube content
curl -sf "http://localhost:3030/search?content_type=ocr&app_name=Google%20Chrome&limit=5&start_time=$START_TIME&end_time=$END_TIME"

# Also try Safari / Arc / other browsers
curl -sf "http://localhost:3030/search?content_type=ocr&limit=5&start_time=$START_TIME&end_time=$END_TIME"

# Check audio transcription captured speech
curl -sf "http://localhost:3030/search?content_type=audio&limit=5&start_time=$START_TIME&end_time=$END_TIME"

# Check accessibility data
curl -sf "http://localhost:3030/search?content_type=accessibility&limit=5&start_time=$START_TIME&end_time=$END_TIME"
```

**Pass criteria:**
- OCR results should contain text from the YouTube page (video title, channel name, or on-screen text)
- Audio results should contain transcription of the speech within 30-60 seconds of playback
- At least 1 frame should have `app_name` matching the browser
- The search response should have `pagination.total > 0`

### Phase 3: Search & Database Verification

Test that search works across all content types.

```bash
# Keyword search (FTS)
curl -sf "http://localhost:3030/search/keyword?q=youtube&limit=5&start_time=$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ)"

# List audio devices (should return non-empty array)
curl -sf http://localhost:3030/audio/list

# List vision monitors (should return non-empty array)
curl -sf http://localhost:3030/vision/list

# Raw SQL: count recent frames
curl -sf -X POST http://localhost:3030/raw_sql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) as frame_count FROM frames WHERE timestamp > datetime(\"now\", \"-10 minutes\") LIMIT 1"}'

# Raw SQL: count recent audio transcriptions
curl -sf -X POST http://localhost:3030/raw_sql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) as transcription_count FROM audio_transcriptions WHERE timestamp > datetime(\"now\", \"-10 minutes\") LIMIT 1"}'

# Raw SQL: count recent OCR text
curl -sf -X POST http://localhost:3030/raw_sql \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT COUNT(*) as ocr_count FROM ocr_text o JOIN frames f ON o.frame_id = f.id WHERE f.timestamp > datetime(\"now\", \"-10 minutes\") LIMIT 1"}'
```

**Pass criteria:**
- Audio device list is non-empty
- Vision monitor list is non-empty
- Recent frame count > 0 (at ~0.5 fps, should be ~300 per 10 min)
- Recent transcription count > 0 if audio is enabled
- Recent OCR count > 0

### Phase 4: App UI Smoke Test

Use bb to interact with the screenpipe desktop app.

```bash
# 1. Activate screenpipe app
bb activate "screenpipe"
sleep 2

# 2. Take screenshot of main window
bb screenshot -o /tmp/qa-app-main.png

# 3. Dump accessibility tree to see available UI elements
bb tree --app "screenpipe" --depth 3

# 4. Try the overlay shortcut (Alt+Space by default)
bb shortcut "space" --modifiers "alt"
sleep 2
bb screenshot -o /tmp/qa-overlay.png

# 5. Close overlay
bb press "escape"
sleep 1
```

**Pass criteria:**
- `bb activate "screenpipe"` succeeds (app is running)
- Screenshot shows screenpipe UI (not blank/crashed)
- Overlay appears on shortcut press
- Overlay closes on Escape

### Phase 5: Log Analysis

Check logs for errors, panics, and known bad patterns.

```bash
# Find today's log file
LOG_FILE="$HOME/.screenpipe/screenpipe-app.$(date +%Y-%m-%d).log"

if [ -f "$LOG_FILE" ]; then
    echo "=== Errors ==="
    grep -c -E "ERROR|error" "$LOG_FILE" || echo "0 errors"

    echo "=== Panics ==="
    grep -c "panic" "$LOG_FILE" || echo "0 panics"

    echo "=== DB contention ==="
    grep -c "Slow DB" "$LOG_FILE" || echo "0 slow DB"

    echo "=== Pipeline stalls ==="
    grep -c "stall" "$LOG_FILE" || echo "0 stalls"

    echo "=== Audio issues ==="
    grep -c -E "audio.*timeout|audio.*error" "$LOG_FILE" || echo "0 audio issues"

    echo "=== Queue stats (last 5) ==="
    grep "Queue stats" "$LOG_FILE" | tail -5

    echo "=== Last 20 lines ==="
    tail -20 "$LOG_FILE"
else
    echo "WARN: Log file not found at $LOG_FILE"
fi
```

**Pass criteria:**
- 0 panics
- Error count < 10 in last hour of logs
- 0 "Slow DB" warnings > 3 seconds
- 0 pipeline stalls
- Queue stats show non-zero processed counts

### Phase 6: Process & Resource Check

```bash
# Check for orphaned processes
ps aux | grep -E "screenpipe|ffmpeg" | grep -v grep

# Check memory usage (should be < 2GB for the main process)
ps aux | grep screenpipe-app | grep -v grep | awk '{print "RSS: " $6/1024 " MB", "PID: " $2}'

# Check disk usage of screenpipe data
du -sh ~/.screenpipe/data/ 2>/dev/null || echo "WARN: no data dir"
du -sh ~/.screenpipe/*.db 2>/dev/null || echo "WARN: no db files"

# Check no port conflicts
timeout 5 lsof -i :3030 2>/dev/null || echo "WARN: lsof timeout (known macOS issue, not a bug)"
```

**Pass criteria:**
- Only expected screenpipe processes running (app + sidecar, possibly ffmpeg)
- Memory < 2GB for main process
- No zombie ffmpeg processes

## Running the Full QA Suite

To run all phases sequentially:

1. Make sure screenpipe is running and recording
2. Run Phase 1 — if it fails, fix issues before continuing
3. Run Phase 2 — this takes ~90 seconds (video playback)
4. Run Phase 3 — verify data from Phase 2 was captured
5. Run Phase 4 — test the desktop UI
6. Run Phase 5 — check logs for issues
7. Run Phase 6 — verify resources are healthy

## Quick Smoke Test (2 minutes)

If you need a fast check, run only these:

```bash
# Health
curl -sf http://localhost:3030/health | python3 -c "import json,sys; h=json.load(sys.stdin); print('status:', h.get('status'), '| frames:', h.get('frame_status'), '| audio:', h.get('audio_status'))"

# Recent data exists
curl -sf "http://localhost:3030/search?content_type=all&limit=3&start_time=$(date -u -v-5M +%Y-%m-%dT%H:%M:%SZ)"

# Devices detected
curl -sf http://localhost:3030/audio/list | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), 'audio devices')"
curl -sf http://localhost:3030/vision/list | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d), 'monitors')"

# Log errors
grep -c "panic\|ERROR" "$HOME/.screenpipe/screenpipe-app.$(date +%Y-%m-%d).log" 2>/dev/null || echo "0"
```

## Test Report Format

After running QA, summarize results like this:

```
## QA Report — screenpipe vX.Y.Z

**Date:** YYYY-MM-DD
**Platform:** macOS / Windows / Linux
**Duration:** X minutes

### Results

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Health | PASS/FAIL | status, frame_drop_rate, stalls |
| 2. Pipeline E2E | PASS/FAIL | OCR captured, audio transcribed |
| 3. Search & DB | PASS/FAIL | all content types searchable |
| 4. App UI | PASS/FAIL | overlay works, shortcut works |
| 5. Logs | PASS/FAIL | X errors, 0 panics |
| 6. Resources | PASS/FAIL | memory, processes |

### Issues Found
- (list any failures with details)

### Screenshots
- /tmp/qa-youtube-before.png
- /tmp/qa-youtube-after.png
- /tmp/qa-app-main.png
- /tmp/qa-overlay.png
```

## Tips

- If `bb` is not found, check `$HOME/Documents/bigbrother/target/release/bb` or ask the user.
- The YouTube video test takes ~90 seconds. Audio transcription may lag 15-30 seconds behind.
- `lsof` is known to hang on some macOS systems. Always use `timeout 5 lsof ...` or skip it.
- If health returns `"status": "unhealthy"`, check logs immediately — don't proceed with other tests.
- Screenshots taken with `bb screenshot` can be read with the `read` tool to visually inspect results.
- If OCR results are empty after the video test, check that the browser is in the `vision/list` monitors and not excluded.
- Run QA on a **release build** — debug builds have 3-5x higher CPU usage which skews resource checks.
