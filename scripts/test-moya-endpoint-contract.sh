#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${SCREENPIPE_BIN:-"$ROOT_DIR/target/release/screenpipe"}"
REPORT_DIR="${SCREENPIPE_TEST_REPORT_DIR:-"$ROOT_DIR/target/moya-endpoint-contract-$(date +%Y%m%d-%H%M%S)"}"
DATA_DIR="${SCREENPIPE_TEST_DATA_DIR:-"$(mktemp -d "${TMPDIR:-/tmp}/screenpipe-moya-data.XXXXXX")"}"
MARKER="${SCREENPIPE_TEST_MARKER:-"MOYA_SCREENPIPE_CONTRACT_$(date +%s)"}"

PORT="${SCREENPIPE_TEST_PORT:-"$(python3 - <<'PY'
import socket

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"}"

mkdir -p "$REPORT_DIR/responses"

MARKER_HTML="$REPORT_DIR/moya-contract-marker.html"
cat >"$MARKER_HTML" <<HTML
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Moya Screenpipe Endpoint Contract $MARKER</title>
    <style>
      body { font: 18px -apple-system, BlinkMacSystemFont, sans-serif; margin: 48px; line-height: 1.5; }
      textarea { width: 720px; height: 160px; font: 16px ui-monospace, SFMono-Regular, Menlo, monospace; }
    </style>
  </head>
  <body>
    <h1>Moya Screenpipe Endpoint Contract</h1>
    <p id="marker">$MARKER visible browser text for OCR and accessibility capture.</p>
    <textarea id="focused">Focused input marker: $MARKER selected text and typed context for active window extraction.</textarea>
    <script>
      const input = document.getElementById("focused");
      input.focus();
      const start = input.value.indexOf("$MARKER");
      input.setSelectionRange(start, start + "$MARKER".length);
    </script>
  </body>
</html>
HTML

if [[ "${SCREENPIPE_SKIP_OPEN_MARKER_PAGE:-0}" != "1" ]]; then
  open "file://$MARKER_HTML" >/dev/null 2>&1 || true
  sleep "${SCREENPIPE_MARKER_FOCUS_DELAY:-2}"
fi

LOG_FILE="$REPORT_DIR/screenpipe.log"
"$BIN" record \
  --disable-audio \
  --disable-telemetry \
  --audio-transcription-engine whisper-tiny-quantized \
  --port "$PORT" \
  --data-dir "$DATA_DIR" \
  --enable-accessibility \
  --enable-main-body-distillation \
  --enable-input-capture \
  >"$LOG_FILE" 2>&1 &

SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

BASE_URL="http://127.0.0.1:$PORT"
BASE_URL="$BASE_URL" REPORT_DIR="$REPORT_DIR" DATA_DIR="$DATA_DIR" MARKER="$MARKER" LOG_FILE="$LOG_FILE" python3 - <<'PY'
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

base_url = os.environ["BASE_URL"]
report_dir = Path(os.environ["REPORT_DIR"])
responses_dir = report_dir / "responses"
marker = os.environ["MARKER"]
data_dir = os.environ["DATA_DIR"]
log_file = os.environ["LOG_FILE"]

results = []
capture_metadata = {}


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def iso_ms(value):
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_response(label, status, elapsed_ms, body, error=None):
    suffix = "json" if isinstance(body, (dict, list)) else "txt"
    path = responses_dir / f"{label}.{suffix}"
    payload = body
    if isinstance(body, (dict, list)):
        path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    else:
        path.write_text(body or "", encoding="utf-8")
    meta = {
        "label": label,
        "status": status,
        "elapsed_ms": round(elapsed_ms, 1),
        "response_file": str(path),
    }
    if error:
        meta["error"] = error
    return meta, payload


def request_json(label, method, path, *, params=None, body=None, timeout=2.0):
    url = f"{base_url}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.monotonic()
    status = None
    raw = ""
    error = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.monotonic() - started) * 1000
    parsed = None
    parse_error = None
    if raw:
        try:
            parsed = json.loads(raw)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    body_for_file = parsed if parsed is not None else raw
    meta, payload = write_response(label, status, elapsed_ms, body_for_file, error)
    if parse_error:
        meta["json_error"] = parse_error
    meta["url"] = url
    meta["method"] = method
    return meta, parsed


def first_text(value):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def capture_sources(body):
    if not isinstance(body, dict):
        return []
    sources = []
    for field in ("main_body_text", "ocr_text"):
        text = first_text(body.get(field))
        if text:
            sources.append(field)
    focused = body.get("focused_accessibility")
    if isinstance(focused, dict):
        for field in ("input_text", "selected_text"):
            text = first_text(focused.get(field))
            if text:
                sources.append(f"focused_accessibility.{field}")
    for field in ("frame", "screenshot", "screenshot_base64"):
        text = first_text(body.get(field))
        if text:
            sources.append(field)
    structured = body.get("structured_messages")
    if isinstance(structured, list) and any(first_text(item.get("body")) for item in structured if isinstance(item, dict)):
        sources.append("structured_messages[].body")
    return sources


def capture_observed_metadata(body):
    if not isinstance(body, dict):
        return {}
    structured_meta = body.get("structured_meta") if isinstance(body.get("structured_meta"), dict) else {}
    return {
        "app_name": body.get("app_name"),
        "app_bundle_id": body.get("app_bundle_id"),
        "owner_pid": body.get("owner_pid"),
        "window_title": body.get("window_title") or body.get("window_name"),
        "url": body.get("url") or body.get("browser_url"),
        "timestamp": body.get("timestamp") or body.get("captured_at"),
        "text_source": body.get("text_source"),
        "capture_provenance": body.get("capture_provenance"),
        "content_kind": body.get("content_kind"),
        "structured_meta.status": structured_meta.get("status"),
    }


def classify_error(meta, body):
    text = json.dumps(body, sort_keys=True) if isinstance(body, (dict, list)) else ""
    text = f"{text} {meta.get('error', '')} {meta.get('json_error', '')}".lower()
    if "permission" in text or "tcc" in text or "accessibility" in text or "screen_recording" in text:
        return "permission/TCC"
    if meta.get("status") in (500, 504):
        return "merge_or_runtime"
    if meta.get("error"):
        return "environment"
    return "contract"


def record_result(kind, name, meta, body, compatible, details):
    results.append({
        "kind": kind,
        "name": name,
        "status": meta.get("status"),
        "elapsed_ms": meta.get("elapsed_ms"),
        "compatible": bool(compatible),
        "classification": "pass" if compatible else classify_error(meta, body),
        "details": details,
        "response_file": meta.get("response_file"),
        "url": meta.get("url"),
    })


def wait_for_server():
    deadline = time.monotonic() + 60
    last_meta = None
    last_body = None
    while time.monotonic() < deadline:
        meta, body = request_json("readiness_health", "GET", "/health", timeout=2.0)
        last_meta, last_body = meta, body
        if meta.get("status") and 200 <= meta["status"] < 300:
            return True, meta, body
        time.sleep(1)
    return False, last_meta, last_body


def useful_rows(body):
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict):
        rows = None
        for key in ("data", "results", "rows", "items"):
            if isinstance(body.get(key), list):
                rows = body[key]
                break
        if rows is None:
            rows = []
    else:
        rows = []

    useful = []
    for row in rows:
        candidates = [row]
        if isinstance(row, dict) and isinstance(row.get("content"), dict):
            candidates.append(row["content"])
        found_text = None
        found_meta = {}
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            for field in ("text", "content", "ocr_text", "ocrText", "chunk", "transcript", "transcription", "text_content"):
                found_text = first_text(candidate.get(field))
                if found_text:
                    break
            if found_text:
                found_meta = {
                    "app_name": candidate.get("app_name"),
                    "window_name": candidate.get("window_name") or candidate.get("window_title"),
                    "timestamp": candidate.get("timestamp"),
                    "frame": bool(candidate.get("frame")),
                    "frame_id": candidate.get("frame_id"),
                    "focused": candidate.get("focused"),
                }
                break
        if found_text:
            useful.append({"text": found_text, "metadata": found_meta})
    return rows, useful


ready, ready_meta, ready_body = wait_for_server()
if not ready:
    record_result(
        "server",
        "readiness",
        ready_meta or {"status": None, "elapsed_ms": None, "response_file": None, "error": "server_not_ready"},
        ready_body,
        False,
        {"reason": "server did not return 2xx from /health within 60s", "log_file": log_file},
    )
else:
    health_meta, health_body = request_json("health", "GET", "/health", timeout=2.0)
    record_result(
        "health",
        "GET /health",
        health_meta,
        health_body,
        bool(health_meta.get("status") and 200 <= health_meta["status"] < 300 and health_meta["elapsed_ms"] <= 2000),
        {"pass_condition": "2xx within 2s"},
    )

    profiles = [
        ("capture_health_probe", "Health probe", 2.0, {"strict": False, "include_ocr": False, "include_structured_messages": True, "structured_timeout_ms": 250}),
        ("capture_standard", "Standard", 3.0, {"strict": False, "include_ocr": True, "include_structured_messages": True, "structured_timeout_ms": 250}),
        ("capture_assistant_live", "Assistant live context", 5.0, {"strict": False, "include_ocr": True, "include_structured_messages": True, "structured_timeout_ms": 1200}),
        ("capture_assistant_materialization", "Assistant materialization", 8.0, {"strict": False, "include_ocr": True, "include_structured_messages": True, "structured_timeout_ms": 2000}),
    ]
    first_capture_body = None
    for label, name, timeout_s, payload in profiles:
        meta, body = request_json(label, "POST", "/vision/capture-active-window", body=payload, timeout=timeout_s)
        status = meta.get("status")
        sources = capture_sources(body)
        acceptable_health_409 = (
            name == "Health probe"
            and status == 409
            and isinstance(body, dict)
            and body.get("error") == "active_window_unavailable"
            and body.get("reason") == "no_tree_snapshot"
        )
        compatible = acceptable_health_409 or bool(status and 200 <= status < 300 and sources)
        observed = capture_observed_metadata(body)
        if first_capture_body is None and status and 200 <= status < 300 and isinstance(body, dict):
            first_capture_body = body
            capture_metadata = observed
        record_result(
            "capture",
            f"POST /vision/capture-active-window: {name}",
            meta,
            body,
            compatible,
            {
                "max_seconds": timeout_s,
                "parseable_sources": sources,
                "acceptable_health_409": acceptable_health_409,
                "observed_metadata": observed,
            },
        )

    regression_meta, regression_body = request_json(
        "active_window_health_regression",
        "GET",
        "/vision/active-window/health",
        timeout=2.0,
    )
    record_result(
        "nonblocking_regression",
        "GET /vision/active-window/health",
        regression_meta,
        regression_body,
        bool(regression_meta.get("status") and 200 <= regression_meta["status"] < 300),
        {"blocking": False},
    )

    latest_meta, latest_body = request_json(
        "latest_active_window_regression",
        "GET",
        "/vision/latest-active-window",
        params={"max_age_ms": "10000"},
        timeout=3.0,
    )
    record_result(
        "nonblocking_regression",
        "GET /vision/latest-active-window",
        latest_meta,
        latest_body,
        bool(latest_meta.get("status") and 200 <= latest_meta["status"] < 300),
        {"blocking": False},
    )

    now = utc_now()
    start_hour = iso_ms(now - dt.timedelta(hours=1))
    start_10m = iso_ms(now - dt.timedelta(minutes=10))
    end_now = iso_ms(now + dt.timedelta(seconds=5))
    app_name = capture_metadata.get("app_name")
    window_name = capture_metadata.get("window_title")
    search_variants = [
        ("search_recent_context", "Recent context", {"q": marker, "content_type": "ocr", "limit": "40", "start_time": start_hour, "include_frames": "false"}),
        ("search_latest_incoming", "Latest incoming", {"q": marker, "content_type": "ocr", "limit": "10", "start_time": start_hour, "include_frames": "true", "focused": "true", "app_name": app_name or "", "window_name": window_name or ""}),
        ("search_latest_app_only", "Latest incoming fallback: app-only", {"q": marker, "content_type": "ocr", "limit": "10", "start_time": start_hour, "include_frames": "true", "app_name": app_name or ""}),
        ("search_latest_window_only", "Latest incoming fallback: window-only", {"q": marker, "content_type": "ocr", "limit": "10", "start_time": start_hour, "include_frames": "true", "window_name": window_name or ""}),
        ("search_latest_no_filter", "Latest incoming fallback: no-filter", {"q": marker, "content_type": "ocr", "limit": "10", "start_time": start_hour, "include_frames": "true"}),
        ("search_recent_assistant_screen_text", "Recent assistant screen text", {"q": marker, "content_type": "ocr", "limit": "40", "start_time": start_10m, "end_time": end_now, "include_frames": "false"}),
    ]
    for label, name, params in search_variants:
        params = {key: value for key, value in params.items() if value not in (None, "")}
        meta, body = request_json(label, "GET", "/search", params=params, timeout=10.0)
        rows, useful = useful_rows(body)
        compatible = bool(meta.get("status") and 200 <= meta["status"] < 300 and useful)
        record_result(
            "search",
            f"GET /search: {name}",
            meta,
            body,
            compatible,
            {
                "query_params": params,
                "row_count": len(rows),
                "useful_row_count": len(useful),
                "first_useful_row_metadata": useful[0]["metadata"] if useful else None,
                "first_useful_row_text_preview": useful[0]["text"][:160] if useful else None,
            },
        )

blocking = [r for r in results if r["kind"] not in ("nonblocking_regression",) and not r["compatible"]]
nonblocking = [r for r in results if r["kind"] == "nonblocking_regression" and not r["compatible"]]

report_lines = [
    "# Moya Screenpipe Endpoint Contract Report",
    "",
    f"- Base URL: `{base_url}`",
    f"- Data dir: `{data_dir}`",
    f"- Marker: `{marker}`",
    f"- Server log: `{log_file}`",
    f"- Verdict: `{'PASS' if not blocking else 'FAIL'}`",
    "",
    "## Blocking Checks",
    "",
]

for result in [r for r in results if r["kind"] != "nonblocking_regression"]:
    report_lines.extend([
        f"### {'PASS' if result['compatible'] else 'FAIL'} - {result['name']}",
        f"- HTTP status: `{result['status']}`",
        f"- Elapsed: `{result['elapsed_ms']} ms`",
        f"- Classification: `{result['classification']}`",
        f"- Response: `{result['response_file']}`",
        f"- Details: `{json.dumps(result['details'], sort_keys=True)}`",
        "",
    ])

report_lines.extend(["## Nonblocking Regression Checks", ""])
for result in [r for r in results if r["kind"] == "nonblocking_regression"]:
    report_lines.extend([
        f"### {'PASS' if result['compatible'] else 'WARN'} - {result['name']}",
        f"- HTTP status: `{result['status']}`",
        f"- Elapsed: `{result['elapsed_ms']} ms`",
        f"- Classification: `{result['classification']}`",
        f"- Response: `{result['response_file']}`",
        "",
    ])

if blocking:
    report_lines.extend(["## Failure Split", ""])
    buckets = {}
    for result in blocking:
        buckets.setdefault(result["classification"], []).append(result["name"])
    for classification, names in sorted(buckets.items()):
        report_lines.append(f"- `{classification}`: {', '.join(names)}")
    report_lines.append("")

(report_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
(report_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

print(report_dir / "report.md")
sys.exit(0 if not blocking else 1)
PY
