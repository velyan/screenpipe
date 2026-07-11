-- screenpipe — AI that knows everything you've seen, said, or heard
-- https://screenpipe.com
-- Capture action-fidelity upgrades (trajectory/eval data quality):
--
-- 1. ui_events.element_ancestors — the clicked element's path in the window
--    hierarchy, root-first compact JSON:
--      [{"role":"AXWindow","name":"Inbox"},{"role":"AXGroup"},...]
--    Captured budgeted at click time by the a11y ctx-capture worker (macOS).
--    Disambiguates identical labels in different containers ("the Profile
--    button in the sidebar" vs "in the modal") — the #1 gap when replaying
--    recorded workflows step-by-step.
--
-- 2. display_layout — snapshots of the full display arrangement (per display:
--    id, stable_id, name, global-desktop origin x/y in points, width/height
--    as reported by the capture backend, is_primary). Written at capture
--    start and whenever the arrangement changes. Without this, click
--    coordinates (global desktop points) cannot be reliably matched to
--    element-tree bounds (normalized per-monitor) on multi-monitor setups —
--    measured at ~25% join accuracy on historical data for exactly this
--    reason.

ALTER TABLE ui_events ADD COLUMN element_ancestors TEXT;

CREATE TABLE IF NOT EXISTS display_layout (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    -- why this row exists: 'startup' | 'change'
    reason TEXT NOT NULL DEFAULT 'change',
    -- canonical JSON array, displays sorted by id:
    -- [{"id":1,"stable_id":"...","name":"Built-in","x":0,"y":0,
    --   "width":1512,"height":982,"is_primary":true}, ...]
    layout_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_display_layout_timestamp
    ON display_layout(timestamp);
