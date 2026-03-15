CREATE TABLE IF NOT EXISTS active_window_compat (
    frame_id INTEGER PRIMARY KEY,
    capture_provenance TEXT,
    focused_accessibility_json TEXT,
    main_body_text TEXT,
    main_body_meta_json TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(frame_id) REFERENCES frames(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_active_window_compat_provenance_frame_id
    ON active_window_compat(capture_provenance, frame_id DESC);
