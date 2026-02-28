-- Add distilled main-body text fields to frames and a persistent cache table.
-- Also rebuild frames_fts to index main_body_text for distilled-first search.

ALTER TABLE frames ADD COLUMN main_body_text TEXT DEFAULT NULL;
ALTER TABLE frames ADD COLUMN main_body_meta_json TEXT DEFAULT NULL;

CREATE TABLE IF NOT EXISTS text_distillation_cache (
    cache_key TEXT PRIMARY KEY,
    main_body_text TEXT NOT NULL,
    excluded_ui_text_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL,
    provider TEXT NOT NULL,
    prompt_version INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TRIGGER IF EXISTS frames_ai;
DROP TRIGGER IF EXISTS frames_au;
DROP TRIGGER IF EXISTS frames_ad;
DROP TABLE IF EXISTS frames_fts;

CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
    name,
    browser_url,
    app_name,
    window_name,
    focused,
    accessibility_text,
    main_body_text,
    id UNINDEXED,
    tokenize='unicode61'
);

INSERT INTO frames_fts(
    id,
    name,
    browser_url,
    app_name,
    window_name,
    focused,
    accessibility_text,
    main_body_text
)
SELECT
    id,
    COALESCE(name, ''),
    COALESCE(browser_url, ''),
    COALESCE(app_name, ''),
    COALESCE(window_name, ''),
    COALESCE(focused, 0),
    COALESCE(accessibility_text, ''),
    COALESCE(main_body_text, '')
FROM frames;

CREATE TRIGGER IF NOT EXISTS frames_ai AFTER INSERT ON frames BEGIN
    INSERT INTO frames_fts(
        id,
        name,
        browser_url,
        app_name,
        window_name,
        focused,
        accessibility_text,
        main_body_text
    )
    VALUES (
        NEW.id,
        COALESCE(NEW.name, ''),
        COALESCE(NEW.browser_url, ''),
        COALESCE(NEW.app_name, ''),
        COALESCE(NEW.window_name, ''),
        COALESCE(NEW.focused, 0),
        COALESCE(NEW.accessibility_text, ''),
        COALESCE(NEW.main_body_text, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS frames_au AFTER UPDATE ON frames
WHEN (NEW.name IS NOT NULL AND NEW.name != '')
   OR (NEW.browser_url IS NOT NULL AND NEW.browser_url != '')
   OR (NEW.app_name IS NOT NULL AND NEW.app_name != '')
   OR (NEW.window_name IS NOT NULL AND NEW.window_name != '')
   OR (NEW.focused IS NOT NULL)
   OR (NEW.accessibility_text IS NOT NULL AND NEW.accessibility_text != '')
   OR (NEW.main_body_text IS NOT NULL AND NEW.main_body_text != '')
BEGIN
    INSERT OR REPLACE INTO frames_fts(
        id,
        name,
        browser_url,
        app_name,
        window_name,
        focused,
        accessibility_text,
        main_body_text
    )
    VALUES (
        NEW.id,
        COALESCE(NEW.name, ''),
        COALESCE(NEW.browser_url, ''),
        COALESCE(NEW.app_name, ''),
        COALESCE(NEW.window_name, ''),
        COALESCE(NEW.focused, 0),
        COALESCE(NEW.accessibility_text, ''),
        COALESCE(NEW.main_body_text, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS frames_ad AFTER DELETE ON frames BEGIN
    DELETE FROM frames_fts WHERE id = OLD.id;
END;
