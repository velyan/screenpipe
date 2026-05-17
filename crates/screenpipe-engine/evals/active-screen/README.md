# Active-Screen Extraction Eval

This fixture suite checks whether active-window extraction keeps the useful page
or editor content, removes obvious browser and app chrome, and preserves focused
element context.

Run it from the repository root:

```bash
cargo run -p screenpipe-engine --bin screenpipe-eval-active-screen
```

Use JSON output for automation:

```bash
cargo run -p screenpipe-engine --bin screenpipe-eval-active-screen -- --json
```

Fixtures are intentionally plain JSON so private local captures can be added
without changing the runner. Each fixture can assert:

- `must_include`: text that must survive into `main_body_text`
- `must_exclude`: noisy UI text that should be removed
- `focus_must_include`: text expected in focused element metadata
- `requires_focus_bounds`: whether focused element bounds must be present
- `min_score`: fixture-specific pass threshold
