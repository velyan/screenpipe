# design-sync notes — screenpipe

screenpipe's design system is shadcn/ui components in `apps/screenpipe-app-tauri/components/ui/`
(Radix + class-variance-authority + Tailwind, CSS variables). It is **not** a published
package — there is no `dist/`. The converter runs in **synth/`--entry` mode** against a
generated barrel.

## Build wiring (how this repo syncs)

- **PKG_DIR** must resolve to `apps/screenpipe-app-tauri`. Achieved by passing
  `--entry ./apps/screenpipe-app-tauri/.ds-build/entry.tsx`; the package.json walk-up from
  that dir lands on the app's `package.json` (name `screenpipe`).
- **`--node-modules apps/screenpipe-app-tauri/node_modules`** (has react/react-dom/radix/cva/lucide).
- **Barrel** `apps/screenpipe-app-tauri/.ds-build/entry.tsx` = `export * from "@/components/ui/<file>"`
  for every ui file. Regenerate it if components are added/removed (one line per file).
- **`cfg.tsconfig` = `.ds-build-tsconfig.json`** (app-relative). It mirrors the app's `@/*`
  aliases AND routes three imports to shims so Tauri/app-only deps don't enter the bundle:
  - `@/lib/utils` → `.ds-build/shims/utils.ts` (only `cn` is used by ui components)
  - `@/lib/utils/validation` → `.ds-build/shims/validation.ts` (`debounce` + type; used by Validated*)
  - `@tauri-apps/api/app` → `.ds-build/shims/tauri-app.ts` (`hide` no-op; imported by Dialog)
  If a ui component starts importing something new from `@/lib/...` that drags in Tauri/app
  state, add a shim + a paths entry here.
- **CSS** = `cfg.cssEntry = .ds-build/compiled.css`, produced by compiling Tailwind:
  `cd apps/screenpipe-app-tauri && npx tailwindcss -c .ds-build/tailwind.config.cjs -i app/globals.css -o .ds-build/compiled.css`
  `.ds-build/tailwind.config.cjs` re-roots the app's content globs to absolute paths AND adds
  `.design-sync/previews/**` so authored-preview utility classes are compiled in.
  **Recompile this before the final build** (and any time previews add new utility classes).
  `.ds-build/tw-base.cjs` is a copy of `tailwind.config.ts` (which is already CJS).
- Everything under `apps/screenpipe-app-tauri/.ds-build/` and `.ds-build-tsconfig.json` is
  gitignored build scaffolding (regenerated, not committed).

## Fonts

- Brand font is **JetBrains Mono** (Tailwind `fontFamily.sans` + `.mono`); fallbacks are
  SF Mono / IBM Plex Mono / ui-monospace. It is NOT shipped in the repo, so the app renders in
  system mono. For the DS we **vendor JetBrains Mono** (OFL) weights 400/500/600/700 from
  `@fontsource/jetbrains-mono` into `.ds-build/fonts/`, wired via `cfg.extraFonts`.
- `cfg.runtimeFontPrefixes` suppresses `[FONT_MISSING]` for SF Mono / IBM Plex Mono (intentional
  system fallbacks) and Cambria / Georgia (serif fallbacks introduced by `@tailwindcss/typography`).

## Contracts

- No built `.d.ts`, so `propsBodyFor` finds nothing — **all 29 contracts are hand-written in
  `cfg.dtsPropsFor`** (clean enums from each `cva` + the key Radix primitive props). When a
  component's variants/props change upstream, update `dtsPropsFor` to match.

## Grouping

- All components live in `components/ui/` (a generic dir), so src-dir grouping yields `general`.
  Groups are set via **frontmatter-only category stubs** in `.design-sync/docs/<Name>.md`
  (`cfg.docsDir`). An empty doc body is falsy, so the prompt stays auto-synthesized (Props +
  Examples + Related) while `category:` sets the group. Groups: forms / overlays / feedback / display.

## Known render warns (triage list — a warn NOT here is new)

- `[FONT_MISSING] Cambria/Georgia` is suppressed via runtimeFontPrefixes; if it appears for a
  NEW family, hunt it.
- Floor-card / blank `[RENDER_BLANK]`/`[RENDER_THIN]` only legitimately appears for components
  with no authored preview — all 29 are authored, so any such warn is real.

## Component-specific authoring notes (from wave learnings)

- **Grayscale brand**: `destructive` variants render solid black (Button/Badge/Alert) — faithful,
  not a bug. Do not "fix" to red.
- **Overlays** (Dialog, AlertDialog, Popover, DropdownMenu, Tooltip, Toast, ContextMenu) are
  rendered open via `defaultOpen`/`open`/`forceMount` and pinned with `cfg.overrides.<Name>`
  (`cardMode: single` + a `viewport`). Tooltip needs a `TooltipProvider`; Toast needs
  `ToastProvider` + `ToastViewport` (forced `position: static` so it renders in the card, not a
  screen corner). ContextMenu has no controlled-open prop (right-click only) — its preview
  dispatches a real `contextmenu` MouseEvent on the trigger in a `useEffect` on mount so the
  menu opens for the static capture (forceMount alone renders it invisibly at opacity 0).
- **Command** renders inline (cmdk) — no override needed.
- **MultiSelect** renders only its CLOSED trigger in static capture (popover can't open).
- **Validated\*** : `required` is the reliable variant axis; debounced validation never fires
  in static capture.
- **Calendar** : use a FIXED date for determinism; `mode="range"` is a strong variant vs `single`.
- **CodeBlock** : ships its own dark theme (coldarkDark) — that's the real in-app look, not a
  page-theme bug.
- **HelpTooltip** : hover-only; static capture shows just the (?) icon in context — acceptable.

## Re-sync risks (what can silently go stale)

- **`.ds-build/compiled.css` is generated** — recompile Tailwind before the final build or the
  DS ships stale/missing utility classes.
- **`dtsPropsFor` is hand-written** — it does not track upstream prop changes automatically.
  Re-check against `components/ui/*.tsx` `cva` blocks on a re-sync.
- **The barrel `.ds-build/entry.tsx`** must be regenerated if `components/ui/` files are
  added/removed (otherwise new components are missing or removed ones error the bundle).
- **JetBrains Mono** is vendored from `@fontsource/jetbrains-mono` (installed in `.ds-sync/`,
  gitignored) — on a fresh clone, re-copy the woff2 (re-install the fontsource pkg) before build.
- **Bundle is ~2.4 MB** (react-syntax-highlighter via CodeBlock + react-day-picker via Calendar
  dominate). Acceptable; trim by excluding those components if size becomes a concern.
- **`.design-sync/config.json` is not committed to git** (screenpipe is public; per standing user
  preference, sync config/notes/previews are kept local rather than pushed). This means the
  pinned `projectId` doesn't survive a fresh clone or a different machine — check `get_project`
  on it before trusting it (see 2026-07-01 entry below for what happens when it doesn't).
- **2026-07-01**: the pinned project (`831720e3-…`) 404'd — deleted or otherwise gone. Recreated
  as "Screenpipe Design System" (`b168ac78-0ab7-402c-86ac-6554d70e2193`) and re-uploaded the
  already-built, already-validated `ds-bundle/` (30 components, 0 bad, 0 unmerged learnings) with
  no rebuild needed since no `components/ui/*.tsx` source had changed since the last build. If
  the project vanishes again, check `list_projects` first — an existing "Design System" project
  (unrelated name, `5c84ea65-…`) also exists on this account; don't confuse the two.

## Dock (app group) — floating-overlay baseline

`Dock` is a **presentational reproduction** of the live floating dock
(`app/shortcut-reminder/page.tsx` + `audio-equalizer.tsx` + `screen-matrix.tsx`),
added as the baseline to iterate on. It is NOT the app component — it's decoupled
from Tauri/WebSocket/store and driven entirely by props (DockProps).

- Source: `apps/screenpipe-app-tauri/.ds-build/app/dock.tsx` (gitignored, local).
  Re-exported via the barrel (`export * from "./app/dock"`); pinned in
  `componentSrcMap` at `.ds-build/app/dock.tsx`; contract in `dtsPropsFor.Dock`.
- **Group derivation gotcha**: the file lives under `.ds-build/app/` specifically so
  the src-dir group derives to `app`. If it were under `.ds-build/components/` it
  would derive to `ds-build` and the `docs/Dock.md` `category: App` would be IGNORED
  (doc category only overrides general/misc groups). Keep it under an `app/` segment.
- Distinct visual language from the B&W primitives: **dark glass** (`rgba(0,0,0,0.88)`,
  white/25 border, 1px white internal dividers), monospace shortcut chips, two canvas
  visualizers (AudioEqualizer = 8 speech-driven bars; ScreenMatrix = CRT sweep + scan
  lines), Phone meeting toggle (pulsing dot when active), X close. `cardMode: column`.
- To re-sync after upstream dock changes: re-spec `app/shortcut-reminder/page.tsx` and
  update `.ds-build/app/dock.tsx` to match (it's a hand reproduction, not auto-derived).

### Dock — collapsed (non-hover) state

The dock's collapse/expand is **native** (`src-tauri/swift/shortcut_reminder.swift`),
not in the React `app/shortcut-reminder/page.tsx` (which only renders the expanded
row). The `Dock` component now reproduces BOTH from that Swift source:
- `collapsed={false}` (default): expanded single-row (user-specified layout).
- `collapsed`: the non-hover capsule — `Capsule` `black 0.75` + white/15 stroke,
  `[app icon 12×scale] · [equalizer 18×scale + screen-matrix 18×scale] · [phone]`.
  Spec from `collapsedView`/`CollapsedAppIconButton`/`CollapsedPhoneButton` (kBaseCollapsedW/H = 62/22).
- The **app icon** is the real screenpipe icon (`src-tauri/icons/32x32.png`) base64-inlined
  at `.ds-build/app/icon.ts` (imported by dock.tsx; overridable via the `appIconSrc` prop).
  Re-encode that file if the app icon changes.
