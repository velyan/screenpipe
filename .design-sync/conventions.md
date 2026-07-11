# Building with the screenpipe design system

screenpipe's UI is **Black & White Geometric Minimalism**: monospace type, pure
grayscale (no color), sharp corners. Every component renders that brand
automatically — your job is to compose them and lay them out in the same idiom.

## Setup & wrapping

- **No theme provider needed.** All design tokens are CSS variables on `:root` in
  `styles.css` (imported transitively), so components pick up the brand on their
  own. A `.dark` class on an ancestor switches to the dark palette.
- **Two components need a provider wrapper — omit it and they throw:**
  - `Tooltip` must sit under a `<TooltipProvider>` (wrap once, near the root).
    `HelpTooltip` wraps its own provider — use it directly.
  - `Toast` needs a `<ToastProvider>` around it and a `<ToastViewport>` sibling.
- Components load from the bundle global `window.ScreenpipeUI.*`. Compound parts
  are individual exports — `Card` + `CardHeader`/`CardTitle`/`CardContent`/`CardFooter`,
  `Select` + `SelectTrigger`/`SelectContent`/`SelectItem`, `Dialog` +
  `DialogContent`/`DialogHeader`/`DialogTitle`/`DialogFooter`, etc. Each component's
  `.prompt.md` shows the exact composition.

## The styling idiom — Tailwind utilities + brand tokens

Lay out YOUR markup with normal Tailwind utility classes (`flex`, `grid`,
`grid-cols-3`, `gap-4`, `p-6`, `mt-4`, `space-y-4`, `w-80`, `max-w-md`, `text-sm`,
`font-medium`, `items-center`, `justify-between` … all compiled and available).
For **color**, ALWAYS use these token-backed classes — never raw hex or palette
colors like `bg-blue-500` — so output stays on-brand and theme-aware:

| Role | Classes |
|---|---|
| Surfaces | `bg-background` `bg-card` `bg-popover` `bg-muted` `bg-secondary` `bg-accent` |
| Text | `text-foreground` `text-muted-foreground` `text-primary` `text-secondary-foreground` |
| Primary (solid black) | `bg-primary` + `text-primary-foreground` |
| Destructive | `bg-destructive` + `text-destructive-foreground` — renders **black, not red** (grayscale brand) |
| Borders / inputs | `border` `border-border` `border-input` `bg-input` · focus ring uses `var(--ring)` |

Same tokens exist as CSS variables for custom CSS: `var(--background)`,
`var(--foreground)`, `var(--primary)`, `var(--muted)`, `var(--border)`, …

Two brand rules are baked into the tokens — don't fight them:
- **Sharp corners.** `--radius` is `0`; `rounded-*` classes have no visual effect.
  Keep edges square.
- **Monospace everywhere.** The page font is `JetBrains Mono` (set on `html`). Don't
  introduce a sans-serif font.

## Where the truth lives

- `styles.css` `@import`s `_ds_bundle.css` (compiled tokens + every utility class)
  and `fonts/fonts.css` (JetBrains Mono). Read it for the exact token set.
- Per component: `components/<group>/<Name>/<Name>.prompt.md` (usage + examples)
  and `<Name>.d.ts` (props). Groups: `forms`, `overlays`, `feedback`, `display`.

## Idiomatic snippet

```tsx
const { Card, CardHeader, CardTitle, CardDescription, CardContent, Button, Badge } = window.ScreenpipeUI;

<Card className="w-80">
  <CardHeader className="flex flex-row items-center justify-between">
    <div>
      <CardTitle>Meeting notes</CardTitle>
      <CardDescription className="text-muted-foreground">Weekly sync · 42 min</CardDescription>
    </div>
    <Badge variant="secondary">Transcribed</Badge>
  </CardHeader>
  <CardContent className="text-sm">
    <p>12 action items extracted across 4 speakers.</p>
    <Button className="mt-4">Open timeline</Button>
  </CardContent>
</Card>
```
