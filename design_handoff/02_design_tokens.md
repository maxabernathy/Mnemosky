# 02 · Design Tokens

Every value here appears in `tokens.json` in machine-readable form. Prefer importing that over transcribing.

All sizes are quoted for the **gallery · 8K** build (3840 × 2160). Desktop and 4K scales are derived in `tokens.json > scales`.

---

## Colour

### Surface (dark theme — all interior modes)

| Token | Hex | Role |
|---|---|---|
| `color.surface.night` | `#0F0E0B` | Outermost shell background |
| `color.surface.night2` | `#15130F` | Panel / canvas background |
| `color.surface.panel` | `#1A1713` | Cards, blocks, side panels |
| `color.surface.panel2` | `#211E18` | Elevated elements inside a panel |
| `color.surface.line` | `#2B2821` | Dividers, borders, frame outlines |

### Ink (foreground on dark)

| Token | Hex | Role |
|---|---|---|
| `color.ink.primary` | `#EDEEDC` | Primary text; wordmarks on dark |
| `color.ink.secondary` | `#AAB097` | Metadata, mono labels, field labels |
| `color.ink.tertiary` | `#6F7564` | Timers, counters, least-salient copy |

### Accent

| Token | Hex | Role |
|---|---|---|
| `color.accent.mint` | `#9FE7C2` | Live state, active mode, "confirmed" |
| `color.accent.mercury` | `#C9D3BA` | Mercury-blob loader base; secondary accent |
| `color.accent.grass` | `#7C9270` | Grass-bokeh background base (title, splash) |
| `color.accent.grassSoft` | `#AEC099` | Grass-bokeh highlight layer |

### Semantic (detections)

| Token | Hex | Role |
|---|---|---|
| `color.semantic.satellite` | `#D7B877` | Satellite bbox + tag |
| `color.semantic.airplane` | `#D98E5A` | Airplane bbox + tag |
| `color.semantic.anomalous` | `#C98FB6` | Anomalous / unclassified bbox + tag |

### Ledger category colours (same 7 across every screen)

Use `color.ink.secondary` for inactive, `color.accent.mint` for the active/filtered category. Do **not** colour-code the seven categories differently — they share one visual tier and rely on layout order for identity.

---

## Typography

### Families

| Token | Stack |
|---|---|
| `font.serif` | `"Times New Roman", Times, Tinos, "Liberation Serif", serif` |
| `font.serifItalic` | same stack, `font-style: italic` |
| `font.mono` | `"JetBrains Mono", "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace` |

If Times New Roman is unavailable on the target platform, acceptable substitutes in order of preference: **Playfair Display Italic**, **Cormorant Garamond Italic**, **Source Serif 4 Italic**. Do not substitute a modern sans-serif (Inter, Roboto, etc.).

### Roles (gallery sizes)

| Role | Family | Size | Leading | Tracking | Case | Notes |
|---|---|---|---|---|---|---|
| `display.wordmark` | serifItalic | **240 px** | 0.95 | 0.005 em | lowercase | Hero only. One per screen. |
| `display.hero` | serifItalic | **140 px** | 1.0 | 0 | sentence | Section H2 |
| `display.mode` | serifItalic | **120 px** | 1.0 | 0 | sentence | Mode names in headers |
| `display.modeCard` | serifItalic | **72 px** | 1.0 | 0 | Title | The four-mode strip |
| `body.italic` | serifItalic | **60 px** | 1.28 | 0 | sentence | Manifesto, long-form |
| `body.italicSmall` | serifItalic | **46 px** | 1.05 | 0 | sentence | Pillar/card h4 |
| `body.text` | serif | **24 px** | 1.45 | 0 | sentence | Card paragraphs |
| `numeric.hero` | serifItalic | **96 px** | 1.0 | 0 | tabular | Big stats; `font-variant-numeric: tabular-nums` |
| `numeric.cell` | serifItalic | **40 px** | 1.0 | 0 | tabular | Ledger cell values |
| `mono.meta` | mono | **22–28 px** | 1.0 | **0.24–0.30 em** | **UPPERCASE** | All metadata. Mandatory uppercase + tracking. |
| `mono.label` | mono | **18–20 px** | 1.0 | **0.26–0.30 em** | **UPPERCASE** | Field labels, section eyebrows |
| `mono.micro` | mono | **14–16 px** | 1.0 | 0.20 em | UPPERCASE | Keyboard keys, unit suffixes |

**Rule:** mono is **always uppercase with generous tracking**. Never lowercase mono. Never italic mono. Never tracked serif (tracking is 0 for serif roles).

### The number-unit pattern

Always `{italic-serif numeral}` followed by a space and `{uppercase mono unit}`:

```
17 CONFIRMED       4,812 FRAMES       128 FPS
```

Never reverse, never merge. The two families stay side-by-side with air between them.

---

## Spacing

Use a 4-px base scale at desktop, a 6-px scale at 4K, and an **8-px scale at gallery**:

| Token | Gallery (8K) | 4K | Desktop |
|---|---|---|---|
| `space.1` | 8 | 6 | 4 |
| `space.2` | 16 | 12 | 8 |
| `space.3` | 24 | 18 | 12 |
| `space.4` | 32 | 24 | 16 |
| `space.5` | 48 | 36 | 24 |
| `space.6` | 64 | 48 | 32 |
| `space.7` | 80 | 60 | 40 |
| `space.8` | 120 | 90 | 60 |
| `space.9` | 180 | 135 | 90 |
| `space.10` | 240 | 180 | 120 |

Screen padding at gallery is `space.9` (180 px) or `space.10` (240 px) depending on whether the section carries a large hero.

---

## Radius

| Token | px (gallery) | Use |
|---|---|---|
| `radius.sm` | 8 | Keyboard keys, tiny pills |
| `radius.md` | 10 | Dashed canvas guide frame |
| `radius.lg` | 16 | Cards, panels, the app shell itself |
| `radius.xl` | 18 | Mode rail icons, mode cards, Ledger cells, swatches |

Never fully rounded. Never 0.

---

## Borders

- `1 px` `color.surface.line` — default divider inside panels
- `2 px` `color.surface.line` — section header rule
- `2 px dashed #1F1D17` — canvas guide frame (decorative)
- `2 px solid {semantic.*}` + `0 0 0 1 px rgba(*, 0.18) + 0 0 42 px rgba(*, 0.22)` — detection bbox glow. This is one styling unit; don't simplify the box-shadow.

---

## Motion

| Token | Duration | Easing | Use |
|---|---|---|---|
| `motion.instant` | 120 ms | `cubic-bezier(0.2, 0, 0, 1)` | Mode switch sidebar reflow, keyboard feedback |
| `motion.settle` | 260 ms | `cubic-bezier(0.2, 0.8, 0.2, 1)` | Ledger cell filter, Observer card dilate |
| `motion.meditative` | 7 000 ms | linear | Splash loader cycle, attract-loop beats |
| `motion.blob` | — | — | Mercury loaders; see `06_mercury_loaders.md` for the goo-filter spec |

Motion rules:

1. No loading spinners. Ever. Mercury blobs only.
2. No easing sharper than `cubic-bezier(0.2, 0, 0, 1)`.
3. No animation longer than 400 ms in interactive contexts (mode switches, filters, selections).
4. The attract loop is the **only** place where second-scale motion is allowed.

---

## Elevation (box-shadow)

We use elevation sparingly; the dark surfaces carry hierarchy mostly through line and inset. The one exception is the Ledger ribbon:

```css
/* Ledger ribbon over canvas */
background: rgba(15, 14, 11, 0.55);
backdrop-filter: blur(6 px);
border: 1 px solid color.surface.line;
```

And the active mode rail icon:

```css
background: color.accent.mint;
color: color.surface.night;
box-shadow: 0 0 0 6 px rgba(159, 231, 194, 0.12);
```

No card drop-shadows. No glows except the detection bboxes.

---

## The grass-bokeh background

Used on **title screen** and **splash only** — not inside any mode. It's a **layered radial gradient**, not an image. Recipe:

```css
background:
  radial-gradient(2400 px 1600 px at 18% 22%, #E9F0DF 0%, transparent 55%),
  radial-gradient(1600 px 1400 px at 86% 10%, #CFDCBF 0%, transparent 60%),
  radial-gradient(2000 px 1800 px at 70% 88%, #96AA83 0%, transparent 60%),
  radial-gradient(1400 px 1100 px at  8% 92%, #B7C5A1 0%, transparent 55%),
  linear-gradient(180 deg, #AEC099 0%, #7C9270 100%);
```

An overlay layer of blurred white bokeh:

```css
background:
  radial-gradient(320 px at 12% 28%, rgba(255,255,255,.55), transparent 70%),
  radial-gradient(260 px at 22% 80%, rgba(255,255,255,.42), transparent 70%),
  radial-gradient(420 px at 82% 22%, rgba(255,255,255,.36), transparent 70%),
  radial-gradient(320 px at 90% 72%, rgba(255,255,255,.38), transparent 70%);
filter: blur(40 px);
```

Scale these px values proportionally for smaller shells.

---

## Accessibility checks

All ink/surface pairs used clear WCAG AA at 22 px+ (the minimum we use mono at, even at desktop size). The two at-risk pairs:

- `ink.tertiary` (`#6F7564`) on `night` (`#0F0E0B`) — AA at 22 px+ only. Do not use below 18 px.
- `accent.mercury` (`#C9D3BA`) on `night2` (`#15130F`) — AA large only. Use for loaders, not body text.

Semantic colours on `night2`:

- `satellite` `#D7B877` — AA at 22 px+
- `airplane` `#D98E5A` — AAA at 22 px+
- `anomalous` `#C98FB6` — AA at 22 px+

All pass at gallery scale. Recheck if you take this below desktop size.
