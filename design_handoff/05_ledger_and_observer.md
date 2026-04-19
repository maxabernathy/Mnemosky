# 05 · Ledger Ribbon & Observer Card

These are the two cross-cutting surfaces. They appear in every mode (with minor variations) and are what make the Observatory feel like a single instrument rather than four screens.

---

## The Translation Ledger Ribbon

### Purpose

A live, always-visible ribbon of the seven-category rejection taxonomy used by `satellite_trail_detector.py`. It makes the detector's "reasons for saying no" an ambient material, not a hidden appendix.

### Categories (fixed, in this order)

1. `TOO SHORT` — tracklet length below threshold
2. `STAR-LIKE` — eccentricity below threshold
3. `LOW SNR` — signal-to-noise below threshold
4. `ECCENTRIC` — angle/curvature out of range
5. `BRIGHT CLUSTER` — contaminated by bright neighbours
6. `DUPE` — duplicate of a prior tracklet
7. `CONFIRMED` — passes all; this is the positive category

Names are final. Do not alphabetise, reorder, or rename. The order reflects the order rejections happen in the pipeline, with `CONFIRMED` as the terminal positive.

### Position

```css
position: absolute;
left: 56 px; right: 64 px; bottom: 40 px;
```

Inside the canvas, pinned to the bottom edge of the guide frame. Absolute-positioned over the frame contents.

### Layout

```css
display: grid;
grid-template-columns: repeat(7, 1fr);
gap: 18 px;
```

Seven equal cells, 18 px gap. Ribbon has no surrounding background — each cell carries its own.

### Cell

```
┌─────────────────────┐
│  128                │   ← numeric.cell (italic serif 40 px)
│  TOO SHORT          │   ← mono.meta 18 px, tracking 0.22 em
└─────────────────────┘
```

- `border: 1 px solid color.surface.line`
- `background: rgba(15, 14, 11, 0.55)`
- `backdrop-filter: blur(6 px)` (or equivalent on platform — Qt: use a translucent `QFrame` with a `QGraphicsBlurEffect` on the canvas behind, not on the cell itself)
- `border-radius: 10 px`
- `padding: 16 px 18 px`

### Cell states

| State | Border | Ink | Numeral |
|---|---|---|---|
| default | `color.surface.line` | `color.ink.secondary` | `color.ink.primary` |
| hover / tap | `rgba(159, 231, 194, 0.3)` | `color.ink.primary` | `color.ink.primary` |
| **active / filtering** | `rgba(159, 231, 194, 0.5)` | `color.accent.mint` | `color.accent.mint` |

Active state uses `motion.settle` to transition. Only one cell can be active at a time. Tapping the active cell clears the filter.

### Behaviour per mode

| Mode | Tap a cell |
|---|---|
| **Observe** | Filters the canvas to show only detections matching that rejection category (grays out the rest). Tag badges on bboxes update accordingly. |
| **Tune** | Ribbon is **hidden**. Tuning happens before detections. |
| **Review** | Queues "next N rejections of this type" — doesn't filter the current view. Label above the ribbon changes to `JUMP TO CATEGORY`. |
| **Inspect** | Numbers are scoped to neighbourhood (same tracklet, frame ±10). Label above the ribbon changes to `NEIGHBOURHOOD · FRAME ±10`. Tapping is non-functional (no filter in Inspect). |

### Counts

All counts are **live**, updating as the detector emits events. They represent the **current session**, reset on detector start.

`CONFIRMED` counts separately — it's the one positive category. Keep it styled like the others.

### Data binding

Each cell is bound to `session.ledger[category]`:

```ts
type LedgerCategory = 'too-short' | 'star-like' | 'low-snr' | 'eccentric' |
                     'bright-cluster' | 'dupe' | 'confirmed';
type LedgerState = { [K in LedgerCategory]: number };
```

Updates at 2–5 Hz is sufficient; don't push on every detector event if the backend emits at 100+ Hz.

### Accessibility

At gallery scale, cells are 500+ px wide — taps from 2–4 m distance are comfortable with capacitive touch. At desktop, cells are ~180 px — still clickable. Do not add tooltips; the cell label *is* the label.

---

## The Observer Card

### Purpose

Haraway's situated-knowledge made a literal object. Every observation has an observer; that observer's situation — where they stand, under what sky, with what instrument — is part of the data, not ambient context.

### Position

In the topbar, between the mode name and the local-time slot. Flex `1` to fill the remaining width.

### Content

A single-line sentence in `mono.meta` 20–24 px, `color.ink.secondary`, uppercase, tracking 0.28 em. Sections separated by `·` with spaces either side:

```
OBSERVER · 40.71°N · -74.01°W · BORTLE 5 · FOV 22° · "CLEAR NIGHT, HIGH CIRRUS" · ALGORITHM: DEFAULT + MATCHED-FILTER
```

### Fields (all required; empty fields collapse)

| Field | Source | Format | Notes |
|---|---|---|---|
| `OBSERVER` | profile | fixed label | Always present |
| lat/long | profile | `DD.DD°N/S` / `DD.DD°W/E` | Two decimal places |
| Bortle class | profile | `BORTLE 1`–`BORTLE 9` | Omit if unknown |
| FOV | profile | `FOV 22°` | Omit if unknown |
| notes | free-text | `"…"` with quotes | Lowercase inside the quotes; sentence casing; ≤80 chars |
| algorithm | runtime | `ALGORITHM: X + Y` | Concatenate active algorithms with `+` |

The **free-text notes** stay lowercase inside the quotes (the surrounding mono frame is uppercase, but the quoted text is literal — preserve user casing). Example:

```
… · "clear night, high cirrus" · …
```

Note the lowercase content in quotes vs. the uppercase surrounding metadata. This is deliberate.

### Dilate state (for the attract loop)

In the attract loop beat "Observer's card" (see `07_attract_loop.md`), the card dilates to **fill the canvas** for six seconds:

- Background: the grass-bokeh stage (same recipe as the title screen)
- Type: `body.italic` 60 px in `color.surface.night` (dark ink on the bright bokeh)
- Same sentence content, just scaled up

Then it collapses back into the topbar on `motion.settle`.

### Desktop port

In the OpenCV shell port, the Observer Card is a **single line above the canvas**, same sentence, ~14 px mono. Same content grammar, different size.

### Editing

Not in scope for this handoff. The observer profile is configured once per session (either via a config file or an existing UI path) and displayed read-only in the shell.

---

## Why they're in the same document

Because they're the same idea in two places:

- The **Ledger ribbon** surfaces the *instrument's* epistemology (what it rejects and why).
- The **Observer card** surfaces the *observer's* epistemology (who is looking, from where).

Together they make Mnemosky's claim — that seeing is situated and rejection is a category of knowledge — legible at a glance.
