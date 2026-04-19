# 03 · The Shell

The shell is the persistent chrome around every mode. It is **the same five regions** for Tune, Observe, Review, and Inspect. Only the canvas contents and the sidebar blocks change.

## Layout — 8K gallery

```
┌────────────────────────────────────────────────────────────────────────────┐
│  TOPBAR · 120 px                                                            │ ← Observer card lives here
├──────┬────────────────────────────────────────────────────┬─────────────────┤
│ RAIL │                                                    │                 │
│ 180  │                  CANVAS (flex 1)                   │ SIDEBAR · 640   │
│  px  │   ┌──────────────────────────────────────┐         │                 │
│      │   │ dashed guide frame @ 70 px inset     │         │                 │
│      │   │                                      │         │                 │
│      │   │     mode content                     │         │                 │
│      │   │                                      │         │                 │
│      │   │   ┌─────── LEDGER RIBBON ─────────┐  │         │                 │
│      │   │   │ 7 cells · always visible      │  │         │                 │
│      │   │   └───────────────────────────────┘  │         │                 │
│      │   └──────────────────────────────────────┘         │                 │
├──────┴────────────────────────────────────────────────────┴─────────────────┤
│  STATUS · 80 px                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

| Region | Dimension (gallery) | Persistent? | Purpose |
|---|---|---|---|
| Topbar | full width × 120 px | always | Wordmark, current mode, Observer card, local time |
| Rail | 180 × full height | always | Four mode icons (`T O R I`), vertical stack |
| Canvas | flex × flex | always | Mode-specific content. Hosts the dashed guide frame, the mode output, and the Ledger ribbon |
| Sidebar | 640 × full height | always | Mode-specific blocks (detection details, keys, stats…) |
| Status | full width × 80 px | always | Live system state: frame counter, throughput, output file |

### Grid (CSS / Qt equivalent)

```css
display: grid;
grid-template-columns: 180px 1fr 640px;
grid-template-rows: 120px 1fr 80px;
```

Topbar and status span all three columns. Rail + canvas + sidebar occupy the middle row.

## Topbar

Height 120 px, padding 0 56 px. Flex row, centre-aligned, gap 48 px.

| Slot | Style | Content |
|---|---|---|
| Brand | `display.mode` (italic serif 48 px) | `mnemosky` — lowercase, always |
| Mode name | `mono.meta` in `color.accent.mint`, tracking 0.32 em | `OBSERVE` / `TUNE` / `REVIEW` / `INSPECT` |
| Observer card | `mono.meta` 20 px in `color.ink.secondary`, flex:1 | See `05_ledger_and_observer.md` |
| Local time | `mono.meta` 20 px in `color.ink.tertiary`, right-aligned | `T · 22H04M38S · LOCAL` |

Topbar background: `linear-gradient(180deg, rgba(255,255,255,0.03), transparent)` over `color.surface.night`.

## Rail

180 px wide, 40 px padding top/bottom, icons stacked with 38 px gap, centre-aligned.

Each icon is **88 × 88 px**, `radius.xl`, border `2 px color.surface.line`, background `color.surface.night2`. The icon itself is a single **italic serif letter at 40 px** in `#B8BCA7`: `T`, `O`, `R`, `I`. No glyphs, no SVG icons.

**Active icon:**
- `background: color.accent.mint`
- `color: color.surface.night`
- `border-color: color.accent.mint`
- `box-shadow: 0 0 0 6 px rgba(159, 231, 194, 0.12)`

Transition: `motion.instant`. No other rail states. The rail is touch-tappable on the console but primarily keyboard-driven (`1 / 2 / 3 / 4`).

## Canvas

Full-bleed inside its grid cell. Colour:

```css
background:
  radial-gradient(35% 40% at 62% 28%, rgba(215, 184, 119, 0.10), transparent 70%),
  linear-gradient(180deg, #0C0B08, #10100A);
```

The subtle gold wash simulates a faintly lit sky and only appears in Observe and Review.

### Guide frame (decorative)

A 2 px dashed border in `#1F1D17`, inset 70 px on all sides, `radius.md`. Plus four L-shaped corner ticks in `rgba(233, 234, 217, 0.45)`, 24 × 24 px at 54 px inset.

This frame is **purely decorative** — it evokes a viewfinder. Do not attach any logic to it. Do not render in Inspect mode (Inspect uses its own split-pane layout).

### Canvas contents

Mode-specific. See `04_modes.md`. But **every canvas hosts the Ledger ribbon** at `position: absolute; left: 56 px; right: 64 px; bottom: 40 px;` — see `05_ledger_and_observer.md`.

## Sidebar

640 px wide. Padding 36 px (top/left/right), 28 px bottom. Flex column, gap 28 px.

Sidebar is a **stack of blocks**. Each block is:

```css
border: 1 px solid color.surface.line;
border-radius: 16 px;
background: color.surface.panel;
padding: 26 px 28 px;
```

Blocks have a title in `body.italicSmall` (italic serif 32 px, `color.ink.primary`) and content in either `kv2` (key/value grid), `big-val` (hero number), `bar` (progress bar), or `keys` (keyboard cheatsheet pills). Each is fully speced in `04_modes.md` per mode.

The sidebar **always ends with a Keys block** showing that mode's keyboard shortcuts. This is non-negotiable — it's how visitors learn the keyboard model.

## Status

Height 80 px. Flex row, centre-aligned, padding 0 56 px, gap 40 px. Background `color.surface.night` (darker than the topbar's gradient).

| Slot | Style | Content |
|---|---|---|
| Live dot | `12 × 12 px` circle, `color.accent.mint`, ring `0 0 0 6 px rgba(159, 231, 194, 0.18)` | Pulses at 2 Hz when live |
| Primary state | `mono.meta` 20 px, `color.ink.primary` | `OBSERVING · FRAME 4,812 / 9,360` |
| Metrics | `mono.meta` 20 px, `color.ink.tertiary` | `51 % · 128 FPS · GPU ON · LOSS · BALANCED` |
| Right slot | `mono.meta` 20 px, `color.ink.tertiary`, `margin-left: auto` | `OUTPUT · URSA-MINOR-01.MP4 · LEDGER JSONL OPEN` |

## State model

The shell is a small state machine:

```
mode        : 'tune' | 'observe' | 'review' | 'inspect'     (default: 'observe')
selection   : DetectionId | null                             (the active bbox across modes)
ledgerFilter: LedgerCategory | null                          (filters canvas when set)
observer    : ObserverCard                                   (see 05)
session     : SessionMetrics                                 (drives sidebar + status)
detectorRun : 'idle' | 'running' | 'paused' | 'ended'
```

Mode switches never drop `selection` or `ledgerFilter`. If the user selects a detection in Observe and switches to Inspect (`I`), Inspect opens on that same detection. If they return to Observe (`2`), the bbox is still selected.

## Keyboard model (complete)

| Key | Everywhere | Mode-specific |
|---|---|---|
| `1` | Switch to Tune | |
| `2` | Switch to Observe | |
| `3` | Switch to Review | |
| `4` | Switch to Inspect | |
| `Esc` | Deselect · clear Ledger filter | |
| `Space` | Pause/resume detector run | |
| `.` | | Observe: pin current detection |
| `F` | | Observe: freeze frame |
| `I` | | Observe: jump to Inspect on selection (=`4`) |
| `A` | | Review: accept current |
| `R` | | Review: reject current |
| `L` | | Review: mark as "learn from this" |
| `Tab` | | Tune: cycle algorithm tab |
| `S` | | Tune: save parameters to profile |
| `Q` | Quit (confirm dialog) | |

This is the entire keyboard surface. Do not add shortcuts in the port unless requested.

## Responsive behaviour across the three scales

| | Desktop 1920×1080 | 4K 2560×1440 | **Gallery 3840×2160** |
|---|---|---|---|
| Rail width | 96 | 136 | **180** |
| Sidebar width | 320 | 480 | **640** |
| Topbar height | 64 | 88 | **120** |
| Status height | 40 | 56 | **80** |
| Rail icon size | 48 | 68 | **88** |
| Ledger cell V numeral | 24 | 32 | **40** |
| Canvas guide inset | 32 | 48 | **70** |

Those values come from `tokens.json > scales`. Everything else scales by the `factor` on that object.

## What is **not** in the shell

- No tabs in the topbar.
- No dropdown menus.
- No settings gear.
- No modal overlays (except the quit-confirm dialog).
- No help button — the Keys block in the sidebar is the help.
- No branding other than the wordmark.

If you feel an urge to add any of these, default to "no" unless a user flow requires it. The shell stays quiet so the instrument can be loud.
