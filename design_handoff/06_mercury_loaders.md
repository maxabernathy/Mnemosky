# 06 · Mercury Loaders

The four polished loaders from `references/Organic Loaders.html`. They appear **only at thresholds** — never as a background animation, never as a spinner replacement for fast operations.

## When to show a loader

| Threshold | Loader | Duration |
|---|---|---|
| Cold start / opening a session file | `trefoil` | Until load complete, min 1.2 s |
| Saving parameters to profile (Tune, `S`) | `gather` | Until write complete, min 0.8 s |
| Tier-2 matched-filter pass (background) | `eye` | Runs for the pass (seconds to minutes) |
| "Learning…" after a Review `L`-mark | `drip` | 1.5–2.5 s (synthetic — we want this beat to feel considered, even if the EMA update is instant) |

Do **not** show a loader for: mode switches, Ledger filter changes, bbox selection, slider drags, frame advances. These are instant.

## The four variants

Each loader is a 280 × 280 px stage (`radius.xl` frame, `color.surface.panel` background). Inside is an SVG at 100% with a `gooey` filter that merges the blobs.

| Variant | Motion | Semantic |
|---|---|---|
| `trefoil` | Three satellites orbit a core, merging in and out | Loading, warming up |
| `gather` | Four blobs from corners gather into the centre, then scatter | Gathering, saving |
| `eye` | A central blob with orbiting satellites that re-merge | Watching, processing a pass |
| `drip` | A pendulum drops a bead that merges into a pool | Committing, learning |

Do not invent a fifth. Do not remix them with different palettes in the shell. The four are a closed set.

## SVG goo technique

The merging effect is a single SVG filter applied to a `<g>` wrapping multiple `<circle>` elements:

```svg
<filter id="goo">
  <feGaussianBlur in="SourceGraphic" stdDeviation="8"/>
  <feColorMatrix values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 22 -11"/>
  <feComposite in="SourceGraphic" operator="atop"/>
</filter>
```

The `feColorMatrix` alpha row `0 0 0 22 -11` is the goo — don't change those numbers. `stdDeviation` controls merge-radius; 8 is right for 280 px stages, bump to 14 for 420 px splash.

## Mercury fill (radial gradient)

```svg
<radialGradient id="merc" cx="38%" cy="28%" r="85%">
  <stop offset="0%"  stop-color="#fbfdf6"/>
  <stop offset="14%" stop-color="#e9efdf"/>
  <stop offset="34%" stop-color="#a9b8a2"/>
  <stop offset="58%" stop-color="#5b6a5a"/>
  <stop offset="82%" stop-color="#2e3830"/>
  <stop offset="100%" stop-color="#141a16"/>
</radialGradient>
```

Optionally layer a `worldTint` (green-tinted, low-opacity, `mix-blend-mode: screen`) over the top for the title/splash hero blobs. Do not use that tint on in-shell loaders — they sit on a dark panel and don't need it.

## Motion

Each loader runs a meditative loop on `motion.meditative` (7 s cycle). Slow and organic. Hover (on capacitive touch: touch) accelerates the loop to 3 s — small reward for interaction during long operations.

No bouncing, no elastic easing. All motion is sinusoidal or smooth linear.

## Accessibility

Loaders are decorative; the threshold's **status message** carries meaning. Example accompanying copy (italic serif 46 px below the loader on splash, 24 px inline in the shell):

- `trefoil`  → *"warming the optics"*
- `gather`   → *"gathering parameters"*
- `eye`      → *"tier 2 pass · frame 4,812 / 9,360"*
- `drip`     → *"learning from your correction"*

Never leave a loader without a caption.

See `references/Organic Loaders.html` for the exact SVG markup of all four.
