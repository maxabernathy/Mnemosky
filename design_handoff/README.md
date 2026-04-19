# Handoff · mnemosky · The Observatory

**Feature:** Museum-edition UI shell for the Mnemosky satellite-trail detector
**Scale target:** 8K (3840 × 2160), standing gallery at 2–4 m viewing distance, capacitive touch console
**Secondary target:** Port the same visual system back into the existing OpenCV/Python desktop shell (v0.3.x)
**Design version:** v2 (April 2026)

---

## 0 · How to read this package

The HTML files in `references/` are **design references, not production code**. They are high-fidelity prototypes that show intended look, motion, and behaviour. Do **not** ship them as-is and do **not** copy their inline CSS verbatim.

Your job is to **recreate the designs inside the target codebase's environment**, re-expressed with the codebase's own component, theming, and state conventions. The existing Mnemosky codebase is a single-file Python OpenCV app (`satellite_trail_detector.py`, ~15 700 lines); a full museum build will likely require a new shell (Qt/PySide6 or a Tauri/Electron web shell wrapping the detector as a service). Phase 2 of delivery — porting the warm palette, type pairing, Ledger ribbon, and Observer card back into OpenCV — can happen in-tree with no migration.

If no shell has been chosen yet, **prefer Qt/PySide6** for the museum build: it matches the Python core, has native 8K scaling, supports capacitive touch, integrates room-sensor hardware via serial/GPIO, and avoids the browser attack surface of a wall-mounted kiosk.

## 1 · Fidelity

**High-fidelity.** All colours, spacing, type sizes, component geometry, and interaction vocabulary in this package are final. Where exact numeric values appear (hex codes, pixel sizes, tracking values, durations), treat them as spec — not suggestion.

The one thing that is **low-fi** is the detection rendering itself (stars, bbox placements in the Observe screenshot). Those are representative placeholders — wire the real detector output in their place.

## 2 · What's in this package

```
design_handoff_observatory/
├── README.md                         ← this file
├── 01_overview.md                    ← the concept, pillars, and mode map
├── 02_design_tokens.md               ← colors, type, spacing, radius, motion (human-readable)
├── tokens.json                       ← same tokens, machine-readable (import into code)
├── 03_shell_spec.md                  ← the four-mode shell: layout, chrome, keyboard, state
├── 04_modes.md                       ← Tune · Observe · Review · Inspect, per-mode specs
├── 05_ledger_and_observer.md         ← Translation Ledger ribbon + Observer card, the two cross-cutting surfaces
├── 06_mercury_loaders.md             ← the four organic loaders: when, how, and the SVG goo technique
├── 07_attract_loop.md                ← the six-beat museum attract loop + room-adaptation rules
├── 08_splash.md                      ← the loading splash spec
├── 09_hardware_and_delivery.md       ← display, console, sensors, phased rollout
└── references/
    ├── Mnemosky UI Audit.html        ← the v2 master concept (primary reference)
    ├── Organic Loaders.html          ← the four polished mercury loaders
    ├── splash_loading.html           ← the cycling splash
    └── archive/                      ← v1s for context only; do NOT implement these
```

## 3 · Where to start

1. Read `01_overview.md` to understand the concept and why the shell is shaped this way.
2. Import `tokens.json` (or transcribe `02_design_tokens.md`) into whatever styling system the target uses — Qt QSS theme, CSS custom properties, SwiftUI `Color`/`Font`, Tailwind config, etc.
3. Build the shell skeleton from `03_shell_spec.md` first. The Observer bar, mode rail, canvas, sidebar, Ledger ribbon, and status bar are all one layout — every mode reuses it.
4. Implement modes in this order: **Observe → Review → Tune → Inspect.** Observe is the default view, Review has the most-used keyboard flow, Tune and Inspect reuse their machinery.
5. Add mercury loaders (`06_mercury_loaders.md`) only at the threshold moments listed — do not sprinkle them.
6. Wire the attract loop (`07_attract_loop.md`) last; it's a state machine on top of the existing modes.

## 4 · Assets

- **No raster assets.** Everything in the design is rendered from type, CSS, or SVG.
- **No brand assets to license.** "mnemosky" is set in Times New Roman italic (or any system serif italic if that's unavailable — Playfair Display Italic, Cormorant Italic, and Source Serif Italic are acceptable substitutes; do not use a modern sans).
- **Monospace** is the only other family: JetBrains Mono, IBM Plex Mono, or SF Mono / Consolas / the platform default. Avoid Roboto Mono.
- **Icons** are not used in the shell. Letters (`T · O · R · I`) stand in for the four modes on the left rail — this is intentional, do not replace with glyphs.

## 5 · Content & copy

All visible copy in the references is final and intentional. Notably:

- Lowercase `mnemosky` wordmark — never title-case it.
- Italic tagline `centrifugal data management` — this is the product descriptor.
- All metadata/labels are **uppercase mono with 0.24–0.30 em tracking**. No lowercase labels anywhere.
- Number + unit pattern is always `{italic-serif numeral} {uppercase-mono unit}` — e.g. `17 CONFIRMED`, `4,812 FRAMES`. Never `17 confirmed` or `17Confirmed`.
- The seven Translation Ledger categories are fixed: `too short · star-like · low snr · eccentric · bright cluster · dupe · confirmed`. Do not rename or add.

## 6 · What this package does NOT cover

- **Detector algorithm changes.** The four-mode shell and Ledger ribbon imply no changes to preprocessing, matched filtering, classification, or HITL. Keep that machinery; the UI re-exposes it.
- **Data schemas / JSONL format.** Wire into whatever the existing detector emits.
- **Authentication, analytics, telemetry.** Out of scope.
- **Localisation.** All copy is English-only in v2. Decide per-venue.
- **Accessibility at gallery scale.** Type is already 22 px+ and contrast ratios are safe, but if the museum requires WCAG AA conformance audit, that's a follow-up pass.

## 7 · Open questions for the developer

Listed at the end of `09_hardware_and_delivery.md`. Tldr: venue dimensions, ambient light spec, confirmed corpus of footage, docent-script language, final call on Qt vs. web shell, hardware budget.

---

_Prepared by mnemozine · tracksuit — observatory master concept v2 · April 2026._
