# 04 · The Four Modes

Each mode reuses the shell from `03_shell_spec.md`. This doc specifies what goes **inside the canvas and sidebar** for each.

---

## 2 · Observe (the default)

Replaces: the processing dashboard.

### Canvas

Full-bleed rendering of the current video frame (or still). Over it:

- **Stars** — rendered from the detector's star map. `4 × 4 px` circles, `rgba(233, 234, 217, 0.7)`, with a soft `box-shadow: 0 0 10 px rgba(233, 234, 217, 0.4)`. Larger stars (brighter) at 6 × 6 px. These are data-driven, not decorative.
- **Detection bboxes** — one per live detection. Border 3 px in the appropriate semantic colour; see `02_design_tokens.md` for the glow shadow recipe. Each has a **tag** at `top: -44 px; left: 0` in `mono.meta` 20 px, colour matches the bbox.

Example tag copy:

```
SAT · SNR 3.2 · TRACKLET #5
PLANE
SAT · SNR 2.7 · TRACKLET #8
ANOMALOUS · UNCLASSIFIED
```

- **Ledger ribbon** — always, at the bottom of the canvas. See `05_ledger_and_observer.md`.

### Sidebar blocks (top to bottom)

1. **Session eyebrow** — `mono.label` 18 px, `color.ink.tertiary`, text `SESSION · TRACKSUIT`.
2. **Current detection** — shown only when one is selected.
   - Title: "Current detection" (italic serif 32 px)
   - `kv2` grid (180 px / 1fr):
     - TYPE → value in `color.semantic.*`
     - SNR → numeric
     - LENGTH → `285 px`
     - ANGLE → `135°`
     - BRIGHTNESS → `12.5 / 28`
     - TRACKLET → `#5 · 7 frames`
3. **Confidence**
   - Title: "Confidence"
   - Progress bar (12 px high, `radius.sm`, bg `#211D18`) with mint→mercury gradient fill
   - `kv2` below: PLATT, FUSION, RESCUE values
4. **Live · this run**
   - Title: "Live · this run"
   - Hero number: `numeric.hero` value + `mono.micro` unit (e.g. `17 CONFIRMED`)
   - `kv2` below: FRAMES, TRACKLETS, THROUGHPUT
5. **Keys**
   - Title: "Keys"
   - Pill row:
     ```
     F · freeze   . · pin   I · inspect   3 · review   L · learn   Q · quit
     ```

### Interactions

- Tap a bbox on the canvas → selects that detection, populates "Current detection" + "Confidence" blocks. Also sets `selection` in global state.
- `.` pins the current detection (appears in the attract loop later).
- `F` freezes the frame (detector keeps running in background; canvas stops updating).
- Tap a Ledger cell → filters canvas to show only that rejection category.
- `2` re-enters Observe from any other mode with state preserved.

### Empty states

- No detections yet: canvas shows the live frame, Ledger ribbon shows 0s, sidebar shows only the Session eyebrow, Live block, and Keys.
- Detector idle: "Current detection" block shows a single italic-serif line *"No detector running — press Space to begin."*

---

## 1 · Tune

Replaces: the three `--preview` windows.

### Canvas

One centered frame at detector aspect ratio. The canvas swaps between preview stages via a **tab strip** at the top of the canvas area (inside the dashed guide):

```
RAW  │  PREPROCESSED  │  MASK  │  MATCHED-FILTER  │  CLASSIFIED
```

Tabs are `mono.label` 20 px with 0.28 em tracking, `color.ink.secondary`, separated by `│` characters in `color.surface.line`. Active tab is `color.accent.mint`.

Keyboard: `Tab` cycles forward, `Shift+Tab` cycles back.

**No Ledger ribbon in Tune.** The Ledger is a runtime concept; tuning happens before detections exist.

### Sidebar blocks (top to bottom)

1. **Algorithm** — radio group:
   - `DEFAULT`
   - `MATCHED-FILTER`
   - `RADON`
   - `HYBRID`
2. **Stage-grouped sliders** — one block per stage: `PREPROCESS`, `DETECT`, `CLASSIFY`, `TIER 2`. Each block holds 3–6 sliders. Slider row:
   - `mono.label` 18 px name on the left (tracked, uppercased)
   - Track 12 px tall, bg `#211D18`, fill mint
   - Numeric value on the right in `numeric.cell` (italic serif 40 px)
3. **Delta · vs. saved** — list of currently-dirty parameters with their ΔFROM/TO values. Empty when unchanged.
4. **Actions**
   - Apply to session (primary)
   - Save to profile… (secondary — opens a small inline naming field)
   - Revert
5. **Keys** — `Tab · cycle stage`, `S · save`, `Esc · revert`, `2 · observe`

### Interactions

- Slider change is live: the canvas re-renders the current frame through the modified pipeline. Debounce at 120 ms.
- `S` saves; inline field appears for a profile name; `Enter` commits, `Esc` cancels.
- Profile list is out of scope for this package — mirror whatever the existing detector uses.

---

## 3 · Review

Replaces: `--review`.

### Canvas

Two-pane split **at the guide frame level** (not full-canvas). Left: the current detection's full-frame context with the bbox highlighted. Right: a zoomed 3×3 crop stack (past/present/future frames in a 3×3 grid), so the reviewer can judge trajectory.

**The Ledger ribbon still lives at the bottom of the canvas.** Tapping a category during Review does *not* filter (filtering is an Observe concept); instead it queues "show me the next N rejections of this type." This is the one place the Ledger has a slightly different behaviour — be intentional about it.

### Sidebar blocks (top to bottom)

1. **Queue position** — `mono.label` + `numeric.hero` showing `28 / 112` of current pass.
2. **Current rejection** — category + reason in italic serif body, plus the seven rejection flags the detector raised on this candidate (list of mono labels, the triggered ones in `color.accent.mint`, the rest in `color.ink.tertiary`).
3. **Learned parameter delta** — this is the HITL-in-public block. Shows the parameter(s) that will move when you `L`-mark this item. Before you press `L`:
   ```
   LENGTH-MIN         14.00 → 14.00        (no change pending)
   ```
   After `L` is pressed (and before moving on):
   ```
   LENGTH-MIN         14.00 → 13.62        ± 0.38
   ```
   in `color.accent.mint`. Animating the numeric change with `motion.settle`.
4. **Keys**
   ```
   A · accept   R · reject   L · learn   ← back   → skip   2 · observe
   ```

### Interactions

- `A` / `R` advance the queue by 1.
- `L` tags the item and updates the "Learned parameter delta" block in place with the pending delta (uses `motion.settle`). The actual Tier-1 EMA update fires on the next `A` or `R` after `L`.
- `←` / `→` move through the queue without acting.
- The keyboard rhythm `A A A R A L A A R A` is the intended loop — optimise for that rhythm; no confirmation dialogs.

### Empty states

- Queue empty: canvas shows the Observer card dilated, sidebar shows a single italic-serif line *"Nothing to review."* and the Keys block with `2 · observe`.

---

## 4 · Inspect

Replaces: the debug view + Ledger viewer.

### Canvas

**No dashed guide frame.** Canvas splits into a 2 × 2 grid of sub-panels, each with its own thin border and `mono.label` title:

```
┌───────────────────────┬───────────────────────┐
│ PREPROCESSING         │ MASK                  │
│ (denoised frame)      │ (binary mask overlay) │
├───────────────────────┼───────────────────────┤
│ MATCHED-FILTER        │ DECISION TREE         │
│ (response heatmap)    │ (classifier path)     │
└───────────────────────┴───────────────────────┘
```

All four panels are **of the current `selection`** (the detection carried in from another mode). If `selection === null`, Inspect shows a full-canvas italic-serif prompt: *"Select a detection first. Press 2."*

The **Ledger ribbon stays at the bottom** but in Inspect it shows counts for the candidate's *neighbourhood* (same tracklet, same frame ±10), not the session total. Label changes to `NEIGHBOURHOOD · FRAME ±10` in `mono.label` above the ribbon.

### Sidebar blocks (top to bottom)

1. **Selection** — big tag (italic serif 60 px) showing tracklet ID, plus a second-line `mono.meta` with `FRAME 4,812 · T+45m12s`.
2. **Classifier decision** — a vertical decision path:
   ```
   ENTER                  ✓
   LENGTH > 14 px         ✓  (285 px)
   ECCENTRICITY > 0.90    ✓  (0.976)
   BRIGHTNESS IN BAND     ✓  (12.5 / 28)
   TRACKLET ≥ 4 FRAMES    ✓  (7 frames)
   CONFIDENCE > 0.65      ✓  (0.82)
   ─────────────────────────
   CLASSIFY               SATELLITE
   ```
   Green check = passed, red `×` = failed. Path rendered with 1 px line connectors in `color.surface.line`.
3. **Raw values** — a `kv2` grid of every classifier input value.
4. **Keys** — `← · prev in tracklet   → · next in tracklet   3 · review this   Esc · back to observe`

### Interactions

- The four panels are non-interactive (they're a diagnostic view). Right-click or long-press on a panel: *"Open frame in external viewer"* — out of scope for the museum, keep for the desktop port.
- `←` / `→` traverse the tracklet; other modes update their `selection` in sync.

---

## Mode-transition behaviour

Switching modes is **near-instant**. The topbar, rail, canvas frame, and status stay. The canvas contents and sidebar crossfade on `motion.instant` (120 ms). The rail's active icon also transitions on `motion.instant`.

There is **no full-screen transition, no slide, no flash.** The same room continues — just different furniture.
