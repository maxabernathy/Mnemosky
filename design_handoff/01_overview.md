# 01 · Overview — The Observatory

## What this is

Mnemosky is an instrument for finding dim signal (satellite trails) in noisy sky. At a desk it's a Python/OpenCV app. In a gallery, it's a telescope the room looks through — a machine whose thinking is always visible.

The Observatory is the UI shell that unifies every surface Mnemosky currently has (`--preview` × 3, processing dashboard, `--review`, debug, ledger view, config) into **one shell with four modes and one grammar**.

## The concept in one sentence

> One shell. Four modes (Tune · Observe · Review · Inspect). One grammar of frame, sidebar, and Ledger ribbon — scaled from desk to gallery without redesign.

## The four pillars

### 1. One shell, four modes

Every existing surface in v0.2 maps cleanly to one of four modes:

| Mode | Key | Replaces | Purpose |
|---|---|---|---|
| **Tune** | `1` | The three `--preview` windows | Dial algorithm parameters; see a single frame, algorithm tabs, stage-grouped sliders |
| **Observe** | `2` | Processing dashboard | Default view. Full-bleed frame, detections live, Ledger ribbon always visible |
| **Review** | `3` | `--review` | Triage confirmed/rejected with `A · R · L` keys; learned-parameter delta visible on-screen |
| **Inspect** | `4` | Debug + ledger viewer | Drill into one detection: preprocessing · mask · matched-filter response · decision tree |

Mode-switch is a **single keystroke** (`1 / 2 / 3 / 4`). The top bar, rail, canvas frame, and status bar **stay**. Only the sidebar and the canvas contents reconfigure.

### 2. The Ledger is always visible

The Translation Ledger — Mnemosky's seven-category rejection taxonomy — lives as an **ambient ribbon under every frame**:

```
TOO SHORT · STAR-LIKE · LOW SNR · ECCENTRIC · BRIGHT CLUSTER · DUPE · CONFIRMED
   128        44         22        19            11            7      17
```

Categories count up in real time during Observe. Tapping one filters the canvas to **only that kind of rejection**. The STS spine becomes an ambient material of the instrument, not a hidden appendix.

### 3. Observer card, not observer flags

Latitude, longitude, Bortle class, FOV, and free-text notes collapse into a **single uppercase-mono sentence in the top bar**:

```
observer · 40.71°n · -74.01°w · bortle 5 · fov 22° · "clear night, high cirrus" · algorithm: default + matched-filter
```

Haraway's situated knowledge made a literal object. At 8K it reads from 4 m as a quiet paragraph, not a metadata panel. In the desktop port it's a single line above the canvas. Same content, same grammar, different size.

### 4. Organic feedback, scientific data

**Mercury-blob loaders appear only at thresholds** — loading, saving, "learning…", Tier-2 matched-filter passes. Between thresholds the UI stays **geometric** — dashed guide frames, sharp bboxes, hairline dividers.

The instrument's slow thoughts are organic. Its fast answers are precise. Do not mix the two.

## Why it scales to a museum

The same shell plays three sizes without redesign:

| Context | Window | Frame | Sidebar | Type base | Loader size |
|---|---|---|---|---|---|
| Desktop (v0.3 port) | 1920 × 1080 | full-bleed | 320 px | 14 px | 120 px |
| Large desktop / 4K | 2560 × 1440 | full-bleed | 480 px | 18 px | 200 px |
| **Gallery · 8K** | **3840 × 2160** | **full-bleed** | **640 px** | **22 px** | **420 px** |

Only the tokens change. Layout, hierarchy, and keyboard model are identical.

## What's being deliberately rejected

- **No icons in the shell rail.** Letters only. Icons at 8K on a serif-italic stage look like a different product.
- **No dropdown menus.** Everything visible is on-screen; hidden commands are keyboard-only.
- **No modals.** Modes are the only "view change."
- **No gradients except the grass-bokeh background.** Flat fills on every component.
- **No emoji. No rounded-left-border accent cards. No glassy hero gradients.** See "Avoiding AI slop tropes" in project conventions.

## Tone of voice

- Italic serif for **ideas**: "The Observatory sees by rejecting."
- Uppercase mono for **facts**: `OBSERVER · 40.71°N · -74.01°W`.
- Tabular italic serif for **numbers**: `17 CONFIRMED`.
- Never sentence-casing labels. Never title-casing `mnemosky`.

## The relationship between the museum build and the desktop app

They are **the same product at different sizes**, not two apps.

- Phase 1 (this handoff) — ratify the shell grammar. That's what's specified here.
- Phase 2 — port the palette, type pairing, Ledger ribbon, and Observer card into the **existing OpenCV shell** for v0.3.x. No framework change. Partial — the OpenCV windowing won't deliver the same polish, but the language lands.
- Phase 3 — ship the **gallery build** in Qt/PySide6 (or web shell) with the full four modes, attract loop, and room-sensor adapters.

A detector improvement landed in phase 2 lands in phase 3 for free, because both shells talk to the same `satellite_trail_detector.py` core.
