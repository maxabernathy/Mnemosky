# 07 · Attract Loop & Room Adaptation

The gallery build never plays a tape. A **real detection run is always running** on a prepared corpus of footage; the display shifts *what* it shows depending on time of day and visitor presence.

## The six-beat loop

Total runtime ≈ 3m 45s, then the loop returns. But it is **not** a fixed-timecode video — each beat transitions on either a timer *or* a meaningful event from the live detector (whichever comes first). The loop is a state machine over the existing modes, not a playback.

| Beat | Duration | Mode | What it does |
|---|---|---|---|
| 1. **Breath** | 45 s | Observe | Canvas holds still on a confirmed tracklet for ~8 s at a time. Ledger counts up. |
| 2. **The decision tree** | 45 s | Inspect | Dilate a single detection into the four-panel Inspect view. `trefoil` loader bridges the transition. |
| 3. **The correction** | 45 s | Review | Replay a pre-recorded HITL correction: `A A A R L A`. The learned-parameter bar visibly moves. `drip` loader fires on `L`. |
| 4. **Observer's card** | 30 s | Observe | The Observer card dilates to fill the canvas (see `05_ledger_and_observer.md`). Six seconds of Haraway. |
| 5. **Parameters** | 45 s | Tune | The 13 learned parameters breathe along their safety bounds — a slow organism of priors. |
| 6. **Back to the night** | 15 s | Observe | Ledger glows once. Canvas returns to live. Loop restarts. |

## Timing rules

- Beat-to-beat transitions are **always** a `motion.settle` (260 ms) on mode crossfade, + any loader bridge if listed.
- No beat is shorter than 15 s. Sub-15-s beats feel like a commercial.
- `Breath` (beat 1) can stretch to 60 s if detector produces interesting confirmations; the event-based transitions let it run long.

## Room adaptation

Three orthogonal inputs modify the loop:

### 1. Presence sensor (distance + count)

| Room state | Behaviour |
|---|---|
| Empty (no one within 4 m) | Full meditative loop. Type holds large. Long holds. |
| Visitor 2–4 m | Sidebar dilates. Observer card stays in topbar. Keys block fades up in the sidebar. |
| Visitor 0–2 m | Mode is fully interactive. Console lights up. The loop pauses — the detector keeps running but transitions stop auto-firing; visitor drives via keys/touch. |
| Visitor leaves (≥ 10 s empty) | Loop resumes from Beat 1 (Breath). |

Transition from "interactive" back to "attract" fades on `motion.meditative` — slow, so the room doesn't feel like it's kicking the visitor out.

### 2. Time of day (warm/cool drift)

Adjust the **grass-bokeh title and splash surface** across the day. Imperceptible per-minute, meaningful across a visit. Do **not** drift the interior mode surfaces — those stay on the dark palette all day.

| Hour | Shift on `color.accent.grass` |
|---|---|
| 07:00 | +6% warmth (pull toward `#899C6E`) |
| 12:00 | −4% warmth (pull toward `#748A6B`) |
| 17:00 | +6% warmth |
| 21:00+ | baseline |

Shift is linear between anchor points. Day cycle can be pulled from system clock; no external service needed.

### 3. Docent takeover

A hardware key on the console (physical key switch, not software) toggles **Docent mode**:

| | Attract | Docent |
|---|---|---|
| Auto-transitions | on | **off** |
| Keyboard | on | on + **full control** |
| Pinned detections | ignored | **become a slideshow**, advanced with `→` |
| Room sensor | on | **off** (docent controls pace) |
| Quit / profile switches | disabled | enabled |

Exiting Docent mode returns to Attract on the Breath beat.

## Implementation sketch

```ts
type AttractState =
  | { kind: 'attract'; beat: 1|2|3|4|5|6; elapsed: number }
  | { kind: 'interactive'; since: number }
  | { kind: 'docent' };

// every 100 ms:
function tick(now: number, room: RoomSensor, key: DocentKey) {
  if (key.on) return 'docent';
  if (room.present && room.distance < 2) return 'interactive';
  if (state.kind === 'interactive' && now - state.since > 10_000 && !room.present) {
    return { kind: 'attract', beat: 1, elapsed: 0 };
  }
  // …advance beat on timer or detector event
}
```

Detector events that can cut a beat short:

- Beat 1 (Breath): new confirmed detection → hold on it for 6 s
- Beat 5 (Parameters): a parameter touches a safety bound → hold on the slider for 4 s

## Greatest-hits vs. live

The gallery runs on **two streams**:

1. **Live re-processing** of a prepared corpus (Radon + NN hybrid). This is the canvas content in Beat 1, 4, 6.
2. **Pre-recorded HITL corrections** — the "greatest hits" queue used in Beat 3. These are real past corrections, replayed so the model appears to learn on camera.

Visitor-driven interactive sessions in Beat 2 (Inspect) and Beat 5 (Tune) pull from the same live stream as Beat 1.

## What the loop is **not**

- **Not a screensaver.** The detector is always working.
- **Not a slideshow.** Transitions are state changes, not scene cuts.
- **Not narrated.** There is no audio in v2. A docent provides narration when present. If the venue requires ambient audio, specify a room tone (not music, not voice).
