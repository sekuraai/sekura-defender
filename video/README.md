# Submission video

106-second submission video, built in code: an anime.js timeline rendered frame by frame in
headless Chromium and piped to ffmpeg. No screen recording, no editor — every figure on screen
is read from `src/content.js`, which is transcribed from `data/fireworks/report-*.json`, so the
video cannot drift from the results.

**The video is committed here, in two encodings of the same 106 seconds:**

| file | codec | size | use |
|---|---|---|---|
| [`sekura-defender-submission.mp4`](sekura-defender-submission.mp4) | H.264 High / AAC-LC | 6.7 MB | **the submission file** — what browsers, phones and upload pipelines expect |
| [`sekura-defender-submission.webm`](sekura-defender-submission.webm) | VP9 / Opus | 3.7 MB | playback convenience on distros that ship no H.264 decoder (Fedora, openSUSE) |

Both are 1920×1080, 30 fps, 106.0 s, rendered from the same timeline. Re-rendering writes to
`out/`, which is ignored.

```bash
npm install && npx playwright install chromium
node tools/snap.mjs 3 13 27 41 54 74 84 95 103   # still frames for checking
node tools/render.mjs --workers 4                # full render (~90 s)
node tools/serve.mjs                             # scrubber at /src/index.html?preview
```

## Scenes

| # | window | what |
|---|---|---|
| s0 | 0–6 s | title |
| s1 | 6–19 s | the task: a scanner paying frontier prices for schema work |
| s2 | 19–32 s | why fine-tune: deterministic validators |
| s3 | 32–45 s | the data funnel and entity-keyed splits |
| s4 | 45–58 s | training, and why per-step loss is not evidence |
| s5 | 58–78 s | before/after across 13 tasks |
| s6 | 78–89 s | the planner result |
| s7 | 89–100 s | the ship gate failing, and the triage regression |
| s8 | 100–106 s | repo |

Every visual is driven by the single anime.js timeline (`src/main.js`), seeked via
`window.__seek(ms)`. No CSS transitions and no timers, or the render stops being deterministic
and frames differ between workers.

Fonts and `src/vendor/anime.esm.js` are vendored so the render needs no network.
