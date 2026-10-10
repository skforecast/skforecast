# Animation export

Exports the documentation animations in [`docs/animations/`](../../../docs/animations/) to MP4 videos, for example to insert them in a presentation. PowerPoint and Keynote play MP4 (H.264) natively.

## Setup

Requires Node.js 22.12 or later and Google Chrome. The dependencies are installed locally in this folder (`node_modules/`, git-ignored):

```bash
cd tools/docs/animations_to_video
npm install
```

- `puppeteer-core` drives the installed Google Chrome (no browser is downloaded).
- `ffmpeg-static` provides an ffmpeg binary, so ffmpeg does not need to be installed in the system.

## Usage

```bash
cd tools/docs/animations_to_video

node export_video.mjs                          # all animations, light and dark themes
node export_video.mjs global-forecasting       # a single animation
node export_video.mjs --theme light --fps 30 --scale 2 --out videos

# animations of another repository that uses the same engine
node export_video.mjs --src ../../../../skforecast-ai/docs/animations deterministic-first
```

Videos are written to `tools/docs/animations_to_video/videos/` (git-ignored), named `<animation>-<theme>.mp4`.

| Option | Default | Description |
|:-------|:--------|:------------|
| `--theme` | `both` | `light`, `dark` or `both` |
| `--fps` | `30` | Frames per second |
| `--scale` | `2` | Pixel density: `2` gives 2560 pixels wide, sharp on projectors |
| `--out` | `videos` | Output folder, relative to this folder |
| `--src` | `../../../docs/animations` | Folder with the animations, relative to this folder. Point it to another repository that uses the same engine (`skf-anim.js`) to export its animations |

Set `CHROME_PATH` if Chrome is not installed in its default location.

## How it works

Each animation is a pure function of time. The script serves `docs/animations/` locally, opens every page with `?export=1&theme=...` (controls hidden, `window.skfRender(t)` exposed by the engine) and renders every frame at its exact time before capturing it. Nothing runs in real time, so the result does not depend on the speed of the machine. The frames are piped to ffmpeg and encoded with H.264 (`crf 18`, `yuv420p`, `faststart`).

The animations are part of the skforecast documentation, licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The exported videos keep the skforecast mark in the bottom-right corner.
