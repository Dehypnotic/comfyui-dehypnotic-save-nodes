## Save MP3 node for ComfyUI

<img width="366" height="245" alt="image" src="https://github.com/user-attachments/assets/6fdae56c-6f65-4581-8af8-1b9e15d5b892" />


Simple, flexible MP3 saver with bitrate options and handy path/filename templates.

Features
- Audio input: accepts common formats used by audio nodes
- File path: ComfyUI output subfolder or absolute path (any drive)
- Filename prefix
- Bitrate mode: variable, constant, average
- Quality: low, medium, high (mapped per mode)
- Outputs: `AUDIO` and `STRING` (bitrate info summary), output node compatible (can terminate a graph)

Installation
1) Go to the `custom_nodes/` directory in ComfyUI.
2) Clone or copy the `comfyui-save-mp3` folder:
   ```bash
   git clone https://github.com/Dehypnotic/comfyui-save-mp3.git
   ```
3) Restart ComfyUI.

Optional backends (no system install required)
- Auto-download ffmpeg: `pip install imageio-ffmpeg` (first run caches a static ffmpeg)
- Drop-in ffmpeg: place `ffmpeg`/`ffmpeg.exe` in a `bin/` folder next to the node
- Or install `lameenc`: `pip install lameenc`

Backend preference: uses ffmpeg when available; otherwise falls back to `lameenc`.

Bitrate/quality mapping
- Variable (VBR): high → `-q:a 0` (~245 kbps), medium → `-q:a 4` (~165 kbps), low → `-q:a 7` (~100 kbps)
- Constant (CBR): high → `320k`, medium → `192k`, low → `128k`
- Average (ABR): high → `256k`, medium → `192k`, low → `160k` (uses `-abr 1`)

Path and filename templates
Placeholders supported in `file_path` and `filename_prefix`:
- `[time(%Y-%m-%d)]` → formatted time (strftime)
- `[date]` → `YYYY-MM-DD`
- `[datetime]` → `YYYY-MM-DD_HH-MM-SS`
- `[unix]` → epoch seconds
- `[guid]` / `[uuid]` → random UUID4 hex
- `[model]` → tries `extra_pnginfo` keys: `model`, `checkpoint`, `ckpt_name`, `model_name`; else `unknown`
- `[env(NAME)]` → environment variable `NAME`

Examples
- `audio/[time(%Y-%m-%d)]`
- `runs/[model]/[datetime]`
- `D:/Exports/[env(USERNAME)]/[guid]`

Security and external save paths (ComfyUI Manager compliant)
- By default, saving is allowed under ComfyUI’s `output/` directory.
- To allow external locations (e.g., other drives), create a local JSON file next to this node named `dehypnotic_save_allowed_paths.json` with:
  ```json
  { "allowed_roots": ["D:/AudioExports", "E:/TeamShare/Audio"] }
  ```
Alternatively (advanced): you can set the environment variable `SAVE_MP3_ALLOWED_PATHS` to point to the JSON file. This is optional — for most users it’s enough to place the JSON file next to the node or in one of the global ComfyUI locations listed below.
- You can also place the file globally under your ComfyUI root:
  - `<ComfyUI>/save_mp3_allowed_paths.json`
  - `<ComfyUI>/config/save_mp3_allowed_paths.json`
  - `<ComfyUI>/user/save_mp3_allowed_paths.json`
  - `<ComfyUI>/user/config/save_mp3_allowed_paths.json`
- The node refuses writes outside `output/` unless the path is under one of the whitelisted roots. Edit this file offline and restart ComfyUI.

Notes
- On Windows, prefer `%H-%M-%S` instead of `%H:%M:%S` in strftime patterns.
- If neither ffmpeg nor lameenc is available, the node raises a clear error with install hints.


Addendum: whitelist behavior and safety
- Recommended location for `save_mp3_allowed_paths.json` is under the ComfyUI root (e.g., `ComfyUI/config/`) so it survives node updates.
- Loader lookup order: env var → global ComfyUI locations → node folder.
- A node‑local file is used only if it defines at least one allowed root; empty example files are ignored.
- Lines starting with `#` are treated as comments in the JSON file.
- An allowed root permits saving in that folder and all subfolders; whitelist a deeper path to restrict more tightly.
