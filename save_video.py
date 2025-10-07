# save_video.py
# ComfyUI node: Save Video (imageio-ffmpeg, audio as input w/ auto SR)
#
# - Inputs: images (IMAGE), optional audio (AUDIO)
# - Containers/codecs configurable (mp4/mkv/webm/mov with h264/h265/vp9/av1/prores/dnxhr)
# - Sequential filenames, optional date subfolder; can also export selected frames only
# - Loops single frame to audio length automatically
# - Uses imageio-ffmpeg (bundled FFmpeg), no system install needed
#
# pip install imageio imageio-ffmpeg

import os
import re
import math
import wave
import json
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import imageio_ffmpeg  # type: ignore
    _IMAGEIO_FFMPEG_ERROR = None
except Exception as exc:
    imageio_ffmpeg = None  # type: ignore
    _IMAGEIO_FFMPEG_ERROR = exc


VALID_PRESETS = ("ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow")

VIDEO_CODEC_OPTIONS = {
    "h264": {
        "label": "H.264 (libx264)",
        "ffmpeg": "libx264",
        "pix_fmt": "yuv420p",
        "args": [],
        "supports_crf": True,
        "supports_preset": True,
    },
    "h265": {
        "label": "H.265 / HEVC (libx265)",
        "ffmpeg": "libx265",
        "pix_fmt": "yuv420p",
        "args": [],
        "supports_crf": True,
        "supports_preset": True,
    },
    "vp9": {
        "label": "VP9 (libvpx-vp9)",
        "ffmpeg": "libvpx-vp9",
        "pix_fmt": "yuv420p",
        "args": [["-b:v", "0"]],
        "supports_crf": True,
        "supports_preset": False,
    },
    "av1": {
        "label": "AV1 (libaom-av1)",
        "ffmpeg": "libaom-av1",
        "pix_fmt": "yuv420p",
        "args": [["-b:v", "0"], ["-cpu-used", "6"], ["-row-mt", "1"]],
        "supports_crf": True,
        "supports_preset": False,
    },
    "prores": {
        "label": "ProRes 422 HQ (prores_ks)",
        "ffmpeg": "prores_ks",
        "pix_fmt": "yuv422p10le",
        "args": [["-profile:v", "3"]],
        "supports_crf": False,
        "supports_preset": False,
    },
    "dnxhr": {
        "label": "DNxHR HQ (dnxhr_hq)",
        "ffmpeg": "dnxhr_hq",
        "pix_fmt": "yuv422p10le",
        "args": [],
        "supports_crf": False,
        "supports_preset": False,
    },
}

CONTAINER_OPTIONS = {
    "mp4": {
        "label": "MP4",
        "extension": "mp4",
        "allowed_codecs": {"h264", "h265", "av1"},
        "audio_codec": "aac",
        "extra": [["-movflags", "+faststart"]],
    },
    "mkv": {
        "label": "Matroska (MKV)",
        "extension": "mkv",
        "allowed_codecs": set(VIDEO_CODEC_OPTIONS.keys()),
        "audio_codec": "aac",
        "extra": [],
    },
    "webm": {
        "label": "WebM",
        "extension": "webm",
        "allowed_codecs": {"vp9", "av1"},
        "audio_codec": "libopus",
        "extra": [],
    },
    "mov": {
        "label": "QuickTime MOV",
        "extension": "mov",
        "allowed_codecs": {"h264", "h265", "prores", "dnxhr"},
        "audio_codec": "aac",
        "extra": [],
    },
}

# ------------------------- helpers -------------------------

def _next_seq_number(folder: Path, prefix: str, delim: str, padding: int) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}{re.escape(delim)}(\d{{{padding}}})\b")
    max_num = 0
    if folder.exists():
        for p in folder.iterdir():
            if not p.is_file():
                continue
            m = pattern.match(p.stem)
            if m:
                try:
                    n = int(m.group(1))
                    if n > max_num:
                        max_num = n
                except ValueError:
                    pass
    return max_num + 1

def _normalize_frames(images) -> List[np.ndarray]:
    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    data = images

    if isinstance(data, (list, tuple)) and len(data) == 1:
        single = data[0]
        if torch is not None and isinstance(single, torch.Tensor):
            single = single.detach().cpu().numpy()
        if isinstance(single, np.ndarray) and single.ndim == 4:
            data = single

    if torch is not None and isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    frames_list: List[np.ndarray] = []

    if isinstance(data, np.ndarray):
        if data.ndim == 4:
            frames_list = [data[i] for i in range(data.shape[0])]
        elif data.ndim == 3:
            frames_list = [data]
        else:
            raise ValueError(f"Expected IMAGE as [N,H,W,C] or [H,W,C], got {data.shape}")
    elif isinstance(data, (list, tuple)):
        for item in data:
            if torch is not None and isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if isinstance(item, np.ndarray) and item.ndim == 4:
                frames_list.extend([item[i] for i in range(item.shape[0])])
            else:
                frames_list.append(item)
    else:
        frames_list = [data]

    out: List[np.ndarray] = []
    for f in frames_list:
        a = np.asarray(f)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError(f"Expected frame [H,W,3/4], got {a.shape}")
        if a.dtype != np.uint8:
            a = np.clip(a, 0.0, 1.0)
            a = (a * 255.0).round().astype(np.uint8)
        if a.shape[2] == 4:
            a = a[:, :, :3]
        out.append(a)
    return out

def _split_audio_input(audio: Union[np.ndarray, tuple, list, dict]) -> Tuple[np.ndarray, int]:
    sr = 48000
    samples = None

    try:
        import torch  # type: ignore
    except Exception:
        torch = None  # type: ignore

    if isinstance(audio, dict):
        sr_val = audio.get("sample_rate") or audio.get("sr")
        if sr_val is not None:
            if torch is not None and isinstance(sr_val, torch.Tensor):
                sr = int(sr_val.detach().cpu().item())
            elif hasattr(sr_val, "item"):
                sr = int(sr_val.item())
            else:
                sr = int(sr_val)
        for key in ("samples", "waveform", "audio", "data"):
            if key in audio:
                samples = audio[key]
                break
        if samples is None:
            raise ValueError("Audio dict missing samples (samples/waveform/audio/data).")

    elif isinstance(audio, (tuple, list)):
        sample_candidate = None
        sr_candidate = None
        for item in audio:
            if sample_candidate is None and (
                hasattr(item, "shape")
                or isinstance(item, (list, tuple))
                or (hasattr(item, "__array__"))
            ):
                sample_candidate = item
                continue
            if sr_candidate is None and isinstance(item, (int, float)) and not isinstance(item, bool):
                sr_candidate = int(item)
        if sample_candidate is None and len(audio) > 0:
            sample_candidate = audio[0]
        if sample_candidate is None:
            raise ValueError("AUDIO tuple/list did not contain samples")
        samples = sample_candidate
        if sr_candidate is not None:
            sr = sr_candidate
    else:
        samples = audio

    if torch is not None and isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    a = np.asarray(samples)
    if a.ndim == 0:
        a = a.reshape(1, 1)
    if a.ndim == 3:
        shapes = a.shape
        sample_axis = int(np.argmax(shapes))
        remaining = [ax for ax in range(3) if ax != sample_axis]
        channel_axis = None
        batch_axis = None
        for ax in remaining:
            if channel_axis is None and shapes[ax] <= 8:
                channel_axis = ax
            else:
                batch_axis = ax
        if channel_axis is None:
            channel_axis = remaining[0]
            batch_axis = remaining[1]
        if batch_axis is None:
            batch_axis = remaining[1]
        a = np.moveaxis(a, (batch_axis, channel_axis, sample_axis), (0, 1, 2))
        a = a[0]

    if a.ndim == 1:
        a = a.reshape(1, -1)
    elif a.ndim == 2:
        if a.shape[0] < a.shape[1] and a.shape[1] <= 8:
            a = a.T
    else:
        a = np.squeeze(a)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        elif a.ndim == 2:
            pass
        else:
            raise ValueError(f"AUDIO must be 1D or 2D, got {a.shape}")

    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if a.max() > 1.5 or a.min() < -1.5:
        a = np.clip(a, -32768, 32767) / 32767.0
    else:
        a = np.clip(a, -1.0, 1.0)

    return a, int(sr)

def _audio_to_wav_path(audio_in: Union[np.ndarray, tuple, list]) -> str:
    a, sr = _split_audio_input(audio_in)
    a16 = (np.clip(a, -1.0, 1.0) * 32767.0).round().astype(np.int16)
    nch, nsamps = a16.shape

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(a16.T.tobytes())
    return tmp_path

def _parse_ffmpeg_duration(stderr_text: str) -> Optional[float]:
    m = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}\.\d+|\d{2})", stderr_text)
    if not m:
        return None
    hh, mm, ss = m.groups()
    try:
        return int(hh)*3600 + int(mm)*60 + float(ss)
    except Exception:
        return None

def _probe_audio_duration(ffmpeg_exe: str, audio_path: str) -> Optional[float]:
    try:
        p = subprocess.run([ffmpeg_exe, "-hide_banner", "-i", audio_path],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return _parse_ffmpeg_duration(p.stderr or "")
    except Exception:
        return None

def _build_cmd(ffmpeg_exe: str, w: int, h: int, fps: int,
               out_path: Path, codec_info: dict, container_info: dict,
               acodec: Optional[str], audio_path: Optional[str],
               crf: int, preset: str) -> list:
    cmd = [
        ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-"
    ]
    if audio_path and acodec:
        cmd += ["-i", audio_path]

    vf = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    cmd += ["-vf", vf, "-c:v", codec_info["ffmpeg"]]

    if codec_info.get("supports_preset"):
        cmd += ["-preset", preset]
    if codec_info.get("supports_crf"):
        cmd += ["-crf", str(crf)]

    pix_fmt = codec_info.get("pix_fmt")
    if pix_fmt:
        cmd += ["-pix_fmt", pix_fmt]

    for extra in codec_info.get("args", []):
        cmd += extra

    if audio_path and acodec:
        cmd += ["-c:a", acodec, "-b:a", "192k", "-ar", "48000"]
    else:
        cmd += ["-an"]

    for extra in container_info.get("extra", []):
        cmd += extra

    cmd += [str(out_path)]
    return cmd

# --------------------------- node ---------------------------

class SaveVideo:
    """
    Save Video (simple) â€” minimal kontroller, audio som direkte input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_mode": (("video", "frames", "video & frames"), {"default": "video", "tooltip": "Choose whether to save video, frames, or both."}),
                "images": ("IMAGE", {"tooltip": "Frame input. Batch data supported."}),
                "output_dir": ("STRING", {"default": "output/video", "tooltip": "Folder where the video is saved."}),
                "date_subfolder_pattern": ("STRING", {"default": "", "tooltip": "Optional strftime pattern for subfolders, e.g. %Y-%m-%d."}),
                "filename_prefix": ("STRING", {"default": "VID", "tooltip": "Filename prefix, e.g. VID_0001.mp4."}),
                "filename_delimiter": ("STRING", {"default": "_", "tooltip": "Delimiter between prefix and sequence number."}),
                "number_padding": ("INT", {"default": 4, "min": 1, "max": 10, "tooltip": "Digits in the sequence number (0001, 0002, ...)."}),
                "number_start": ("INT", {"default": 1, "min": 0, "max": 1_000_000, "tooltip": "Starting value for the sequence number."}),
                "container": (tuple(CONTAINER_OPTIONS.keys()), {"default": "mp4", "tooltip": "Container format (mp4, mkv, webm, mov)."}),
                "video_codec": (tuple(VIDEO_CODEC_OPTIONS.keys()), {"default": "h264", "tooltip": "Video codec to use for encoding."}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240, "tooltip": "Frames per second (CFR)."}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51, "tooltip": "Quality (lower = better, larger files). Typical 18-28 for H.264."}),
                "preset": (VALID_PRESETS, {"default": "fast", "tooltip": "Encoder speed versus quality (ultrafast ... veryslow)."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional audio track. Mono/stereo supported."}),
                "loop_still_to_audio": ("BOOLEAN", {"default": True, "tooltip": "If only one frame plus audio, loop the frame to match audio length."}),
                "show_progress": ("BOOLEAN", {"default": True, "tooltip": "Write progress information to the console."}),
                "show_preview": ("BOOLEAN", {"default": True, "tooltip": "Generate preview sequence in the node output."}),
                "preview_step": ("INT", {"default": 1, "min": 1, "max": 50, "tooltip": "Show every Nth frame in the preview sequence."}),
                "preview_max_frames": ("INT", {"default": 24, "min": 1, "max": 120, "tooltip": "Maximum number of frames generated for preview."}),
                "frames_dir": ("STRING", {"default": "", "tooltip": "Save selected frames here (relative to the video folder). Leave empty to skip."}),
                "frames_select": ("STRING", {"default": "-2", "tooltip": "Frames to save: -2=last, -1=all, 0=first, or comma list (e.g. 0,5,10)."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "video_path",)
    FUNCTION = "save"
    CATEGORY = "video/io"
    OUTPUT_NODE = True

    # ----------------------- path helpers -----------------------

    def _normalize_path(self, path: Path) -> Path:
        return Path(os.path.abspath(str(path)))

    def _base_output_dir(self) -> Path:
        try:
            from folder_paths import get_output_directory  # type: ignore
            base = Path(get_output_directory()).expanduser()
        except Exception:
            base = Path.cwd() / "output"
        return self._normalize_path(base)

    def _comfy_root(self) -> Path:
        base = self._base_output_dir()
        return self._normalize_path(base.parent)

    def _load_allowed_roots(self) -> List[Path]:
        env_cfg = os.environ.get("DEHYPNOTIC_SAVE_ALLOWED_PATHS")
        candidates: List[str] = []
        if env_cfg:
            candidates.append(env_cfg)

        comfy_root = self._comfy_root()
        names = ("dehypnotic_save_allowed_paths.json", "allowed_paths.json")
        for name in names:
            candidates.append(str(comfy_root / "user" / "config" / name))
            candidates.append(str(comfy_root / "user" / name))
            candidates.append(str(comfy_root / "config" / name))
            candidates.append(str(comfy_root / name))

        here = Path(__file__).resolve().parent
        for name in names:
            candidates.append(str(here / name))

        seen = set()
        for path_str in candidates:
            if not path_str:
                continue
            candidate = Path(os.path.expandvars(path_str)).expanduser()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                try:
                    with open(candidate, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                except Exception:
                    continue
                filtered = "\n".join(
                    line for line in raw.splitlines() if not line.lstrip().startswith("#")
                )
                try:
                    data = json.loads(filtered)
                except Exception:
                    continue
                entries = []
                if isinstance(data, dict):
                    entries = data.get("allowed_roots") or data.get("roots") or []
                elif isinstance(data, list):
                    entries = data
                roots: List[Path] = []
                for entry in entries:
                    if isinstance(entry, str):
                        roots.append(self._normalize_path(Path(os.path.expandvars(entry)).expanduser()))
                if roots:
                    return roots
        return []

    def _same_drive(self, a: Path, b: Path) -> bool:
        da = os.path.splitdrive(str(self._normalize_path(a)))[0].lower()
        db = os.path.splitdrive(str(self._normalize_path(b)))[0].lower()
        return da == db

    def _is_under_dir(self, path: Path, base: Path) -> bool:
        try:
            ap = self._normalize_path(path)
            bp = self._normalize_path(base)
            if not self._same_drive(ap, bp):
                return False
            return os.path.commonpath([str(ap), str(bp)]) == str(bp)
        except Exception:
            return False

    def _validate_target_dir(self, target_dir: Path) -> None:
        target_abs = self._normalize_path(target_dir)
        base_output = self._base_output_dir()
        if self._is_under_dir(target_abs, base_output):
            return

        for root in self._load_allowed_roots():
            if self._is_under_dir(target_abs, root):
                return

        msg = (
            "External save path is not allowed.\n"
            "This node only writes inside ComfyUI's output directory, "
            "unless the path is whitelisted offline.\n\n"
            "To allow external locations, create/edit a JSON file named "
            "'dehypnotic_save_allowed_paths.json' in your ComfyUI root (or user/config) folder "
            "with content like:\n\n"
            '{\n  "allowed_roots": ["D:/VideoExports", "E:/TeamShare/Video"]\n}\n\n'
            "You can also set the DEHYPNOTIC_SAVE_ALLOWED_PATHS environment variable to point to this file."
        )
        raise PermissionError(msg)

    def _output_folder(self, base: str, pattern: str) -> Path:
        base_output = self._base_output_dir()
        user_path = Path(str(base or "")).expanduser()

        if user_path.is_absolute():
            folder = user_path
        else:
            rel_parts = [p for p in user_path.parts if p and p != "."]
            if rel_parts and rel_parts[0].lower() in ("output", "outputs"):
                rel_parts = rel_parts[1:]
            rel_path = Path(*rel_parts) if rel_parts else Path()
            folder = base_output / rel_path

        if pattern.strip():
            pattern_part = Path(datetime.now().strftime(pattern.strip()))
            folder = folder / pattern_part

        folder = self._normalize_path(folder)
        self._validate_target_dir(folder)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def save(
        self,
        save_mode,
        images,
        output_dir,
        date_subfolder_pattern,
        filename_prefix,
        filename_delimiter,
        number_padding,
        number_start,
        container,
        video_codec,
        fps,
        crf,
        preset,
        audio=None,
        loop_still_to_audio=True,
        show_progress=True,
        show_preview=True,
        preview_step=1,
        preview_max_frames=24,
        frames_dir="",
        frames_select="0",
    ):
        if imageio_ffmpeg is None:
            msg = (
                "Save Video node requires 'imageio' and 'imageio-ffmpeg'. "
                "Install with: pip install imageio imageio-ffmpeg."
            )
            if _IMAGEIO_FFMPEG_ERROR:
                msg += f" Import error: {_IMAGEIO_FFMPEG_ERROR}"
            raise RuntimeError(msg)

        frames = _normalize_frames(images)
        if not frames:
            raise ValueError("No frames provided.")

        mode = str(save_mode).strip().lower()
        save_video = mode in ("video", "video & frames")
        save_frames = mode in ("frames", "video & frames")
        if not save_video and not save_frames:
            save_video = True

        container_key = str(container).lower()
        codec_key = str(video_codec).lower()

        container_info = CONTAINER_OPTIONS.get(container_key)
        codec_info = VIDEO_CODEC_OPTIONS.get(codec_key)

        if save_video:
            if container_info is None:
                raise ValueError(f"Unsupported container '{container}'.")
            if codec_info is None:
                raise ValueError(f"Unsupported video codec '{video_codec}'.")
            if codec_key not in container_info["allowed_codecs"]:
                allowed = ", ".join(sorted(container_info["allowed_codecs"]))
                raise ValueError(
                    f"Codec '{codec_key}' is not supported in container '{container_key}'. Allowed: {allowed}."
                )
        else:
            if container_info is None:
                container_info = {"extension": "bin", "audio_codec": None, "extra": []}
            if codec_info is None:
                codec_info = VIDEO_CODEC_OPTIONS["h264"]

        acodec = None
        audio_path = None
        tmp_wav = None
        if save_video and audio is not None:
            acodec = container_info.get("audio_codec")
            if acodec:
                tmp_wav = _audio_to_wav_path(audio)
                audio_path = tmp_wav

        total_frames = len(frames)

        base = self._output_folder(output_dir, date_subfolder_pattern)
        seq = _next_seq_number(base, filename_prefix, filename_delimiter, number_padding)
        if number_start > 0:
            seq = max(seq, number_start)
        stem = f"{filename_prefix}{filename_delimiter}{seq:0{number_padding}d}"

        out_path = None
        video_dir = base
        video_path_str = ""

        if save_video:
            extension = container_info.get("extension", "mp4")
            out_path = self._normalize_path(base / f"{stem}.{extension}")
            video_dir = out_path.parent

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()  # type: ignore[arg-type]
            if len(frames) == 1 and audio_path and loop_still_to_audio:
                dur = _probe_audio_duration(ffmpeg_exe, audio_path)
                if dur and dur > 0:
                    total_frames = int(math.ceil(dur * fps))
                    if show_progress:
                        print(f"[SaveVideo] Looping single frame for {dur:.2f}s -> {total_frames} frames @ {fps} fps")
                elif show_progress:
                    print("[SaveVideo] Could not read audio duration; using single frame only.")

            h, w, _ = frames[0].shape
            cmd = _build_cmd(
                ffmpeg_exe=ffmpeg_exe, w=w, h=h, fps=fps,
                out_path=out_path, codec_info=codec_info, container_info=container_info,
                acodec=acodec, audio_path=audio_path, crf=crf, preset=preset
            )

            if show_progress:
                try:
                    print(f"[SaveVideo] Output base: {base.resolve()}")
                except Exception:
                    print(f"[SaveVideo] Output base: {base}")
                print(f"[SaveVideo] Container: {container_key} | Codec: {codec_key}")
                print(f"[SaveVideo] -> {out_path}")
                print(f"[SaveVideo] FFmpeg: {' '.join(cmd)}")

            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=False)

            try:
                if len(frames) == 1 and total_frames > 1:
                    buf = frames[0].tobytes()
                    step = max(1, total_frames // 50)
                    for i in range(total_frames):
                        proc.stdin.write(buf)
                        if show_progress and (i + 1) % step == 0:
                            pct = 100.0 * (i + 1) / total_frames
                            print(f"[SaveVideo] {i+1}/{total_frames} ({pct:5.1f}%)")
                else:
                    total = len(frames)
                    step = max(1, total // 50)
                    for i, f in enumerate(frames, 1):
                        proc.stdin.write(f.tobytes())
                        if show_progress and (i % step == 0):
                            pct = 100.0 * i / total
                            print(f"[SaveVideo] {i}/{total} ({pct:5.1f}%)")
            finally:
                try:
                    if proc.stdin:
                        proc.stdin.flush()
                except Exception:
                    pass
                if proc.stdin:
                    proc.stdin.close()
                stderr_text = ""
                stderr_bytes = b""
                if proc.stderr:
                    try:
                        stderr_bytes = proc.stderr.read()
                    except Exception:
                        stderr_bytes = b""
                try:
                    stderr_text = stderr_bytes.decode("utf-8", errors="ignore") if isinstance(stderr_bytes, (bytes, bytearray)) else str(stderr_bytes)
                except Exception:
                    stderr_text = ""
                ret = proc.wait()

            out_exists = out_path.exists()
            out_size = out_path.stat().st_size if out_exists else 0
            success = (ret == 0) and out_exists and out_size > 0

            if tmp_wav and success:
                try:
                    os.unlink(tmp_wav)
                except Exception:
                    pass

            if not success:
                msg = (
                    f"FFmpeg failed (code {ret}). Output exists={out_exists} size={out_size}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stderr:\n{stderr_text.strip()}"
                )
                raise RuntimeError(msg)

            try:
                video_path_str = str(out_path.resolve())
            except Exception:
                video_path_str = str(out_path)

            if show_progress:
                print(f"[SaveVideo] Done: {video_path_str} ({out_size} bytes)")
                try:
                    entries = sorted([p.name for p in Path(video_dir).iterdir()])
                    preview_list = ', '.join(entries[:20])
                    more = '' if len(entries) <= 20 else f" (+{len(entries)-20} more)"
                    print(f"[SaveVideo] Dir list ({video_dir}): {preview_list}{more}")
                except Exception:
                    pass
        else:
            if show_progress:
                try:
                    print(f"[SaveVideo] Output base: {base.resolve()}")
                except Exception:
                    print(f"[SaveVideo] Output base: {base}")
                print("[SaveVideo] Save mode: frames only")

        def _parse_frame_selection(spec: str, total: int) -> List[int]:
            s = (spec or "").strip()
            if not s:
                return []
            if s == "-1":
                return list(range(total))
            if s == "-2":
                return [max(0, total - 1)] if total > 0 else []
            try:
                parts = [x.strip() for x in s.split(',') if x.strip()]
                idxs: List[int] = []
                for p in parts:
                    v = int(p)
                    if v < 0:
                        if v == -1:
                            return list(range(total))
                        if v == -2 and total > 0:
                            idxs.append(total - 1)
                    elif v < total:
                        idxs.append(v)
                return sorted(set(idxs))
            except Exception:
                return []

        frames_dir_effective = frames_dir if isinstance(frames_dir, str) else ""
        if save_frames and not frames_dir_effective.strip():
            frames_dir_effective = "frames"

        frames_saved_path = None
        if save_frames and isinstance(frames_dir_effective, str) and frames_dir_effective.strip():
            idxs = _parse_frame_selection(frames_select, len(frames))
            if idxs:
                frames_path = Path(frames_dir_effective.strip()).expanduser()
                if frames_path.is_absolute():
                    target_folder = self._normalize_path(frames_path)
                else:
                    target_folder = self._normalize_path(video_dir / frames_path)

                self._validate_target_dir(target_folder)
                target_folder.mkdir(parents=True, exist_ok=True)

                if show_progress:
                    print(f"[SaveVideo] Saving frames -> {target_folder} | select='{frames_select}' -> {len(idxs)} frames")

                try:
                    import imageio.v2 as imageio  # type: ignore
                except Exception:
                    imageio = None  # type: ignore

                for idx in idxs:
                    a = frames[idx]
                    if a.dtype != np.uint8:
                        a = np.clip(a, 0, 1)
                        a = (a * 255.0).round().astype(np.uint8)
                    if a.shape[2] == 4:
                        a = a[:, :, :3]
                    fname = f"{stem}_frame_{idx:04d}.png"
                    fpath = target_folder / fname
                    try:
                        if imageio is not None:
                            imageio.imwrite(str(fpath), a)
                    except Exception:
                        pass
                frames_saved_path = target_folder
            elif show_progress and save_frames:
                print(f"[SaveVideo] Frame export skipped (selection '{frames_select}').")

        ui_seq_images = []
        imageio = None
        if show_preview:
            try:
                import imageio.v2 as imageio  # type: ignore
            except Exception:
                imageio = None  # type: ignore

            if imageio is not None and len(frames) > 0 and preview_max_frames > 0:
                try:
                    from folder_paths import get_temp_directory  # type: ignore
                    seq_root = Path(get_temp_directory())
                except Exception:
                    seq_root = video_dir

                step = max(1, int(preview_step))
                idxs = list(range(0, len(frames), step))
                if len(frames) == 1:
                    idxs = [0]
                if preview_max_frames > 0:
                    idxs = idxs[:preview_max_frames]

                for i, idx in enumerate(idxs, 1):
                    a = frames[idx if idx < len(frames) else -1]
                    if a.dtype != np.uint8:
                        a = np.clip(a, 0, 1)
                        a = (a * 255.0).round().astype(np.uint8)
                    if a.shape[2] == 4:
                        a = a[:, :, :3]
                    fname = f"{stem}_seq_{i:04d}.png"
                    fpath = seq_root / fname
                    try:
                        imageio.imwrite(str(fpath), a)
                    except Exception:
                        continue
                    ui_seq_images.append({
                        "filename": fname,
                        "subfolder": "",
                        "type": "temp",
                    })

        abs_path = ""
        if video_path_str:
            abs_path = video_path_str
        elif frames_saved_path is not None:
            try:
                abs_path = str(frames_saved_path.resolve())
            except Exception:
                abs_path = str(frames_saved_path)

        if not abs_path:
            try:
                abs_path = str(base.resolve())
            except Exception:
                abs_path = str(base)

        ui = {"text": abs_path}
        if ui_seq_images:
            ui["images"] = ui_seq_images

        return {"ui": ui, "result": (images, abs_path,)}

NODE_CLASS_MAPPINGS = {
    "SaveVideoDehypnotic": SaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideoDehypnotic": "Save Video & Frames (Dehypnotic)",
}



