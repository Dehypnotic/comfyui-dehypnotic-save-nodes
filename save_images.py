# save_image_multiformat.py
# ComfyUI node: Image Save (Multi-format)
#
# - Lagrer PNG / JPG / WEBP / GIF / BMP / TIFF via Pillow (som følger med ComfyUI)
# - Sekvensiell filnavngiving med padding og startverdi
# - Valgfri datobasert undermappe (strftime-mønster)
# - Kvalitet/optimering, WEBP lossless, DPI
# - Overwrite modes: increment / replace / skip
# - Embed workflow:
#     * PNG  -> tEXt/iTXt "workflow"
#     * WEBP -> XMP (comfy:Workflow)
# - (Ny) Embed PNG preview-thumbnail:
#     * PNG  -> iTXt "thumbnail" med data:image/png;base64,<...>
# - Returnerer input-bildet (passthrough) og lagrede filstier (linjedelt streng)

import os
import re
import json
import typing as _t
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import base64
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#

VALID_EXTS = ["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"]

# Vanlige dato-mønstre for undermapper (siste tom streng = ingen dato-undermappe)
COMMON_DATE_PATTERNS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m",
    "%Y/%m",
    "%Y%m%d",
    "%Y%m",
    "%Y-%m-%d_%H-%M-%S",
    "%Y/%m/%d/%H-%M-%S",
    "",
]


def to_pil(img: np.ndarray) -> Image.Image:
    """
    ComfyUI IMAGE er typisk float32 [0,1] (H,W,C). Konverter trygt til uint8 PIL.
    Beholder alfa hvis tilstede.
    """
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 1.0)
        a = (a * 255.0).round().astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        mode = "RGBA"
    else:
        mode = "RGB"
    return Image.fromarray(a, mode=mode)


def next_seq_number(folder: Path, prefix: str, delim: str, padding: int) -> int:
    """
    Scanner mappa etter filer som matcher prefix_delimNNNN og returnerer neste nummer.
    Fungerer på tvers av restarter (ingen globale tellere).
    """
    pattern = re.compile(rf"^{re.escape(prefix)}{re.escape(delim)}(\d{{{padding}}})\b")
    max_num = 0
    if folder.exists():
        for p in folder.iterdir():
            if not p.is_file():
                continue
            m = pattern.match(p.stem)
            if m:
                try:
                    num = int(m.group(1))
                    if num > max_num:
                        max_num = num
                except ValueError:
                    pass
    return max_num + 1


def _make_webp_xmp(workflow_json: str) -> bytes:
    """
    Minimal XMP-pakke for WEBP. Pillow kan lagre denne via xmp= parameter.
    """
    from xml.sax.saxutils import escape

    payload = escape(workflow_json)
    xmp = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:comfy="https://comfy.org/ns/1.0/">
   <comfy:Workflow>{payload}</comfy:Workflow>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    return xmp.encode("utf-8")


def _make_png_thumbnail_text(pil_img: Image.Image, max_size: int = 256) -> str:
    """
    Lager en liten PNG-thumbnail (maks bredde/høyde = max_size) og returnerer
    en data-URL tekst: "data:image/png;base64,<...>" egnet for iTXt.
    """
    if max_size <= 0:
        max_size = 256
    thumb = pil_img.copy()
    # Bevar alfa; PIL\'s thumbnail gjør in-place resize m/ antialias
    thumb.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = BytesIO()
    # Kompakt PNG for thumbnail
    thumb.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def save_single_image(
    pil_img: Image.Image,
    path: Path,
    ext: str,
    quality: int,
    optimize: bool,
    lossless_webp: bool,
    dpi: int,
    embed_workflow: bool,
    workflow_data: Optional[str],
    embed_thumbnail: bool,
    thumbnail_max_size: int,
) -> None:
    """
    Lagrer ett PIL-bilde til ønsket format med relevante parametre og ev. metadata.
    """
    ext_l = ext.lower()
    save_kwargs = {}

    if dpi and dpi > 0:
        # De fleste formater bruker (x_dpi, y_dpi)
        save_kwargs["dpi"] = (dpi, dpi)

    if ext_l in ("jpg", "jpeg"):
        save_kwargs["quality"] = int(quality)
        save_kwargs["optimize"] = bool(optimize)
        pil_img = pil_img.convert("RGB")  # JPEG støtter ikke alfa

    elif ext_l == "png":
        save_kwargs["optimize"] = bool(optimize)

    elif ext_l == "webp":
        if lossless_webp:
            save_kwargs["lossless"] = True
        else:
            save_kwargs["quality"] = int(quality)

    # --- Metadata-innstøping ------------------------------------------------- 
    if ext_l == "png":
        # Bygg PngInfo med valgfri workflow + thumbnail
        try:
            from PIL import PngImagePlugin

            meta = PngImagePlugin.PngInfo()
            if embed_workflow and workflow_data:
                meta.add_text("workflow", workflow_data)
            if embed_thumbnail:
                try:
                    thumb_text = _make_png_thumbnail_text(pil_img, max_size=thumbnail_max_size)
                    # Bruk iTXt via add_text (Pillow velger iTXt når unicode/len>?)
                    meta.add_text("thumbnail", thumb_text)
                except Exception:
                    pass  # thumbnail er bare et ekstra
            pil_img.save(path, format="PNG", pnginfo=meta, **save_kwargs)
            return
        except Exception:
            # Faller tilbake til vanlig lagring dersom noe feiler
            pass

    if ext_l == "webp" and embed_workflow and workflow_data:
        # Pillow (>=9.1 ca.) støtter xmp= for WEBP. Prøv det, og fall tilbake ved TypeError.
        try:
            xmp_bytes = _make_webp_xmp(workflow_data)
            pil_img.save(path, format="WEBP", xmp=xmp_bytes, **save_kwargs)
            return
        except TypeError:
            # Eldre Pillow uten xmp-argument – fallet ned til standard lagring.
            pass

    # Standard lagring uten ekstra metadata
    save_format = ext_l.upper()
    if save_format == "JPG":
        save_format = "JPEG"
    pil_img.save(path, format=save_format, **save_kwargs)


# -----------------------------------------------------------------------------#
# Node
# -----------------------------------------------------------------------------#

class SaveImages:
    """
    Lagrer bilder til disk i flere formater med sekvensiell navngiving.
    Kan embedde workflow (PNG/WEBP) og PNG preview-thumbnail.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_dir": ("STRING", {"default": ""}),
                # Hvis satt, lagres i output_dir / strftime(mønster)
                "date_subfolder_pattern": (tuple(COMMON_DATE_PATTERNS), {"default": "%Y-%m-%d"}),
                "filename_prefix": ("STRING", {"default": "QIE"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "number_padding": ("INT", {"default": 4, "min": 1, "max": 10}),
                "number_start": ("INT", {"default": 1, "min": 0, "max": 1_000_000}),
                "extension": (tuple(VALID_EXTS), {"default": "png"}),
                "quality": ("INT", {"default": 100, "min": 1, "max": 100}),
                "optimize_image": ("BOOLEAN", {"default": True}),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "dpi": ("INT", {"default": 300, "min": 1, "max": 1200}),
                "overwrite_mode": (("increment", "replace", "skip"), {"default": "increment"}),
                "embed_workflow": ("BOOLEAN", {"default": False}),
                "embed_thumbnail": ("BOOLEAN", {"default": True}),
                "thumbnail_max_size": ("INT", {"default": 256, "min": 32, "max": 1024}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "saved_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image/io"

    # --- Whitelist Path Logic (ported from audio node) ---

    def _get_comfy_dir(self, name: str) -> _t.Optional[str]:
        try:
            import folder_paths
            if hasattr(folder_paths, f"get_{name}_directory"):
                return getattr(folder_paths, f"get_{name}_directory")()
            # Fallback for older ComfyUI versions
            if name in folder_paths.folder_names_and_paths:
                return folder_paths.folder_names_and_paths[name][0]
        except Exception:
            pass
        return None

    def _get_comfy_root(self) -> str:
        comfy_output = self._get_comfy_dir("output")
        if comfy_output:
            return os.path.abspath(os.path.join(comfy_output, os.pardir))
        # Fallback if everything else fails
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def _resolve_out_dir(self, path: str) -> str:
        # If path is empty or whitespace, default to the main output directory.
        if not path or not path.strip():
            return self._get_comfy_dir("output")

        # Absolute paths are used as-is.
        if os.path.isabs(path):
            return path

        # If the user writes "output/something", resolve it from the Comfy root
        # to prevent "output/output/something".
        comfy_root = self._get_comfy_root()
        path_norm = path.replace("/", os.sep).replace("\\", os.sep)
        if path_norm.startswith("output" + os.sep):
            return os.path.join(comfy_root, path)

        # For any other relative path (e.g., "MySubfolder"),
        # resolve it relative to the main output directory.
        base_output = self._get_comfy_dir("output")
        return os.path.join(base_output, path)

    def _load_allowed_roots(self) -> _t.List[str]:
        """Load external save roots from a shared JSON file or env var."""
        env_cfg = os.environ.get("DEHYPNOTIC_SAVE_ALLOWED_PATHS")
        candidates = []
        if env_cfg and os.path.isfile(env_cfg):
            candidates.append(env_cfg)

        comfy_root = self._get_comfy_root()
        global_names = (
            "dehypnotic_save_allowed_paths.json",
            "allowed_paths.json",
        )
        # Search order: user config -> comfy root -> node folder
        for name in global_names:
            candidates.append(os.path.join(comfy_root, "user", "config", name))
            candidates.append(os.path.join(comfy_root, "user", name))
            candidates.append(os.path.join(comfy_root, "config", name))
            candidates.append(os.path.join(comfy_root, name))
            candidates.append(os.path.join(os.path.dirname(__file__), name))

        for path in candidates:
            if not path or not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    # Allow comments in JSON file
                    raw = "".join(line for line in f if not line.strip().startswith(("//", "#")))
                    data = json.loads(raw)
                
                roots = data.get("allowed_roots", []) if isinstance(data, dict) else []
                if isinstance(roots, list):
                    norm_roots = [os.path.abspath(os.path.expandvars(r)) for r in roots if isinstance(r, str)]
                    if norm_roots:
                        return norm_roots
            except Exception:
                pass
        return []

    def _is_under_dir(self, path: str, base: str) -> bool:
        if not base: return False
        try:
            ap = os.path.abspath(path)
            bb = os.path.abspath(base)
            return os.path.commonpath([ap, bb]) == bb
        except Exception:
            return False

    def _validate_target_dir(self, target_dir: str) -> None:
        """Raise PermissionError if target_dir is not in a permitted location."""
        abs_target = os.path.abspath(target_dir)

        # 1. Always allow ComfyUI's output and temp directories
        comfy_output = self._get_comfy_dir("output")
        if self._is_under_dir(abs_target, comfy_output):
            return

        comfy_temp = self._get_comfy_dir("temp")
        if self._is_under_dir(abs_target, comfy_temp):
            return

        # 2. Allow whitelisted paths
        allowed_roots = self._load_allowed_roots()
        for root in allowed_roots:
            if self._is_under_dir(abs_target, root):
                return

        # 3. If no match, deny access
        msg = (
            "External save path is not allowed.\\n"
            f'(Attempted to write to: "{abs_target}")\\n\\n'
            "This node only writes inside ComfyUI's `output` or `temp` directories, "
            "unless the path is whitelisted offline.\\n\\n"
            "To allow external locations, create/edit a JSON file named "
            "\'dehypnotic_save_allowed_paths.json\' in your ComfyUI root (or user/config) folder "
            "with content like:\\n\\n"
            '{\\n  "allowed_roots": ["D:/ImageExports", "E:/TeamShare/Assets"]\\n}\\n'
        )
        raise PermissionError(msg)

    # Merk: Det er ikke garantert at ComfyUI injiserer workflow-JSON på self,
    # men enkelte bygg gjør det. Vi prøver flere navn + miljøvariabel som fallback.
    def _get_workflow_json(self) -> Optional[str]:
        for attr in ("workflow", "workflow_json", "workflow_str"):
            if hasattr(self, attr):
                v = getattr(self, attr)
                if isinstance(v, str) and v.strip():
                    return v
        env_v = os.environ.get("COMFY_WORKFLOW_JSON")
        if env_v and env_v.strip():
            return env_v
        return None

    def save(
        self,
        images,
        filename_prefix,
        filename_delimiter,
        number_padding,
        number_start,
        extension,
        quality,
        optimize_image,
        lossless_webp,
        dpi,
        overwrite_mode,
        embed_workflow,
        embed_thumbnail,
        thumbnail_max_size,
        output_dir="",
        date_subfolder_pattern="",
    ):
        # Bygg målkatalog (+ valgfri datoundermappe)
        resolved_main_dir = self._resolve_out_dir(output_dir)
        self._validate_target_dir(resolved_main_dir)

        base = Path(resolved_main_dir)
        if date_subfolder_pattern.strip():
            base = base / datetime.now().strftime(date_subfolder_pattern.strip())
        base.mkdir(parents=True, exist_ok=True)

        # Finn start-sekvens
        if overwrite_mode == "increment":
            seq = max(number_start, next_seq_number(base, filename_prefix, filename_delimiter, number_padding))
        else:
            seq = max(1, number_start)

        # Hent workflow JSON hvis vi skal embedde
        workflow_json = self._get_workflow_json() if embed_workflow else None

        saved_paths: List[str] = []

        # Process each image in the batch
        for image_tensor in images:
            pil_img = to_pil(image_tensor.cpu().numpy())

            stem = f"{filename_prefix}{filename_delimiter}{seq:0{number_padding}d}"
            filename = f"{stem}.{extension.lower()}"
            path = base / filename

            # Håndter overwrite-modus
            if path.exists():
                if overwrite_mode == "replace":
                    pass
                elif overwrite_mode == "skip":
                    saved_paths.append(str(path))
                    seq += 1
                    continue
                elif overwrite_mode == "increment":
                    while path.exists():
                        seq += 1
                        stem = f"{filename_prefix}{filename_delimiter}{seq:0{number_padding}d}"
                        filename = f"{stem}.{extension.lower()}"
                        path = base / filename

            save_single_image(
                pil_img=pil_img,
                path=path,
                ext=extension,
                quality=quality,
                optimize=optimize_image,
                lossless_webp=lossless_webp,
                dpi=dpi,
                embed_workflow=embed_workflow,
                workflow_data=workflow_json,
                embed_thumbnail=embed_thumbnail,
                thumbnail_max_size=thumbnail_max_size,
            )

            saved_paths.append(str(path))
            seq += 1

        # Returner originalt bilde (for chaining) + paths
        return (images, "\\n".join(saved_paths),)


# Registrering i ComfyUI
NODE_CLASS_MAPPINGS = {
    "SaveImagesDehypnotic": SaveImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImagesDehypnotic": "Save Images (Dehypnotic)",
}
