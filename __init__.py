NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from .save_audio_mp3 import (  # noqa: E402
    SaveAudioMP3Enhanced,
    NODE_CLASS_MAPPINGS as AUDIO_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as AUDIO_DISPLAY_NAME_MAPPINGS,
)

NODE_CLASS_MAPPINGS.update(AUDIO_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(AUDIO_DISPLAY_NAME_MAPPINGS)
__all__.append("SaveAudioMP3Enhanced")

try:
    from .save_images import (  # noqa: E402
        SaveImages,
        NODE_CLASS_MAPPINGS as IMAGE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as IMAGE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as exc:
    print(f"[comfyui-save-mp3] Failed to import image node: {exc}")
else:
    NODE_CLASS_MAPPINGS.update(IMAGE_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(IMAGE_DISPLAY_NAME_MAPPINGS)
    __all__.append("SaveImages")

try:
    from .save_video import (  # noqa: E402
        SaveVideo,
        NODE_CLASS_MAPPINGS as VIDEO_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as VIDEO_DISPLAY_NAME_MAPPINGS,
    )
except Exception as exc:
    print(f"[comfyui-save-mp3] Failed to import video node: {exc}")
else:
    NODE_CLASS_MAPPINGS.update(VIDEO_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(VIDEO_DISPLAY_NAME_MAPPINGS)
    __all__.append("SaveVideo")

