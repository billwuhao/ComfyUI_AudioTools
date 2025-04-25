from .audiotoolsnode import *
from .string_editor_node import *

from .AddSubtitlesToVideo import AddSubtitlesToTensor
from .MWAudioRecorderAT import AudioRecorderAT

NODE_CLASS_MAPPINGS = {
    "StringEditorPersistentTempFileNode": StringEditorPersistentTempFileNode,
    "AudioConcatenate": AudioConcatenate,
    "AudioAddWatermark": AudioAddWatermark,
    "AdjustAudio": AdjustAudio,
    "TrimAudio": TrimAudio,
    "RemoveSilence": RemoveSilence,
    # "AudioDenoising": AudioDenoising,
    "AudioRecorderAT": AudioRecorderAT,
    "AddSubtitlesToVideo": AddSubtitlesToTensor,
    "MultiLinePromptAT": MultiLinePromptAT,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StringEditorPersistentTempFileNode": "Show or Edit String",
    "AudioConcatenate": "Audio Concatenate",
    "AudioAddWatermark": "Audio Watermark Embedding",
    "AdjustAudio": "Adjust Audio",
    "TrimAudio": "Trim Audio",
    "RemoveSilence": "Remove Silence",
    # "AudioDenoising": "Audio Denoising",
    "AudioRecorderAT": "MW Audio Recorder",
    "AddSubtitlesToVideo": "Add Subtitles To Video",
    "MultiLinePromptAT": "Multi-Line Prompt",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

EXTENSION_NAME = "StringEditor"

WEB_DIRECTORY = "./web"
