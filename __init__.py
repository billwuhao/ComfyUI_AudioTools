from .audiotoolsnode import *
from .minimal_pause_node import MinimalPauseNode
from .AddSubtitlesToVideo import AddSubtitlesToTensor
from .MWAudioRecorderAT import AudioRecorderAT
from .clearvoicenode import ClearVoiceRun

NODE_CLASS_MAPPINGS = {
    "ClearVoiceRun": ClearVoiceRun,
    "LoadAudioMW": LoadAudioMW,
    "MinimalPauseNode": MinimalPauseNode,
    "AudioConcatenate": AudioConcatenate,
    "AudioAddWatermark": AudioAddWatermark,
    "AdjustAudio": AdjustAudio,
    "TrimAudio": TrimAudio,
    "RemoveSilence": RemoveSilence,
    # "AudioDenoising": AudioDenoising,
    "AudioRecorderAT": AudioRecorderAT,
    "AddSubtitlesToVideo": AddSubtitlesToTensor,
    "MultiLinePromptAT": MultiLinePromptAT,
    "StringEditNode": StringEditNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearVoiceRun": "Clear Voice @MW",
    "LoadAudioMW": "Load Audio @MW",
    "MinimalPauseNode": "Pause Node @MW",
    "AudioConcatenate": "Audio Concatenate",
    "AudioAddWatermark": "Audio Watermark Embedding",
    "AdjustAudio": "Adjust Audio",
    "TrimAudio": "Trim Audio",
    "RemoveSilence": "Remove Silence",
    # "AudioDenoising": "Audio Denoising",
    "AudioRecorderAT": "MW Audio Recorder",
    "AddSubtitlesToVideo": "Add Subtitles To Video",
    "MultiLinePromptAT": "Multi-Line Prompt",
    "StringEditNode": "String Edit",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"
