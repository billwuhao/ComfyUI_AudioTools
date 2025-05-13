from .audiotoolsnode import *
from .minimal_pause_node import MinimalPauseNode
from .AddSubtitlesToVideo import AddSubtitlesToTensor
from .MWAudioRecorderAT import AudioRecorderAT
from .stringnodes import StringDisplayNode, StringPauseGateNode

NODE_CLASS_MAPPINGS = {
    "SaveAudioMW": SaveAudioMW,
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
    "StringDisplayNode": StringDisplayNode,
    "StringPauseGateNode": StringPauseGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudioMW": "Save Audio @MW",
    "LoadAudioMW": "Load Audio @MW",
    "MinimalPauseNode": "Pause Node",
    "AudioConcatenate": "Audio Concatenate",
    "AudioAddWatermark": "Audio Watermark Embedding",
    "AdjustAudio": "Adjust Audio",
    "TrimAudio": "Trim Audio",
    "RemoveSilence": "Remove Silence",
    # "AudioDenoising": "Audio Denoising",
    "AudioRecorderAT": "MW Audio Recorder",
    "AddSubtitlesToVideo": "Add Subtitles To Video",
    "MultiLinePromptAT": "Multi-Line Prompt",
    "StringDisplayNode": "📺 String Viewer",
    "StringPauseGateNode": "⏸ String Pause Gate",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"
