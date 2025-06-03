from easydict import EasyDict
import folder_paths
import os
import torch
import tempfile
import torchaudio
from typing import Optional

from .clearvoice import MF2SE

models_dir = folder_paths.models_dir
checkpoint_dir = os.path.join(models_dir, "TTS", 'MossFormer2_SE_48K')
cache_dir = folder_paths.get_temp_directory()

def statistical_compare(tensor1, tensor2):
    """通过统计特征快速比较"""
    stats1 = {
        'mean': tensor1.mean(),
        'std': tensor1.std(),
        'max': tensor1.max(),
        'min': tensor1.min()
    }
    stats2 = {
        'mean': tensor2.mean(),
        'std': tensor2.std(),
        'max': tensor2.max(),
        'min': tensor2.min()
    }
    return all(torch.allclose(stats1[k], stats2[k], rtol=1e-3) for k in stats1)

def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")

MF2SE_MODEL = None
class ClearVoiceRun:
    def __init__(self):
        self.audio_tensor = None
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run_inference"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def run_inference(self, audio, unload_model=False):
        args = {
            'checkpoint_dir': checkpoint_dir,
            'output_dir': 'output',
            'input_path': 'data/input',
            'task': 'speech_enhancement',
            'use_cuda': 1,
            'num_gpu': 1,
            'sampling_rate': 48000,
            'network': "MossFormer2_SE_48K",
            'one_time_decode_length': 20,
            'decode_window': 4,
            'win_type': 'hamming',
            'win_len': 1920,
            'win_inc': 384,
            'fft_len': 1920,
            'num_mels': 60,
        }

        audio_data = audio["waveform"].squeeze(0)
        sr = audio["sample_rate"]
        audio_path = cache_audio_tensor(
            cache_dir,
            audio_data,
            sr,
        )
        args["input_path"] = audio_path
        _args = EasyDict(args)

        global MF2SE_MODEL
        if MF2SE_MODEL is None or self.audio_tensor is None or not statistical_compare(audio_data, self.audio_tensor):
            MF2SE_MODEL = MF2SE(_args)

        audio_output = MF2SE_MODEL.process()
        audio_output = torch.from_numpy(audio_output)
        audio_output = audio_output.unsqueeze(0)

        if unload_model:
            MF2SE_MODEL.clean()
            MF2SE_MODEL = None
            self.audio_tensor = None
            torch.cuda.empty_cache()

        return ({"waveform": audio_output, "sample_rate": 48000},)


NODE_CLASS_MAPPINGS = {
    "ClearVoiceRun": ClearVoiceRun,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearVoiceRun": "Clear Voice",
}