import sox
import tempfile
import os
import torchaudio
import librosa
import torch
import ast
import silentcipher
import folder_paths
from typing import List, Union, Optional


models_dir = folder_paths.models_dir
input_dir = folder_paths.get_input_directory()

def get_path():
    from pathlib import Path
    import yaml

    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(script_dir, "extra_help_file.yaml")
    try:

        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            audios_dir = data["audios_dir"]
            audios_dir = Path(audios_dir)
            if not os.path.exists(audios_dir):
                raise Exception(f"Customize audios loading path: {audios_dir} not exists.")

            print(f"Customize audios loading path: {audios_dir}")
            return audios_dir
    except FileNotFoundError:
        print(f"Error: File not found - extra_help_file.yaml")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
    except KeyError:
        print(f"Error: Missing key 'audios_dir' in YAML file.")
    except Exception as e:
        print(f"Unexpected error: {e}")

    print("No customize audios loading path found, use default path.")
    return input_dir

def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径

    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}

    root_dir = os.path.normpath(root_dir)

    for dirpath, dirnames, filenames in os.walk(root_dir):

        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        current_files = []
        for filename in filenames:

            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue

            full_path = os.path.join(dirpath, filename)

            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)

            current_files.append(full_path)

        if return_type == "dict":

            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)

    return file_dict if return_type == "dict" else file_paths


class LoadAudioMW:
    audios_dir = get_path()
    files = get_all_files(audios_dir, extensions=[".wav", ".mp3", ".flac", ".mp4", ".WAV", ".MP3", ".FLAC", ".MP4"], relative_path=True)

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": (sorted(s.files),),
                    "start_time_sec": ("FLOAT", {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10000.0, 
                        "step": 0.01,
                        "display": "number", 
                        "tooltip": "Start time of the slice in seconds"
                        }),
                    "duration_sec": ("FLOAT", {
                        "default": 5.0,
                        "min": 0.01, 
                        "max": 10000.0, 
                        "step": 0.01,
                        "display": "number",
                        "tooltip": "Desired duration of the slice in seconds"
                        }),
                    },
                    "optional": {
                        "fps": ("FLOAT", {
                            "default": 24.0,
                            "min": 1.0,
                            "max": 60.0,
                            "step": 1.0,
                        })
                    }
                }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("AUDIO", "frame_count")
    FUNCTION = "load"

    def load(self, audio, start_time_sec=0, duration_sec=5, fps=24.0):
        audio_path = os.path.join(LoadAudioMW.audios_dir, audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.unsqueeze(0)

        if waveform is None or waveform.numel() == 0 or sample_rate <= 0:
            raise Exception("SliceAudio received invalid input audio.")

        start_time_sec = max(0.0, start_time_sec)
        duration_sec = max(0.01, duration_sec) 

        start_sample = round(start_time_sec * sample_rate)
        num_samples_requested = round(duration_sec * sample_rate)

        total_samples = waveform.shape[-1] 

        start_sample = max(0, min(start_sample, total_samples))

        actual_end_sample = min(start_sample + num_samples_requested, total_samples)

        if start_sample >= actual_end_sample:
            raise ValueError(f"SliceAudio requested slice starts at or after the calculated end ({start_sample} >= {actual_end_sample}).")
        else:
            sliced_waveform = waveform[..., start_sample:actual_end_sample]

        output_audio = {
            "waveform": sliced_waveform,
            "sample_rate": sample_rate
        }

        return {"result": (output_audio, int(fps * duration_sec) + 1)}

class AudioConcatenate:
    """
    ComfyUI node to concatenate two audio inputs with smart channel handling.

    Inputs are expected in the format {"waveform": torch.Tensor, "sample_rate": int}.
    The node concatenates waveform_a followed by waveform_b.
    If sample rates differ, both waveforms are resampled to the maximum of the two rates.
    Channel Handling Logic:
    - If both inputs are mono (1 channel), output is mono.
    - Otherwise (if either input is > 1 channel), output is stereo (2 channels).
    - Mono (1) to Stereo (2) conversion is done by duplicating the channel.
    - Multi-channel (> 2) to Stereo (2) conversion is done by a simple averaging downmix
      (average all channels to mono, then duplicate for stereo). A warning is printed
      as this is a basic downmix method.

    Output is in the same {"waveform": torch.Tensor, "sample_rate": int} format,
    with the higher sample rate and the determined output channel count (1 or 2).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_a": ("AUDIO",),
                "audio_b": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concatenate"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def concatenate(self, audio_a, audio_b):
        """
        Concatenates two audio waveforms after resampling and smart channel adjustment.

        Args:
            audio_a (dict): First audio input {"waveform": tensor, "sample_rate": int}.
            audio_b (dict): Second audio input {"waveform": tensor, "sample_rate": int}.

        Returns:
            tuple: A tuple containing the output audio dictionary.
                   ({"waveform": concatenated_tensor, "sample_rate": max_sr},)
        """
        waveform_a = audio_a["waveform"]
        sr_a = audio_a["sample_rate"]
        waveform_b = audio_b["waveform"]
        sr_b = audio_b["sample_rate"]

        final_sr = max(sr_a, sr_b)

        resampled_waveform_a = waveform_a
        if sr_a != final_sr:
            print(f"Concatenate Audio Node: Resampling audio A from {sr_a} Hz to {final_sr} Hz")
            resample_a = torchaudio.transforms.Resample(orig_freq=sr_a, new_freq=final_sr).to(waveform_a.device)
            resampled_waveform_a = resample_a(waveform_a)

        resampled_waveform_b = waveform_b
        if sr_b != final_sr:
            print(f"Concatenate Audio Node: Resampling audio B from {sr_b} Hz to {final_sr} Hz")
            resample_b = torchaudio.transforms.Resample(orig_freq=sr_b, new_freq=final_sr).to(waveform_b.device)
            resampled_waveform_b = resample_b(waveform_b)

        channels_a = resampled_waveform_a.shape[1]
        channels_b = resampled_waveform_b.shape[1]

        if channels_a == 1 and channels_b == 1:
            target_channels = 1 
            print("Concatenate Audio Node: Both inputs are mono, output will be mono (1 channel).")
        else:
            target_channels = 2 
            print(f"Concatenate Audio Node: At least one input is not mono ({channels_a} vs {channels_b}), output will be stereo (2 channels).")

        def adjust_channels(wf, current_channels, target_channels, name):
            if current_channels == target_channels:
                return wf
            elif target_channels == 1 and current_channels > 1:

                 print(f"Concatenate Audio Node Warning: Attempting to downmix {name} from {current_channels} to {target_channels} (mono). Simple average downmix applied.")

                 return wf.mean(dim=1, keepdim=True)
            elif target_channels == 2:
                if current_channels == 1:

                    print(f"Concatenate Audio Node: Converting {name} from {current_channels} to {target_channels} channels (mono to stereo).")
                    return wf.repeat(1, target_channels, 1) 
                elif current_channels > 2:

                    print(f"Concatenate Audio Node Warning: Converting {name} from {current_channels} to {target_channels} channels (multi-channel to stereo). Applying simple average downmix.")

                    mono_wf = wf.mean(dim=1, keepdim=True)
                    return mono_wf.repeat(1, target_channels, 1)

            else:

                 raise RuntimeError(f"Concatenate Audio Node: Unsupported channel adjustment requested for {name}: from {current_channels} to {target_channels}.")

        adjusted_waveform_a = adjust_channels(resampled_waveform_a, channels_a, target_channels, "Audio A")
        adjusted_waveform_b = adjust_channels(resampled_waveform_b, channels_b, target_channels, "Audio B")

        concatenated_waveform = torch.cat((adjusted_waveform_a, adjusted_waveform_b), dim=2)

        output_audio = {
            "waveform": concatenated_waveform,
            "sample_rate": final_sr 
        }

        return (output_audio,)

MODEL_CACHE = None
class AudioAddWatermark:
    def __init__(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "add_watermark": ("BOOLEAN", {
                        "default": False, 
                        "tooltip": "Enable audio watermark embedding"
                    }),
                    "key": ("STRING", {
                        "default": "[212, 211, 146, 56, 201]", 
                        "tooltip": "Encryption key as list of integers (e.g. [212,211,146,56,201])"
                    }),
                    "unload_model": ("BOOLEAN", {
                        "default": True,
                        "tooltip": "Unload model from memory after use"
                    })
                    }

                }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "watermark")
    FUNCTION = "watermarkgen"

    def watermarkgen(self, audio, add_watermark, key, unload_model):
        """Main watermark processing pipeline"""
        ckpt_path = os.path.join(models_dir, "TTS", "SilentCipher", "44_1_khz", "73999_iteration")
        os.makedirs(ckpt_path, exist_ok=True)
        config_path = os.path.join(models_dir, ckpt_path, "hparams.yaml")
        global MODEL_CACHE
        if MODEL_CACHE is None:
            MODEL_CACHE = silentcipher.get_model(
                model_type="44.1k", 
                ckpt_path=ckpt_path, 
                config_path=config_path,
                device=self.device,
            )

        watermarker = MODEL_CACHE
        audio_array, sample_rate = self.load_audio(audio)

        audio_array = audio_array.to(self.device)

        if add_watermark:
            key = self._parse_key(key)
            audio_array, sample_rate = self.watermark(watermarker, audio_array, sample_rate, key)

        watermark = self.verify(watermarker, audio_array, sample_rate)
        if unload_model:
            del watermarker
            MODEL_CACHE = None
            torch.cuda.empty_cache()

        return ({"waveform": audio_array.unsqueeze(0).unsqueeze(0).cpu(), "sample_rate": sample_rate}, watermark)

    @torch.inference_mode()
    def watermark(self,
        watermarker: silentcipher.server.Model,
        audio_array: torch.Tensor,
        sample_rate: int,
        watermark_key: list[int],
    ) -> tuple[torch.Tensor, int]:

        if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
            audio_array = audio_array.mean(dim=0)

        audio_array = audio_array.to(self.device)

        audio_array_44khz = torchaudio.functional.resample(
            audio_array, 
            orig_freq=sample_rate, 
            new_freq=44100
        ).to(self.device)

        if len(audio_array_44khz.shape) != 1:
            audio_array_44khz = audio_array_44khz.reshape(-1)

        try:

            encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=30)

            verify_result = watermarker.decode_wav(encoded, 44100, phase_shift_decoding=True)

            if not verify_result["status"]:
                encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=25)
                verify_result = watermarker.decode_wav(encoded, 44100, phase_shift_decoding=True)
        except Exception as e:
            return audio_array, sample_rate

        output_sample_rate = min(44100, sample_rate)
        if output_sample_rate != 44100:
            encoded = torchaudio.functional.resample(
                encoded, 
                orig_freq=44100, 
                new_freq=output_sample_rate
            ).to(self.device)

        return encoded, output_sample_rate

    @torch.inference_mode()
    def verify(self,
        watermarker: silentcipher.server.Model,
        watermarked_audio: torch.Tensor,
        sample_rate: int,
    ) -> str:
        if len(watermarked_audio.shape) > 1 and watermarked_audio.shape[0] > 1:
            watermarked_audio = watermarked_audio.mean(dim=0)

        if sample_rate != 44100:
            watermarked_audio_44khz = torchaudio.functional.resample(
                watermarked_audio, 
                orig_freq=sample_rate, 
                new_freq=44100
            ).to(self.device)
        else:
            watermarked_audio_44khz = watermarked_audio.to(self.device)

        if len(watermarked_audio_44khz.shape) != 1:
            watermarked_audio_44khz = watermarked_audio_44khz.reshape(-1)

        result_phase = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)

        result_no_phase = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=False)

        if result_phase["status"]:
            watermark = "Watermarked:" + str(result_phase["messages"][0])
        elif result_no_phase["status"]:
            watermark = "Watermarked:" + str(result_no_phase["messages"][0])
        else:
            watermark = "No watermarked"

        return watermark

    def load_watermarker(self, device: str = "cuda", use_cache = True) -> silentcipher.server.Model:
        ckpt_path = os.path.join(models_dir, "TTS", "SilentCipher", "44_1_khz", "73999_iteration")
        config_path = os.path.join(models_dir, ckpt_path, "hparams.yaml")

        if not use_cache and MODEL_CACHE is not None:
            return MODEL_CACHE
        else:
            model = silentcipher.get_model(
                model_type="44.1k", 
                ckpt_path=ckpt_path, 
                config_path=config_path,
                device=device,
            )
            MODEL_CACHE = model
            del model
            torch.cuda.empty_cache()

        return MODEL_CACHE

    def _parse_key(self, key_string):
        """Safely parse encryption key from string
        Args:
            key_string: String representation of key list
        Returns:
            List[int]: Parsed key sequence
        """
        try:
            return ast.literal_eval(key_string)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid key format: {str(e)}")

    def load_audio(self, audio) -> tuple[torch.Tensor, int]:
        waveform = audio["waveform"].squeeze(0)
        audio_array = waveform.mean(dim=0)
        sample_rate = audio["sample_rate"]
        return audio_array, int(sample_rate)

class AdjustAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio":("AUDIO",),
                "volume_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "pitch_shift": ("INT", {"default": 0, "min": -1200, "max": 1200, "step": 100}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "normalize": ("BOOLEAN", {"default": True}),
                "add_echo": ("BOOLEAN", {"default": False}),
                "gain_in": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.1}),
                "gain_out": ("FLOAT", {"default": 0.88, "min": 0.1, "max": 1.0, "step": 0.1}),
                "delays": ("INT", {"default": 60, "min": 10, "max": 500, "step": 10}),
                "decays": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "adjust_audio_properties"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def adjust_audio_properties(self, audio, volume_factor, pitch_shift, speed_factor, normalize, add_echo, gain_in, gain_out, delays, decays):
        """
        调节音频的音量、音高和语速，可选规范化音量和添加回声效果。

        参数说明:
        - audio (AUDIO): 输入音频。
        - volume_factor (float): 音量倍数。
        - pitch_shift (int): 音高偏移。
        - speed_factor (float): 语速倍数。
        - normalize (bool): 是否规范化音量到  -0.1dB。
        - add_echo (bool): 是否添加回声效果。
        - gain_in (float): 输入增益。
        - gain_out (float): 输出增益。
        - delays ([int]): 回声延迟时间。
        - decays ([float]): 回声衰减时间。

        返回:
       - audio (AUDIO): 处理后的音频。
        """
        temp_files = []
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_file_path = temp_file.name

        torchaudio.save(audio_file_path, waveform, sample_rate, format="wav", 
                        bits_per_sample=16, encoding="PCM_S")
        temp_files.append(audio_file_path)

        tfm = sox.Transformer()

        tfm.vol(volume_factor, gain_type='amplitude')

        tfm.pitch(pitch_shift / 100.0)

        tfm.tempo(speed_factor, audio_type='s')

        if normalize:
            tfm.norm(-0.1)

        if add_echo:
            tfm.echo(gain_in=gain_in, gain_out=gain_out, delays=[delays], decays=[decays])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_output = temp_file.name
            tfm.build(audio_file_path, temp_output)
            temp_files.append(temp_output)

        audio, sr = torchaudio.load(temp_output)

        audio_tensor = {"waveform": audio.unsqueeze(0), "sample_rate": sr}

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return (audio_tensor,)

class TrimAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio":("AUDIO",),
                "start_time": ("FLOAT", {"default": 0.00, "min": 0.00, "step": 0.01}),
                "end_time": ("FLOAT", {"default": 0.00, "min": 0.00, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim_audio"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def trim_audio(self, audio, start_time, end_time):
        """
        截取音频的指定时间段。

        参数说明:
        - audio (dict): 输入音频，包含 "waveform" 和 "sample_rate"。
        - start_time (float): 开始时间 (秒)。
        - end_time (float): 结束时间 (秒)。

        返回:
        - tuple: (audio_tensor,) 处理后的音频字典，包含 "waveform" 和 "sample_rate"。
        """
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        temp_files = []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
            audio_file_path = temp_input_file.name
            temp_files.append(audio_file_path)

            torchaudio.save(audio_file_path, waveform, sample_rate, format="wav", 
                            bits_per_sample=16, encoding="PCM_S")

        tfm = sox.Transformer()

        tfm.trim(start_time, end_time)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output_file:
            temp_output = temp_output_file.name
            temp_files.append(temp_output)
            tfm.build(audio_file_path, temp_output)

        audio, sr = torchaudio.load(temp_output)
        audio_tensor = {"waveform": audio.unsqueeze(0), "sample_rate": sr}

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return (audio_tensor,)

class RemoveSilence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "min_silence_duration": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 3.0, "step": 0.1}),
                "silence_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 100, "step": 0.1}),
                "buffer_around_silence": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "remove_silence"
    CATEGORY = "🎤MW/MW-Audio-Tools"

    def remove_silence(self, audio, min_silence_duration, silence_threshold, buffer_around_silence=False):
        temp_files = []
        waveform = audio["waveform"].squeeze(0)
        sample_rate = audio["sample_rate"]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
            audio_file_path = temp_input_file.name
            temp_files.append(audio_file_path)
            torchaudio.save(audio_file_path, waveform, sample_rate, format="wav",
                            bits_per_sample=16, encoding="PCM_S")

        y, sr = librosa.load(audio_file_path, sr=sample_rate)
        normalized_y = librosa.util.normalize(y)
        torchaudio.save(audio_file_path, torch.from_numpy(normalized_y).unsqueeze(0), sample_rate, format="wav",
                    bits_per_sample=16, encoding="PCM_S")

        tfm_silence = sox.Transformer()
        tfm_silence.silence(min_silence_duration=min_silence_duration, 
                            silence_threshold=silence_threshold, 
                            buffer_around_silence=buffer_around_silence)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_base:
            temp_output = temp_base.name
            temp_files.append(temp_output)
        tfm_silence.build(audio_file_path, temp_output, sample_rate_in=sample_rate)

        processed_audio, sr = torchaudio.load(temp_output)
        audio_tensor = {"waveform": processed_audio.unsqueeze(0), "sample_rate": sr}

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return (audio_tensor,)

class MultiLinePromptAT:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "multi_line_prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"

    def promptgen(self, multi_line_prompt: str):
        return (multi_line_prompt.strip(),)


class StringEditNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}), 
             },
            "optional": {
                "paste_edit_text": ("STRING", {"multiline": True, "default": "",}),
                "split_tag": ("STRING", {"default": "",}),
                "form_right": ("BOOLEAN", {"default": False}),
             },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "split_1", "split_2", "edited_text", "merged_text")
    FUNCTION = "edit_text"
    CATEGORY = "🎤MW/MW-Audio-Tools" 
    OUTPUT_NODE = False

    def edit_text(self, text, split_tag="", paste_edit_text="", form_right=False):
        str_1, str_2, str_3, str_4, str_5 = text, "", "", paste_edit_text, ""
        str_5 = text + paste_edit_text
        
        if split_tag != "" and text.find(split_tag) > 0:
            if form_right:
                parts = text.rsplit(split_tag, 1)
                str_2, str_3 = parts[0].strip(), parts[1].strip()
            else:
                parts = text.split(split_tag, 1)
                str_2, str_3 = parts[0].strip(), parts[1].strip()

        return (str_1, str_2, str_3, str_4, str_5)