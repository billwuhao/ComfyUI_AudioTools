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
import json
import io
import struct


models_dir = folder_paths.models_dir
input_dir = folder_paths.get_input_directory()

def get_path():
    from pathlib import Path
    import yaml
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(script_dir, "extra_help_file.yaml")
    try:
        # 尝试打开并加载 YAML 文件
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

    # 如果加载失败，返回默认路径
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
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths


from comfy.cli_args import args
import av
# from ComfyUI
def save_audio(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None, quality="128k"):

    filename_prefix += self.prefix_append
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
    results = []

    # Prepare metadata dictionary
    metadata = {}
    if not args.disable_metadata:
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    # Opus supported sample rates
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

    for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.{format}"
        output_path = os.path.join(full_output_folder, file)

        # Use original sample rate initially
        sample_rate = audio["sample_rate"]

        # Handle Opus sample rate requirements
        if format == "opus":
            if sample_rate > 48000:
                sample_rate = 48000
            elif sample_rate not in OPUS_RATES:
                # Find the next highest supported rate
                for rate in sorted(OPUS_RATES):
                    if rate > sample_rate:
                        sample_rate = rate
                        break
                if sample_rate not in OPUS_RATES:  # Fallback if still not supported
                    sample_rate = 48000

            # Resample if necessary
            if sample_rate != audio["sample_rate"]:
                waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

        # Create in-memory WAV buffer
        wav_buffer = io.BytesIO()
        torchaudio.save(wav_buffer, waveform, sample_rate, format="WAV")
        wav_buffer.seek(0)  # Rewind for reading

        # Use PyAV to convert and add metadata
        input_container = av.open(wav_buffer)

        # Create output with specified format
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format=format)

        # Set metadata on the container
        for key, value in metadata.items():
            output_container.metadata[key] = value

        # Set up the output stream with appropriate properties
        input_container.streams.audio[0]
        if format == "opus":
            out_stream = output_container.add_stream("libopus", rate=sample_rate)
            if quality == "64k":
                out_stream.bit_rate = 64000
            elif quality == "96k":
                out_stream.bit_rate = 96000
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "192k":
                out_stream.bit_rate = 192000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        elif format == "mp3":
            out_stream = output_container.add_stream("libmp3lame", rate=sample_rate)
            if quality == "V0":
                #TODO i would really love to support V3 and V5 but there doesn't seem to be a way to set the qscale level, the property below is a bool
                out_stream.codec_context.qscale = 1
            elif quality == "128k":
                out_stream.bit_rate = 128000
            elif quality == "320k":
                out_stream.bit_rate = 320000
        else: #format == "flac":
            out_stream = output_container.add_stream("flac", rate=sample_rate)


        # Copy frames from input to output
        for frame in input_container.decode(audio=0):
            frame.pts = None  # Let PyAV handle timestamps
            output_container.mux(out_stream.encode(frame))

        # Flush encoder
        output_container.mux(out_stream.encode(None))

        # Close containers
        output_container.close()
        input_container.close()

        # Write the output to file
        output_buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(output_buffer.getbuffer())

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        counter += 1

    return { "ui": { "audio": results } }


class SaveAudioMW:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                              "format": (["MP3", "FLAC"],),
                              "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
                              "mp3_quality": (["V0", "128k", "320k"],{"default": "V0"}),
                              },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"

    OUTPUT_NODE = True

    CATEGORY = "🎤MW/MW-Audio-Tools"

    def save_audio(self, audio, format, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None, mp3_quality="128k"):

        return save_audio(self, audio, filename_prefix, format.lower(), prompt, extra_pnginfo, quality=mp3_quality)


# # --- Function to build a file tree structure ---
# def build_file_tree(base_path, extensions, current_path=""):
#     """
#     Builds a recursive tree structure of files and directories.
#     Each node is a dictionary:
#     { "name": "...", "type": "folder" or "file", "path": "relative_path", "children": [...] }
#     """
#     tree = []
#     full_current_path = os.path.join(base_path, current_path)
    
#     if not os.path.isdir(full_current_path):
#         return []

#     # Sort entries to have folders first, then files, alphabetically
#     entries = sorted(os.listdir(full_current_path))
    
#     # Separate folders and files for processing
#     dirs = [d for d in entries if os.path.isdir(os.path.join(full_current_path, d))]
#     files = [f for f in entries if os.path.isfile(os.path.join(full_current_path, f))]

#     for name in dirs:
#         node_path = os.path.join(current_path, name)
#         children = build_file_tree(base_path, extensions, node_path)
#         # Only add folder if it has children or you want to show empty folders
#         if children: # or True, if you want to show empty folders
#             tree.append({
#                 "name": name,
#                 "type": "folder",
#                 "path": os.path.normpath(node_path), # Store normalized relative path
#                 "children": children
#             })

#     for name in files:
#         if any(name.lower().endswith(ext.lower()) for ext in extensions):
#             node_path = os.path.join(current_path, name)
#             tree.append({
#                 "name": name,
#                 "type": "file",
#                 "path": os.path.normpath(node_path) # Store normalized relative path
#             })
#     return tree


# class LoadAudioMW:
#     audios_dir_absolute = get_path()
#     _audio_extensions = [".wav", ".mp3", ".flac", ".mp4", ".WAV", ".MP3", ".FLAC", ".MP4"]
#     _file_tree_data = build_file_tree(audios_dir_absolute, _audio_extensions)
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 # The first element of the tuple is the data for our custom widget.
#                 # The second element is a config dict. "widget": "HIERARCHICAL_FILE_AUDIO" is key.
#                 "selected_audio_path": (s._file_tree_data, {"widget": "HIERARCHICAL_FILE_AUDIO"}),
#                 "start_time_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.01}),
#                 "duration_sec": ("FLOAT", {"default": 5.0, "min": 0.01, "max": 10000.0, "step": 0.01}),
#             }
#         }

#     CATEGORY = "🎤MW/MW-Audio-Tools"
#     RETURN_TYPES = ("AUDIO",)
#     FUNCTION = "load"

#     def load(self, selected_audio_path, start_time_sec=0, duration_sec=5):
#         if not selected_audio_path or selected_audio_path.startswith("ERROR:"):
#             raise ValueError("No valid audio file selected or audio directory error.")

#         # selected_audio_path is the RELATIVE path from the JS widget
#         # We need to join it with the absolute base audio directory
#         if not LoadAudioMW.audios_dir_absolute:
#              raise FileNotFoundError("Audio base directory is not configured.")
             
#         full_audio_path = os.path.join(LoadAudioMW.audios_dir_absolute, selected_audio_path)

#         # --- Rest of your audio loading logic from the original example ---
#         # import torchaudio # Make sure torchaudio is imported
#         waveform, sample_rate = torchaudio.load(full_audio_path)
#         waveform = waveform.unsqueeze(0)
        
#         start_sample = round(start_time_sec * sample_rate)
#         num_samples_requested = round(duration_sec * sample_rate)
#         total_samples = waveform.shape[-1]
#         start_sample = max(0, min(start_sample, total_samples))
#         actual_end_sample = min(start_sample + num_samples_requested, total_samples)

#         if start_sample >= actual_end_sample:
#             raise ValueError(f"SliceAudio: Calculated slice is empty or invalid.")
        
#         sliced_waveform = waveform[..., start_sample:actual_end_sample]
#         output_audio = {"waveform": sliced_waveform, "sample_rate": sample_rate}
#         return (output_audio,)

class LoadAudioMW:
    audios_dir = get_path()
    files = get_all_files(audios_dir, extensions=[".wav", ".mp3", ".flac", ".mp4", ".WAV", ".MP3", ".FLAC", ".MP4"], relative_path=True)
    # for i in files:
    #     import shutil
    #     src_path = folder_paths.get_annotated_filepath(i, audios_dir)
    #     dst_path = os.path.join(input_dir, i)
    #     os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    #     if not os.path.exists(dst_path):
    #         shutil.copy2(src_path, dst_path)
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": (sorted(s.files),),
                    "start_time_sec": ("FLOAT", {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 10000.0, # Arbitrarily high max
                        "step": 0.01,
                        "display": "number", # Use number input instead of slider
                        "tooltip": "Start time of the slice in seconds"
                        }),
                    "duration_sec": ("FLOAT", {
                        "default": 5.0,
                        "min": 0.01, # Minimum duration slightly above zero
                        "max": 10000.0, # Arbitrarily high max
                        "step": 0.01,
                        "display": "number",
                        "tooltip": "Desired duration of the slice in seconds"
                        }),
                    },
                }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO", )
    FUNCTION = "load"

    def load(self, audio, start_time_sec=0, duration_sec=5):
        audio_path = os.path.join(LoadAudioMW.audios_dir, audio)
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.unsqueeze(0)
        
        if waveform is None or waveform.numel() == 0 or sample_rate <= 0:
            raise Exception("SliceAudio received invalid input audio.")

        # Ensure parameters are non-negative
        start_time_sec = max(0.0, start_time_sec)
        duration_sec = max(0.01, duration_sec) # Ensure minimum duration

        # Use rounding for potentially better accuracy near frame boundaries
        start_sample = round(start_time_sec * sample_rate)
        num_samples_requested = round(duration_sec * sample_rate)

        total_samples = waveform.shape[-1] # Get length from the last dimension

        # Note: If start_sample == total_samples, the slice will be empty.
        start_sample = max(0, min(start_sample, total_samples))

        # The end sample for slicing is exclusive: [start:end]
        actual_end_sample = min(start_sample + num_samples_requested, total_samples)

        if start_sample >= actual_end_sample:
            raise ValueError(f"SliceAudio requested slice starts at or after the calculated end ({start_sample} >= {actual_end_sample}).")
        else:
            sliced_waveform = waveform[..., start_sample:actual_end_sample]

        output_audio = {
            "waveform": sliced_waveform,
            "sample_rate": sample_rate
        }
        
        return {"result": (output_audio,)}


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

        # --- Determine Target Sample Rate ---
        final_sr = max(sr_a, sr_b)

        # --- Resample if necessary ---
        # Perform resampling before channel adjustment, as resampling doesn't change channels
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

        # --- Determine Target Channels and Adjust ---
        channels_a = resampled_waveform_a.shape[1]
        channels_b = resampled_waveform_b.shape[1]

        # Determine target channels based on the new rule
        if channels_a == 1 and channels_b == 1:
            target_channels = 1 # Both mono, output mono
            print("Concatenate Audio Node: Both inputs are mono, output will be mono (1 channel).")
        else:
            target_channels = 2 # Otherwise, output stereo
            print(f"Concatenate Audio Node: At least one input is not mono ({channels_a} vs {channels_b}), output will be stereo (2 channels).")


        # Helper function to adjust channels of a single waveform
        def adjust_channels(wf, current_channels, target_channels, name):
            if current_channels == target_channels:
                return wf
            elif target_channels == 1 and current_channels > 1:
                 # Should not happen based on the target_channels logic (we only target 1 if both inputs are 1)
                 # but added as a safeguard/placeholder. Downmixing >1 to 1 would be needed here.
                 print(f"Concatenate Audio Node Warning: Attempting to downmix {name} from {current_channels} to {target_channels} (mono). Simple average downmix applied.")
                 # Simple average downmix to mono
                 return wf.mean(dim=1, keepdim=True)
            elif target_channels == 2:
                if current_channels == 1:
                    # Mono to Stereo: Duplicate the channel
                    print(f"Concatenate Audio Node: Converting {name} from {current_channels} to {target_channels} channels (mono to stereo).")
                    return wf.repeat(1, target_channels, 1) # repeat(batch_dim=1, channel_dim=target_channels, time_dim=1)
                elif current_channels > 2:
                    # Multi-channel to Stereo: Simple average downmix
                    print(f"Concatenate Audio Node Warning: Converting {name} from {current_channels} to {target_channels} channels (multi-channel to stereo). Applying simple average downmix.")
                    # Average all channels to get a mono signal, then duplicate for stereo
                    mono_wf = wf.mean(dim=1, keepdim=True)
                    return mono_wf.repeat(1, target_channels, 1)
                # If current_channels == 2, it matches target_channels=2, handled by the first check.
            else:
                 # This case should also ideally not happen with target_channels being only 1 or 2
                 raise RuntimeError(f"Concatenate Audio Node: Unsupported channel adjustment requested for {name}: from {current_channels} to {target_channels}.")

        # Apply channel adjustment
        adjusted_waveform_a = adjust_channels(resampled_waveform_a, channels_a, target_channels, "Audio A")
        adjusted_waveform_b = adjust_channels(resampled_waveform_b, channels_b, target_channels, "Audio B")

        # --- Concatenate ---
        # Concatenate along the time dimension (dimension 2 for (batch, channels, time))
        concatenated_waveform = torch.cat((adjusted_waveform_a, adjusted_waveform_b), dim=2)

        # --- Prepare Output ---
        output_audio = {
            "waveform": concatenated_waveform,
            "sample_rate": final_sr # Use the determined final sample rate
        }

        # Return the output wrapped in a tuple (ComfyUI requirement)
        return (output_audio,)


class AudioAddWatermark:
    def __init__(self):
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.device = device
        self.cached_model = None

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
                # "optional": {
                #     "check_watermark": ("BOOLEAN", {"default": False, "tooltip": "Check if the audio contains watermark."}),
                #     }
                }


    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "watermark")
    FUNCTION = "watermarkgen"

    def watermarkgen(self, audio, add_watermark, key, unload_model):
        """Main watermark processing pipeline"""
        ckpt_path = os.path.join(models_dir, "TTS", "SilentCipher", "44_1_khz", "73999_iteration")
        config_path = os.path.join(models_dir, ckpt_path, "hparams.yaml")

        if self.cached_model is None:
            self.cached_model = silentcipher.get_model(
                model_type="44.1k", 
                ckpt_path=ckpt_path, 
                config_path=config_path,
                device=self.device,
            )

        watermarker = self.cached_model
        audio_array, sample_rate = self.load_audio(audio)
        # Ensure tensor on correct device
        audio_array = audio_array.to(self.device)

        if add_watermark:
            key = self._parse_key(key)
            audio_array, sample_rate = self.watermark(watermarker, audio_array, sample_rate, key)

        watermark = self.verify(watermarker, audio_array, sample_rate)
        if unload_model:
            del watermarker
            self.cached_model = None
            torch.cuda.empty_cache()
        
        # Move data back to CPU before return
        return ({"waveform": audio_array.unsqueeze(0).unsqueeze(0).cpu(), "sample_rate": sample_rate}, watermark)

    @torch.inference_mode()
    def watermark(self,
        watermarker: silentcipher.server.Model,
        audio_array: torch.Tensor,
        sample_rate: int,
        watermark_key: list[int],
    ) -> tuple[torch.Tensor, int]:
        # Ensure mono channel
        if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
            audio_array = audio_array.mean(dim=0)
        
        audio_array = audio_array.to(self.device)
        
        audio_array_44khz = torchaudio.functional.resample(
            audio_array, 
            orig_freq=sample_rate, 
            new_freq=44100
        ).to(self.device)
        
        # Ensure correct tensor shape (should be 1D)
        if len(audio_array_44khz.shape) != 1:
            audio_array_44khz = audio_array_44khz.reshape(-1)
            
        try:
            # Enhance watermark strength by reducing SDR threshold
            encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=30)
            
            verify_result = watermarker.decode_wav(encoded, 44100, phase_shift_decoding=True)
            
            if not verify_result["status"]:
                encoded, _ = watermarker.encode_wav(audio_array_44khz, 44100, watermark_key, calc_sdr=False, message_sdr=25)
                verify_result = watermarker.decode_wav(encoded, 44100, phase_shift_decoding=True)
        except Exception as e:
            return audio_array, sample_rate

        # Resample back to original rate if needed
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
            
        
        # 尝试不同的解码参数
        # 1. 使用相位偏移解码
        result_phase = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=True)
        
        # 2. 不使用相位偏移解码
        result_no_phase = watermarker.decode_wav(watermarked_audio_44khz, 44100, phase_shift_decoding=False)
        
        # 使用两种方法中任一种成功的结果
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

        if not use_cache and self.cached_model is not None:
            return self.cached_model
        else:
            model = silentcipher.get_model(
                model_type="44.1k", 
                ckpt_path=ckpt_path, 
                config_path=config_path,
                device=device,
            )
            self.cached_model = model
            del model
            torch.cuda.empty_cache()
            
        return self.cached_model


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
                "volume_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "pitch_shift": ("INT", {"default": 0, "min": -1200, "max": 1200, "step": 100}),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "normalize": ("BOOLEAN", {"default": False}),
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
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio_file_path = temp_file.name
        
        # 使用 torchaudio.save() 保存临时文件
        torchaudio.save(audio_file_path, waveform, sample_rate, format="wav", 
                        bits_per_sample=16, encoding="PCM_S")
        temp_files.append(audio_file_path)

        tfm = sox.Transformer()

        # 音量调整
        tfm.vol(volume_factor, gain_type='amplitude')
        # 音高调整
        tfm.pitch(pitch_shift / 100.0)
        # 语速调整
        tfm.tempo(speed_factor, audio_type='s')
        # 可选规范化音量
        if normalize:
            tfm.norm(-0.1)
        # 可选添加回声效果
        if add_echo:
            tfm.echo(gain_in=gain_in, gain_out=gain_out, delays=[delays], decays=[decays])
        # 使用临时文件保存中间结果
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_output = temp_file.name
            tfm.build(audio_file_path, temp_output)
            temp_files.append(temp_output)

        audio, sr = torchaudio.load(temp_output)

        audio_tensor = {"waveform": audio.unsqueeze(0), "sample_rate": sr}
        
        # 清理所有临时文件
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
        
        # 记录所有临时文件路径
        temp_files = []

        # 创建并保存输入音频的临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
            audio_file_path = temp_input_file.name
            temp_files.append(audio_file_path)
            
            # 使用 torchaudio.save() 保存临时文件
            torchaudio.save(audio_file_path, waveform, sample_rate, format="wav", 
                            bits_per_sample=16, encoding="PCM_S")

        tfm = sox.Transformer()

        # 设置截取时间
        tfm.trim(start_time, end_time)

        # 使用临时文件保存截取结果
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_output_file:
            temp_output = temp_output_file.name
            temp_files.append(temp_output)
            tfm.build(audio_file_path, temp_output)

        # 加载处理后的音频
        audio, sr = torchaudio.load(temp_output)
        audio_tensor = {"waveform": audio.unsqueeze(0), "sample_rate": sr}

        # 清理所有临时文件
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
           

# class AudioDenoising:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 # 触发控制
#                 "audio": ("AUDIO",),
#                 "n_fft": ("INT", {  # 限定为2的幂次方
#                     "default": 2048,
#                     "min": 512,
#                     "max": 4096,
#                     "step": 512  # 只能选择512,1024,1536,2048,...4096
#                 }),
#                 "sensitivity": ("FLOAT", {  # 灵敏度精确控制
#                     "default": 1.2,
#                     "min": 0.5,
#                     "max": 3.0,
#                     "step": 0.1  # 0.1步进
#                 }),
#                 "smooth": ("INT", {  # 确保为奇数
#                     "default": 5,
#                     "min": 1,
#                     "max": 11,
#                     "step": 2  # 生成1,3,5,7,9,11
#                 }),
#                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
#             }
#         }

#     RETURN_TYPES = ("AUDIO",)
#     RETURN_NAMES = ("audio",)
#     FUNCTION = "clean"
#     CATEGORY = "🎤MW/MW-Audio-Tools"

#     def _stft(self, y, n_fft):
#         hop = n_fft // 4
#         return librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft)

#     def _istft(self, spec, n_fft):
#         hop = n_fft // 4
#         return librosa.istft(spec, hop_length=hop, win_length=n_fft)

#     def _calc_noise_profile(self, noise_clip, n_fft):
#         noise_spec = self._stft(noise_clip, n_fft)
#         return {
#             'mean': np.mean(np.abs(noise_spec), axis=1, keepdims=True),
#             'std': np.std(np.abs(noise_spec), axis=1, keepdims=True)
#         }

#     def _spectral_gate(self, spec, noise_profile, sensitivity):
#         threshold = noise_profile['mean'] + sensitivity * noise_profile['std']
#         return np.where(np.abs(spec) > threshold, spec, 0)

#     def _smooth_mask(self, mask, kernel_size):
#         smoothed = ndimage.uniform_filter(mask, size=(kernel_size, kernel_size))
#         return np.clip(smoothed * 1.2, 0, 1)  # 增强边缘保留

#     def clean(self, audio, n_fft, sensitivity, smooth, seed):
#         noise_clip = None
        
#         waveform = audio["waveform"].squeeze().numpy().astype(np.float32)
#         sample_rate = audio["sample_rate"]

#         energy = librosa.feature.rms(y=waveform, frame_length=n_fft, hop_length=n_fft//4)
#         min_idx = np.argmin(energy)
#         start = min_idx * (n_fft//4)
#         noise_clip = waveform[start:start + n_fft*2]

#         # 降噪处理
#         noise_profile = self._calc_noise_profile(noise_clip, n_fft)
#         spec = self._stft(waveform, n_fft)
        
#         # 多步骤处理
#         mask = np.ones_like(spec)  # 初始掩膜
#         for _ in range(2):  # 双重处理循环
#             cleaned_spec = self._spectral_gate(spec, noise_profile, sensitivity)
#             mask = np.where(np.abs(cleaned_spec) > 0, 1, 0)
#             mask = self._smooth_mask(mask, smooth//2+1)
#             spec = spec * mask

#         # 相位恢复重建
#         processed = self._istft(spec * mask, n_fft)
        
#         # 动态增益归一化
#         peak = np.max(np.abs(processed))
#         processed = processed * (0.99 / peak) if peak > 0 else processed

#         # 格式转换
#         waveform = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
#         final_audio = {"waveform": waveform, "sample_rate": sample_rate}

#         return (final_audio,)


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

