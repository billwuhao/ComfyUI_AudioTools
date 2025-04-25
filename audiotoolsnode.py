import sox
import tempfile
import os
from glob import glob
import torchaudio
import librosa
import torch
# import sounddevice as sd
# from scipy import ndimage
import ast
import silentcipher
import folder_paths


models_dir = folder_paths.models_dir


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
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cached_model = None
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
                        "default": False,
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
        watermarker = self.load_watermarker(device=self.device, use_cache=True)
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

