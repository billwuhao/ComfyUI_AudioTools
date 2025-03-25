import sox
import tempfile
import os
from glob import glob
import torchaudio
import librosa
import torch
# import sounddevice as sd
# from scipy import ndimage


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


from .AddSubtitlesToVideo import AddSubtitlesToTensor
from .MWAudioRecorderAT import AudioRecorderAT

NODE_CLASS_MAPPINGS = {
    "AdjustAudio": AdjustAudio,
    "TrimAudio": TrimAudio,
    "RemoveSilence": RemoveSilence,
    # "AudioDenoising": AudioDenoising,
    "AudioRecorderAT": AudioRecorderAT,
    "AddSubtitlesToVideo": AddSubtitlesToTensor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdjustAudio": "Adjust Audio",
    "TrimAudio": "Trim Audio",
    "RemoveSilence": "Remove Silence",
    # "AudioDenoising": "Audio Denoising",
    "AudioRecorderAT": "MW Audio Recorder",
    "AddSubtitlesToVideo": "Add Subtitles To Video",
}