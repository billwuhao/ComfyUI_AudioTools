import os
import torch
import folder_paths
import torchaudio.transforms as T
import torch.nn.functional as F
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import look2hear.models

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

models_dir = folder_paths.models_dir
music_model_path = os.path.join(models_dir, "TTS", "TIGER-DnR")
speech_model_path = os.path.join(models_dir, "TTS", "TIGER-speech")

class MusicSeparation:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "audio": ("AUDIO",),
                },
        }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("dialog", "effect", "music")
    FUNCTION = "separate"
    
    def separate(self, audio):
        model = look2hear.models.TIGERDNR.from_pretrained(music_model_path)
        model.to(device)
        model.eval()

        waveform = audio["waveform"].squeeze(0).to(device) # (1, C, T) -> (C, T)

        with torch.no_grad():
            all_target_dialog, all_target_effect, all_target_music = model(waveform[None])

        return ({"waveform": all_target_dialog.unsqueeze(0).cpu(), "sample_rate": 44100}, 
                {"waveform": all_target_effect.unsqueeze(0).cpu(), "sample_rate": 44100},
                {"waveform": all_target_music.unsqueeze(0).cpu(), "sample_rate": 44100})


class SpeechSeparation:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "audio": ("AUDIO",),
                },
        }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_1", "audio_2", "audio_3")
    FUNCTION = "separate"
    
    def separate(self, audio):
        model = look2hear.models.TIGER.from_pretrained(speech_model_path)
        model.to(device)
        model.eval()

        waveform = audio["waveform"].squeeze(0) # (1, C, T) -> (C, T)
        original_sr = audio["sample_rate"]
        num_channels = waveform.shape[0]
        if num_channels > 1:
            print(f"Input audio has {num_channels} channels. Converting to mono...")
            # 将多通道平均为一个通道
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        target_sr = 16000
        duration = 1  # 1秒
        empty_audio = torch.zeros(1, target_sr * duration)

        if original_sr != target_sr:
            print(f"Resampling audio from {original_sr} Hz to {target_sr} Hz...")
            resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
            waveform = resampler(waveform)
            print("Resampling complete.")
            
        audio = waveform.to(device)

        with torch.no_grad():
            ests_speech = model(audio)  # Expected output shape: [B, num_spk, T]

        print(f"Estimated speech shape: {ests_speech.shape}")
        ests_speech = ests_speech.squeeze(0)
        num_speakers = ests_speech.shape[0]
        if num_speakers == 0:
            raise ValueError("No speakers detected.")
        elif num_speakers == 1:
            audio_1 = ests_speech[0].unsqueeze(0).unsqueeze(0).cpu()
            audio_2 = empty_audio.unsqueeze(0)
            audio_3 = empty_audio.unsqueeze(0)
        elif num_speakers == 2:
            audio_1 = ests_speech[0].unsqueeze(0).unsqueeze(0).cpu()
            audio_2 = ests_speech[1].unsqueeze(0).unsqueeze(0).cpu()
            audio_3 = empty_audio.unsqueeze(0)
        else:
            audio_1 = ests_speech[0].unsqueeze(0).unsqueeze(0).cpu()
            audio_2 = ests_speech[1].unsqueeze(0).unsqueeze(0).cpu()
            audio_3 = ests_speech[2].unsqueeze(0).unsqueeze(0).cpu()
            
        return ({"waveform": audio_1, "sample_rate": target_sr}, 
                {"waveform": audio_2, "sample_rate": target_sr},
                {"waveform": audio_3, "sample_rate": target_sr})


class MergeAudioMW:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
        }

    CATEGORY = "🎤MW/MW-Audio-Tools"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("merged_audio",)
    FUNCTION = "merge"

    def merge(self, audio_1, audio_2):
        waveform_1 = audio_1['waveform'].squeeze(0)
        sample_rate_1 = audio_1['sample_rate']
        waveform_2 = audio_2['waveform'].squeeze(0)
        sample_rate_2 = audio_2['sample_rate']

        target_sample_rate = max(sample_rate_1, sample_rate_2)

        if sample_rate_1 != target_sample_rate:
            resampler_1 = T.Resample(orig_freq=sample_rate_1, new_freq=target_sample_rate)
            waveform_1 = resampler_1(waveform_1)

        if sample_rate_2 != target_sample_rate:
            resampler_2 = T.Resample(orig_freq=sample_rate_2, new_freq=target_sample_rate)
            waveform_2 = resampler_2(waveform_2)

        channels_1 = waveform_1.shape[0]
        channels_2 = waveform_2.shape[0]
        target_channels = max(channels_1, channels_2)

        if channels_1 < target_channels:
            repeat_factor = target_channels // channels_1
            remaining_channels = target_channels % channels_1
            waveform_1 = waveform_1.repeat(repeat_factor, 1)
            if remaining_channels > 0:
                 waveform_1 = torch.cat([waveform_1, waveform_1[:remaining_channels, :]], dim=0)

        if channels_2 < target_channels:
            repeat_factor = target_channels // channels_2
            remaining_channels = target_channels % channels_2
            waveform_2 = waveform_2.repeat(repeat_factor, 1)
            if remaining_channels > 0:
                 waveform_2 = torch.cat([waveform_2, waveform_2[:remaining_channels, :]], dim=0)

        len_1 = waveform_1.shape[-1]
        len_2 = waveform_2.shape[-1]

        if len_1 > len_2:
            padding = len_1 - len_2
            waveform_2 = F.pad(waveform_2, (0, padding))
        elif len_2 > len_1:
            padding = len_2 - len_1
            waveform_1 = F.pad(waveform_1, (0, padding))

        merged_waveform = waveform_1 + waveform_2

        merged_waveform = torch.clamp(merged_waveform, -1.0, 1.0)

        return ({"waveform": merged_waveform.unsqueeze(0), "sample_rate": target_sample_rate},)
