[中文](README-CN.md)|[English](README.md)

# ComfyUI Nodes for Audio Processing and Related Tasks

## 📣 Updates

[2025-06-03]⚒️: v1.2.0. Added Music/Vocal Separation, Vocal Extraction, and Audio Merging nodes. Download models [TIGER-speech](https://hf-mirror.com/JusperLee/TIGER-speech/tree/main), [TIGER-DnR](https://hf-mirror.com/JusperLee/TIGER-DnR/tree/main), and place the entire folders into the `models\TTS` directory.

[2025-05-27]⚒️: Added Audio Denoising and Enhancement node. Download model [last_best_checkpoint.pt](https://huggingface.co/alibabasglab/MossFormer2_SE_48K/blob/main/last_best_checkpoint.pt) and place it into the `models\TTS\MossFormer2_SE_48K` directory.

[2025-05-23]⚒️: Fixed logic issues with the Pause node. Now, the Pause node will pause on the first execution when connected in series or parallel. On subsequent executions, it will automatically pass if the preceding nodes have not changed.

[2025-04-28]⚒️: Audio Loading, with custom loading paths including subdirectories.

[2025-04-26]⚒️: Pause workflow anywhere.

[2025-04-25]⚒️: String Editing.

[2025-03-28]⚒️: Added Watermark Embedding node.

[2025-03-26]⚒️: Released version v1.0.0.

## 📖 Introduction

Audio acts as a bridge connecting text, video, and images. A video without audio or text is tasteless. This project currently includes the following main nodes:

- Music/Vocal Separation, Vocal Extraction, Audio Merging, Audio Concatenation
- Audio Denoising and Enhancement
- Pause workflow anywhere
- Audio Loading, with custom loading paths including subdirectories
  - Please rename the `extra_help_file.yaml.example` file to `extra_help_file.yaml`, uncomment `# `, and add custom loading directories like `audios_dir: D:\AIGC\ComfyUI-Data\audios_input`. For Linux, use `/`.
- String Editing.
- Automatic Video Subtitling
- Audio Trimming at Arbitrary Time Markers
- Audio Volume, Speed, Pitch, Echo Processing, etc.
- Remove Silent Parts from Audio
- Audio Recording
- Audio Watermark Embedding

Examples:

- Music/Vocal Separation:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_23-21-05.png)

- Vocal Separation and Extraction:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_22-45-13.png)

- Merge Audio:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_20-50-29.png)

- Denoising and Enhancement:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_20-46-28.png)

- Audio Loading:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-04-28_00-34-19.png)

- String Editing.

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-05-27_16-35-09.png)

- Add Subtitles to Video:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_14-00-28.png)

- Trim Audio at Arbitrary Time Markers:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-14-52.png)

- Audio Volume, Speed, Pitch, Echo Processing, etc.:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-02-40.png)

- Audio Recording and Remove Silent Parts:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-20-30.png)

- Audio Watermark Embedding (Embedding disabled, if watermark exists, it will be automatically detected):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-28_22-18-04.png)

  - To use this node, download all models from [SilentCipher](https://huggingface.co/Sony/SilentCipher/tree/main/44_1_khz/73999_iteration) and place them into the `ComfyUI\models\TTS\SilentCipher\44_1_khz\73999_iteration` directory.

## Installation

Install [sox](https://sourceforge.net/projects/sox/) and add it to your system's PATH.

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_AudioTools.git
cd ComfyUI_AudioTools
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Acknowledgments

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- [TIGER](https://github.com/JusperLee/TIGER)