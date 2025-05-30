[中文](README-CN.md)|[English](README.md)

# Audio Processing Related ComfyUI Nodes

## 📣 Updates

[2025-05-27]⚒️: Add audio denoising enhancement node. Download the model [last_best_checkpoint.pt](https://huggingface.co/alibabasglab/MossFormer2_SE_48K/blob/main/last_best_checkpoint.pt), put it in the directory of `models\TTS\MossFormer2_SE_48K`.

[2025-05-23]⚒️: Fixed the logic issue with the pause node. Now, both serial and parallel execution of pause nodes will pause for the first time. If there is no change in the preceding node, it will automatically pass again.

[2025-04-28]⚒️: Audio loading, customizable loading path, including subdirectories. 

[2025-04-26]⚒️: Pause workflow anywhere.

[2025-04-25]⚒️: String editing. 

[2025-03-28]⚒️: Added watermark embedding node.

[2025-03-26]⚒️: Released version v1.0.0.

## 📖 Introduction

Audio is the bridge connecting text, video, and images. Videos without audio or text are bland. This project currently includes the following main nodes:
- Audio denoising enhancement
- Pause workflow anywhere
- Audio loading, customizable loading path, including subdirectories
  - Please rename the file `extra_help_file.yaml.example` to `extra_help_file.yaml` and remove the annotation `# `. Add a custom loading directory such as `audios_dir: D:\AIGC\ComfyUI-Data\audios_input`, Linux is `/`.
- String editing
- Automatically add subtitles to videos
- Arbitrary time scale audio cropping
- Audio volume, speed, pitch, echo processing, etc.
- Remove silent parts from audio
- Recording
- Audio Watermark Embedding

Examples:

- Pause workflow anywhere:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/20250426115357.png)

- Audio loading:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-04-28_00-34-19.png)

-  String editing:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-05-27_16-35-09.png)

- Add subtitles to video:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_14-00-28.png)

- Arbitrary time scale cropping of audio:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-14-52.png)

- Audio volume, speed, pitch, echo processing, etc.:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-02-40.png)

- Recording and removing silent parts from audio:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-20-30.png)

- Audio Watermark Embedding (Disable watermark embedding; if a watermark exists, it will be automatically detected):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-28_22-18-04.png)

  - To use this node, download all [SilentCipher](https://huggingface.co/Sony/SilentCipher/tree/main/44_1_khz/73999_iteration) models and place them in the `ComfyUI\models\TTS\SilentCipher\44_1_khz\73999_iteration` directory.

## Installation

Install sox and add it to the system path.

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_AudioTools.git
cd ComfyUI_AudioTools
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Acknowledgement

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)