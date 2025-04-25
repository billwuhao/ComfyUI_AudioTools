[中文](README-CN.md)|[English](README.md)

# Audio Processing Related ComfyUI Nodes

Audio is the bridge connecting text, video, and images. Videos without audio or text are bland. This project currently includes the following main nodes:
- Automatically add subtitles to videos
- Arbitrary time scale audio cropping
- Audio volume, speed, pitch, echo processing, etc.
- Remove silent parts from audio
- Recording
- Audio Watermark Embedding
- String show and editing. If editing is enabled, the workflow will pause execution. After editing, you can click "Continue Workflow" to continue execution

Examples:

1, Add subtitles to video:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_14-00-28.png)

2, Combine [ComfyUI_EraX-WoW-Turbo](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo) for automatic speech recognition, and then add subtitles to the video:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-33-54.png)

3, Combine [ComfyUI_EraX-WoW-Turbo](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo), [ComfyUI_gemmax](https://github.com/billwuhao/ComfyUI_gemmax), [ComfyUI_SparkTTS](https://github.com/billwuhao/ComfyUI_SparkTTS), [ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper) for automatic speech recognition, automatic translation, automatic voice cloning, automatic lip sync, automatic subtitle addition to video (detailed example workflow [workflow-examples](./workflow-examples)):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/20250326001631.png)

4, Arbitrary time scale cropping of audio:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-14-52.png)

5, Audio volume, speed, pitch, echo processing, etc.:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-02-40.png)

6, Remove silent parts from audio and recording:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-20-30.png)

7, Audio Watermark Embedding (Disable watermark embedding; if a watermark exists, it will be automatically detected):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-28_22-18-04.png)

- To use this node, download all [SilentCipher](https://huggingface.co/Sony/SilentCipher/tree/main/44_1_khz/73999_iteration) models and place them in the `ComfyUI\models\TTS\SilentCipher\44_1_khz\73999_iteration` directory.

8, String show and editing. If editing is enabled, the workflow will pause execution. After editing, you can click "Continue Workflow" to continue execution:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-04-25_15-48-54.png)

## 📣 Updates

[2025-04-25]⚒️: String show and editing. 

[2025-03-28]⚒️: Added watermark embedding node.

[2025-03-26]⚒️: Released version v1.0.0.

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
