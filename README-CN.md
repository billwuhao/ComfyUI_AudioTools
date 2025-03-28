[中文](README-CN.md)|[English](README.md)

# 音频处理等相关的 ComfyUI 节点

音频是连接文本, 视频, 图像的桥梁, 没有音频或文字的视频是无味的. 这个项目目前包括以下几个主要节点:
- 视频自动添加字幕
- 音频任意时间刻度裁剪
- 音频音量, 速度, 音高, 回音处理等
- 去除音频中无声部分
- 录音
- 音频水印嵌入

示例:

1, 视频添加字幕:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_14-00-28.png)

2, 结合 [ComfyUI_EraX-WoW-Turbo](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo) 语音自动识别, 然后给视频添加字幕:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-33-54.png)

3, 结合 [ComfyUI_EraX-WoW-Turbo](https://github.com/billwuhao/ComfyUI_EraX-WoW-Turbo), [ComfyUI_gemmax](https://github.com/billwuhao/ComfyUI_gemmax), [ComfyUI_SparkTTS](https://github.com/billwuhao/ComfyUI_SparkTTS), [ComfyUI-LatentSyncWrapper](https://github.com/ShmuelRonen/ComfyUI-LatentSyncWrapper) 自动识别语音, 自动翻译, 自动克隆声音, 自动对口型, 视频自动添加字幕(详解示例工作流 [workflow-examples](./workflow-examples)):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/20250326001631.png)

4, 任意时间刻度裁剪音频:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-14-52.png)

5, 音频音量, 速度, 音高, 回音处理等:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-02-40.png)

6, 去除音频中无声部分和录音:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-20-30.png)

7, 音频水印嵌入(关闭水印嵌入, 如果有水印, 会自动检测):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-28_22-18-04.png)

- 如果要使用该节点, [SilentCipher](https://huggingface.co/Sony/SilentCipher/tree/main/44_1_khz/73999_iteration) 全部模型下载放到 `ComfyUI\models\TTS\SilentCipher\44_1_khz\73999_iteration` 目录下.

## 📣 更新

[2025-03-28]⚒️: 增加水印嵌入节点. 

[2025-03-26]⚒️: 发布版本 v1.0.0. 

## 安装

安装 sox 并添加到系统 path.

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_AudioTools.git
cd ComfyUI_AudioTools
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```
