[中文](README-CN.md)|[English](README.md)

# 音频处理等相关的 ComfyUI 节点

## 📣 更新

[2025-06-03]⚒️: v1.2.0. 增加 音乐/人声分离, 人声提取, 音频合并节点. 下载模型 [TIGER-speech](https://hf-mirror.com/JusperLee/TIGER-speech/tree/main), [TIGER-DnR](https://hf-mirror.com/JusperLee/TIGER-DnR/tree/main), 整个文件夹放到 `models\TTS` 目录下.

[2025-05-27]⚒️: 增加音频去噪增强节点. 下载模型 [last_best_checkpoint.pt](https://huggingface.co/alibabasglab/MossFormer2_SE_48K/blob/main/last_best_checkpoint.pt), 放到 `models\TTS\MossFormer2_SE_48K` 目录下.

[2025-05-23]⚒️: 修复暂停节点逻辑问题, 现在暂停节点串联/并联首次执行都会暂停, 再次执行如果前置节点无变化自动通过. 

[2025-04-28]⚒️: 音频加载, 可自定义加载路径, 包含子目录. 

[2025-04-26]⚒️: 任何地方暂停工作流. 

[2025-04-25]⚒️: 字符串编辑. 

[2025-03-28]⚒️: 增加水印嵌入节点. 

[2025-03-26]⚒️: 发布版本 v1.0.0. 

## 📖 介绍

音频是连接文本, 视频, 图像的桥梁, 没有音频或文字的视频是无味的. 这个项目目前包括以下几个主要节点:

- 音乐/人声分离, 人声提取, 音频合并, 音频连接
- 音频去噪增强
- 任何地方暂停工作流
- 音频加载, 可自定义加载路径, 包含子目录
  - 请将 `extra_help_file.yaml.example` 文件改名为 `extra_help_file.yaml`, 并取消注释 `# `, 添加自定义加载目录如 `audios_dir: D:\AIGC\ComfyUI-Data\audios_input`, linux 是 `/`.
- 字符串编辑. 
- 视频自动添加字幕
- 音频任意时间刻度裁剪
- 音频音量, 速度, 音高, 回音处理等
- 去除音频中无声部分
- 录音
- 音频水印嵌入

示例:

- 音乐/人声分离:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_23-21-05.png)

- 人声分离提取:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_22-45-13.png)

- 合并音频:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_20-50-29.png)

- 去噪增强:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-06-03_20-46-28.png)

- 音频加载:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-04-28_00-34-19.png)

- 字符串编辑.

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-05-27_16-35-09.png)

- 视频添加字幕:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_14-00-28.png)

- 任意时间刻度裁剪音频:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-14-52.png)

- 音频音量, 速度, 音高, 回音处理等:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-02-40.png)

- 录音 和 去除音频中无声部分:

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-25_13-20-30.png)

- 音频水印嵌入(关闭水印嵌入, 如果有水印, 会自动检测):

![](https://github.com/billwuhao/ComfyUI_AudioTools/blob/main/images/2025-03-28_22-18-04.png)

  - 如果要使用该节点, [SilentCipher](https://huggingface.co/Sony/SilentCipher/tree/main/44_1_khz/73999_iteration) 全部模型下载放到 `ComfyUI\models\TTS\SilentCipher\44_1_khz\73999_iteration` 目录下.

## 安装

安装 [sox](https://sourceforge.net/projects/sox/) 并添加到系统 path.

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_AudioTools.git
cd ComfyUI_AudioTools
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 鸣谢

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)
- [TIGER](https://github.com/JusperLee/TIGER)