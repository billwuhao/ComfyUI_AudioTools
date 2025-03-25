import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ast


class AddSubtitlesToTensor:
    """
    ComfyUI 节点，用于将字幕添加到视频帧的张量中。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),  
                "subtitles": ("STRING", ),
                "language": (["类中文字符", "English_word"], {"default": "类中文字符"}), 
                "max_words_per_line": (
                    "INT", {"default": 12, "min": 1, "max": 30}),  
                "font_path": (
                    "STRING", {"default": "simhei.ttf"}),  
                "font_size": ("INT", {"default": 30, "min": 1, "max": 200}),  
                "font_color": ("STRING", {"default": "black"}),  
                "subtitle_background_color": (
                    "STRING", {"default": "#B8860B", "tooltip": "None: Do not add background color"}),
                "y_offset": (
                    "INT", {"default": 0, "min": -1000, "max": 1000, "step": 5}),
                "x_offset": (
                    "INT", {"default": 0, "min": -1000, "max": 1000, "step": 5}),
                "width": ("INT", {"default": 1024}), 
                "height": ("INT", {"default": 1024}),  
                "fps": ("FLOAT", {"default": 20, "min": 1, "max": 240}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "add_subtitles"

    CATEGORY = "🎤MW/MW-Audio-Tools"  

    def add_subtitles(self, 
                      images, 
                      subtitles, 
                      language,
                      font_path, 
                      font_size, 
                      font_color, 
                      y_offset, 
                      x_offset, 
                      width, 
                      height, 
                      max_words_per_line, 
                      subtitle_background_color, 
                      fps,
                      ):

        if subtitle_background_color == "None":
            subtitle_background_color = None

        try:
            subtitles = ast.literal_eval(subtitles)
            if not isinstance(subtitles, list):
                raise ValueError("Subtitles must be a list of dictionaries.")
            for sub in subtitles:
                if not isinstance(sub, dict):
                    raise ValueError("Each subtitle must be a dictionary.")
        except Exception as e:
            print(f"Error parsing subtitles: {e}")

        if not isinstance(images, torch.Tensor):
            print("Images must be a torch tensor")
            return (images,)
        
        print(f"images.shape:{images.shape}")
        if len(images.shape) == 3:
            images = images.unsqueeze(0) 
        images = torch.clamp(images, 0.0, 1.0)
        font = ImageFont.truetype(font_path, font_size)

        # x_offset = - x_offset
        y_offset = - y_offset

        num_frames = images.shape[0]
        for i in range(num_frames):
            frame_time = i / fps

            relevant_subtitles = [
                sub
                for sub in subtitles
                if sub["timestamp"][0] <= frame_time <= sub["timestamp"][1]
            ]

            if relevant_subtitles:
                frame = Image.fromarray((images[i].numpy() * 255).astype(np.uint8)) 
                draw = ImageDraw.Draw(frame)

                # 重置 y_offset
                current_y_offset = y_offset  

                for sub in relevant_subtitles:
                    text = sub["text"]
                    wrapped_text = self.wrap_text(text, language, max_words_per_line) 

                    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    # 计算居中后的坐标
                    center_x = width / 2 + x_offset
                    center_y = height / 2 + current_y_offset

                    # 计算左上角坐标
                    x = center_x - text_width / 2
                    y = center_y - text_height / 2

                    if subtitle_background_color:
                        bbox = draw.multiline_textbbox((x, y), wrapped_text, font=font)
                        bg_x0 = bbox[0]
                        bg_y0 = bbox[1] 
                        bg_x1 = bbox[2]  
                        bg_y1 = bbox[3] 
                        draw.rectangle((bg_x0, bg_y0, bg_x1, bg_y1), fill=subtitle_background_color)

                    draw.multiline_text((x, y), wrapped_text, font=font, fill=font_color)

                    # 更新 y 偏移量，以便下一条字幕显示在下方 (如果需要显示多条字幕)
                    current_y_offset += text_height + 10  
                    
                images[i] = torch.from_numpy(np.array(frame).astype(np.float32) / 255) 

        return (images,)
    def wrap_text(self, text, language, max_words_per_line):
        words = text.split()  
        wrapped_lines = []
        current_line = []
        if language == "类中文字符":
            for i in range(0, len(text), max_words_per_line):
                wrapped_lines.append(text[i:i + max_words_per_line].strip())
        else:
            for word in words:
                if len(current_line) < max_words_per_line:
                    current_line.append(word) 
                else:
                    wrapped_lines.append(" ".join(current_line)) 
                    current_line = [word]  

            if current_line:
                wrapped_lines.append(" ".join(current_line))

        return "\n".join(wrapped_lines)
