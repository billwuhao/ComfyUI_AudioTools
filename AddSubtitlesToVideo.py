import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ast


def parse_timestamp(timestamp_str):
    # Remove leading/trailing whitespace and brackets
    timestamp_str = timestamp_str.strip()
    if not (timestamp_str.startswith('[') and timestamp_str.endswith(']')):
        # Handle cases where brackets might be missing, though the format suggests they are present
        # If strict format is required, could raise an error
        print(f"Warning: Timestamp string missing brackets: {timestamp_str}")
        inner_str = timestamp_str
    else:
        inner_str = timestamp_str[1:-1] # Remove '[' and ']'

    try:
        # Split into minutes and seconds.milliseconds
        parts = inner_str.split(':')
        if len(parts) != 2:
            print(f"Warning: Timestamp format incorrect (expecting MM:SS.mmm): {timestamp_str}")
            # Attempt to parse as seconds if possible
            try:
                return float(inner_str)
            except ValueError:
                print(f"Error parsing timestamp: {timestamp_str}. Returning 0.0.")
                return 0.0

        minutes_str = parts[0]
        sec_milli_str = parts[1]

        # Split seconds.milliseconds into seconds and milliseconds
        sec_parts = sec_milli_str.split('.')
        seconds_str = sec_parts[0]
        milliseconds_str = sec_parts[1] if len(sec_parts) > 1 else '0' # Handle missing .mmm

        # Convert to numbers
        minutes = int(minutes_str)
        seconds = int(seconds_str)
        # Pad milliseconds with zeros if necessary (e.g., '5' -> '500')
        milliseconds_str = milliseconds_str.ljust(3, '0') 
        milliseconds = int(milliseconds_str[:3]) # Take only the first 3 digits

        total_seconds = float(minutes * 60 + seconds + milliseconds / 1000.0)
        return total_seconds

    except ValueError:
        print(f"Error parsing timestamp: {timestamp_str}. Returning 0.0.")
        return 0.0
    except Exception as e:
        print(f"An unexpected error occurred while parsing timestamp {timestamp_str}: {e}. Returning 0.0.")
        return 0.0


def convert_subtitle_to_list(subtitle_string, fallback_duration, max_duration_of_segment):
    lines = subtitle_string.strip().split('\n')

    parsed_entries = []
    
    for line in lines:
        line = line.strip()
        if not line: # Skip empty lines
            continue

        # Find the end of the timestamp part (the closing bracket)
        end_bracket_index = line.find(']')

        if end_bracket_index == -1 or not line.startswith('['):
            # Line doesn't look like a subtitle line with timestamp
            print(f"Warning: Skipping line that doesn't match [MM:SS.mmm]Text format: {line}")
            continue

        # Extract timestamp string and text
        timestamp_str = line[:end_bracket_index + 1] # Includes the brackets
        text = line[end_bracket_index + 1:]

        # Parse the timestamp string into seconds (float)
        start_time = parse_timestamp(timestamp_str)

        parsed_entries.append({'start_time': start_time, 'text': text})

    result_list = []
    num_entries = len(parsed_entries)

    for i in range(num_entries):
        start_time = parsed_entries[i]['start_time']
        text = parsed_entries[i]['text']

        end_time = None

        # Rule: End time is the start time of the *next* entry minus 0.2 seconds
        if i < num_entries - 1:
            next_start_time = parsed_entries[i+1]['start_time']
            end_time = next_start_time - 0.05
        else:
            fallback_duration = fallback_duration
            end_time = start_time + fallback_duration

        minimal_duration = 0.01 # A very small duration to avoid end_time <= start_time
        if end_time <= start_time:
             end_time = start_time + minimal_duration
        if end_time - start_time > max_duration_of_segment:
            end_time = round(start_time + max_duration_of_segment, 2)

        # Append to result list
        result_list.append({'timestamp': [start_time, round(end_time,2)], 'text': text})

    return result_list


class AddSubtitlesToTensor:
    """
    ComfyUI 节点，用于将字幕添加到视频帧的张量中。
    """
    @classmethod
    def INPUT_TYPES(s):
        import os
        font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ChironGoRoundTC-600SB.ttf")
        return {
            "required": {
                "images": ("IMAGE",), 
                "language": (["类中文字符", "English_word"], {"default": "类中文字符"}), 
                "max_words_per_line": (
                    "INT", {"default": 12, "min": 1, "max": 30}),  
                "font_path": (
                    "STRING", {"default": font_path}),  
                "font_size": ("INT", {"default": 30, "min": 1, "max": 200}),  
                "font_color": ("STRING", {"default": "black", "tooltip": "HTML color or Color name"}),  
                "subtitle_background_color": (
                    "STRING", {"default": "#B8860B", "tooltip": "HTML color or Color name. None: Do not add background color"}),
                "y_offset": (
                    "INT", {"default": 0, "min": -1000, "max": 1000, "step": 5}),
                "x_offset": (
                    "INT", {"default": 0, "min": -1000, "max": 1000, "step": 5}),
                "width": ("INT", {"default": 1024}), 
                "height": ("INT", {"default": 1024}),  
                "fps": ("FLOAT", {"default": 20, "min": 1, "max": 240}),
            },
            "optional": {
                "json_text": ("STRING",  {"forceInput": True}), 
                "subtitle_text": ("STRING",  {"forceInput": True}),
                "max_duration_of_segment": ("FLOAT", {"default": 6, "min": 0.1, "max": 30}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "add_subtitles"

    CATEGORY = "🎤MW/MW-Audio-Tools"  

    def add_subtitles(self, 
                      images, 
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
                      subtitle_text=None,
                      max_duration_of_segment=6,
                      json_text=None,
                      ):

        if subtitle_background_color == "None":
            subtitle_background_color = None

        assert subtitle_text or json_text, "Either subtitle_text or json_text must be provided."
        fallback_duration = len(images) / fps - 0.2
        if subtitle_text: 
            subtitles = convert_subtitle_to_list(subtitle_text, fallback_duration, max_duration_of_segment)
        else:
            subtitles = ast.literal_eval(json_text)

        try:
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
