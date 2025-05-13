# filename: ComfyUI/custom_nodes/string_display_node.py

import uuid
from pathlib import Path
from aiohttp import web

import folder_paths 
from server import PromptServer
from comfy_execution.graph import ExecutionBlocker
import hashlib

class StringDisplayNode:
    @classmethod
    def IS_CHANGED(cls, input_string, **kwargs):
        # Re-run only when the input string actually changes
        m = hashlib.sha256()
        m.update(input_string.encode('utf-8')) # Encode string for hashing
        return m.digest().hex()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}), # Allow multiline input
            },
            "hidden": {
                # Keep unique_id to link the node instance to the UI widget
                "unique_id": "UNIQUE_ID",
            },
        }

    # Node is not an output node
    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display_and_output"
    CATEGORY = "🎤MW/MW-Audio-Tools" # Category

    def display_and_output(self,
                           text: str,
                           unique_id: str):
        # Ensure text is a string
        string_to_display = str(text)

        # Send the string to the frontend to update the UI widget
        if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
            PromptServer.instance.send_sync("string_display_update", {
                "node_id": unique_id,
                "string_to_display": string_to_display,
            })

        # Immediately return the input string as the output
        return {"result": (string_to_display,)}


TEMP_SUBFOLDER = "string-pause-signals" 

# --- Helper to get signal file path ---
def get_signal_file_path(node_id: str) -> Path:
    """Gets the path for this node's signal file."""
    temp_dir = Path(folder_paths.get_temp_directory())
    signal_dir = temp_dir / TEMP_SUBFOLDER
    signal_dir.mkdir(parents=True, exist_ok=True)
    # File existence signals "continue"
    return signal_dir / f"continue_{node_id}.signal"

# --- HTTP Route ---
if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
    routes = PromptServer.instance.routes
    @routes.post('/string_pause/signal_continue')
    async def signal_continue_handler(request: web.Request) -> web.Response:
        """Creates the signal file to allow the node to continue."""
        try:
            # No JSON needed, just check for node_id in query or form data? Let's use JSON.
            if request.content_type != 'application/json':
                return web.json_response({"status": "error", "message": "Invalid content type"}, status=400)
            data = await request.json()
            node_id = data.get('node_id')

            if not node_id:
                return web.json_response({"status": "error", "message": "node_id missing"}, status=400)

            signal_file = get_signal_file_path(node_id)

            try:
                # Create an empty file to signal continuation
                signal_file.touch(exist_ok=True) # Create if doesn't exist, update timestamp if does

                return web.json_response({"status": "ok", "message": "Continue signal sent."})
            except IOError as e:

                return web.json_response({"status": "error", "message": f"Failed to create signal file: {e}"}, status=500)
        except Exception as e:

            return web.json_response({"status": "error", "message": "Internal server error"}, status=500)


class StringPauseGateNode:
    """A node that simply pauses workflow execution until continued via UI."""

    # Force execution so the file check always runs
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return str(uuid.uuid4())

    @classmethod
    def INPUT_TYPES(cls):
        # Takes any input type just to allow connection, but doesn't use it
        return {
            "required": {
                 "text": ("STRING", {"forceInput": True}), 
             },
            "optional": {
                 "paste_edit_text": ("STRING", {"multiline": True, "default": "",}),
                 "merge_text": ("BOOLEAN", {"default": False}),
                 "enable_pause_gate": ("BOOLEAN", {"default": True}),
             },
            "hidden": {
                 "prompt": "PROMPT",
                 "extra_pnginfo": "EXTRA_PNGINFO",
                 "unique_id": "UNIQUE_ID",
            },
        }

    # Returns the same type it received
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text", "edited_text",)
    FUNCTION = "pause_or_continue"
    CATEGORY = "🎤MW/MW-Audio-Tools" # Simple category
    OUTPUT_NODE = False

    def pause_or_continue(self, text, paste_edit_text, merge_text, enable_pause_gate, unique_id: str, prompt=None, extra_pnginfo=None, **kwargs):
        node_id = unique_id

        signal_file = get_signal_file_path(node_id)

        if merge_text:
            text = paste_edit_text = text + paste_edit_text

        if not enable_pause_gate:
            return {"result": (text, paste_edit_text)}
        else:
            if signal_file.exists():
                # Signal file exists - User clicked Continue
                try:
                    signal_file.unlink() # Delete the signal file to reset for next time
                except OSError as e:
                    print(f"Failed to delete signal file: {e}")
                # Return the text data
                return {"result": (text, paste_edit_text)}
            else:
                # Send message to UI to enable the button
                if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                    PromptServer.instance.send_sync("string_pause_enable_button", { "node_id": node_id })
                # Return ExecutionBlocker for the text output
                return {"result": (ExecutionBlocker(None),)}
