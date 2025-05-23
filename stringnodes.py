import uuid
from pathlib import Path
from aiohttp import web

import folder_paths 
from server import PromptServer
from comfy_execution.graph import ExecutionBlocker
import hashlib
import json # For consistent hashing of collections if needed

# --- StringDisplayNode (NO CHANGES needed for pause/continue logic) ---
# This node is purely for displaying and passing string.
class StringDisplayNode:
    @classmethod
    def IS_CHANGED(cls, text, **kwargs):
        # Re-run only when the input string actually changes
        m = hashlib.sha256()
        if text is None:
            text = ""
        m.update(text.encode('utf-8')) # Encode string for hashing
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

    OUTPUT_NODE = True # This is an output node to display info

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "display_and_output"
    CATEGORY = "🎤MW/MW-Audio-Tools" # Category

    def display_and_output(self, text: str, unique_id: str):
        string_to_display = str(text)

        if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
            PromptServer.instance.send_sync("string_display_update", {
                "node_id": unique_id,
                "string_to_display": string_to_display,
            })

        return {"result": (string_to_display,)}

# --- StringPauseGateNode (MODIFIED for "serial-friendly" pause/continue) ---

TEMP_SUBFOLDER = "string-pause-signals" 

# Global state for storing paused node data
# Structure: { node_id: { 'text': str, 'text_hash': str, 'split_tag': str, 'paste_edit_text': str, 'merge_text': bool } }
PAUSED_STRING_DATA = {} 

# --- Helper functions ---
def get_signal_file_path(node_id: str) -> Path:
    """Gets the path for this node's signal file."""
    temp_dir = Path(folder_paths.get_temp_directory())
    signal_dir = temp_dir / TEMP_SUBFOLDER
    signal_dir.mkdir(parents=True, exist_ok=True)
    return signal_dir / f"continue_{node_id}.signal"

def compute_string_hash(s: str) -> str:
    """Computes a hash for the input string."""
    if s is None:
        s = ""
    m = hashlib.sha256()
    m.update(s.encode('utf-8'))
    return m.hexdigest()

# This helper will hash all relevant inputs to the node's state
def compute_node_input_hash(text, split_tag, paste_edit_text, merge_text) -> str:
    # Convert all relevant inputs to a consistent string representation for hashing
    # Use json.dumps to handle various types consistently, especially bool
    data_to_hash = {
        "text": text,
        "split_tag": split_tag,
        "paste_edit_text": paste_edit_text,
        "merge_text": merge_text
    }
    # sort_keys=True ensures consistent order for hashing dictionaries
    return hashlib.sha256(json.dumps(data_to_hash, sort_keys=True).encode('utf-8')).hexdigest()
# --- HTTP Route ---
if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
    routes = PromptServer.instance.routes
    @routes.post('/string_pause/signal_continue')
    async def signal_continue_handler(request: web.Request) -> web.Response:
        """Creates the signal file to allow the node to continue."""
        try:
            if request.content_type != 'application/json':
                return web.json_response({"status": "error", "message": "Invalid content type"}, status=400)
            data = await request.json()
            node_id = data.get('node_id')

            if not node_id:
                return web.json_response({"status": "error", "message": "node_id missing"}, status=400)

            signal_file = get_signal_file_path(node_id)
            try:
                signal_file.touch(exist_ok=True) 
                return web.json_response({"status": "ok", "message": "Continue signal sent."})
            except IOError as e:
                print(f"[ERROR HTTP Handler] Node {node_id}: Error creating signal file: {e}")
                return web.json_response({"status": "error", "message": f"Failed to create signal file: {e}"}, status=500)

        except Exception as e:
            print(f"[ERROR HTTP Handler] General error in signal_continue_handler: {e}")
            return web.json_response({"status": "error", "message": "Internal server error"}, status=500)
class StringPauseGateNode:
    """A node that pauses workflow execution until continued via UI, or passes through if input unchanged."""

    # Force execution so the file check and data hash logic always runs
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return str(uuid.uuid4())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}), 
             },
            "optional": {
                "split_tag": ("STRING", {"default": "",}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text_or_split_1", "split_2", "edited_text",)
    FUNCTION = "pause_or_continue"
    CATEGORY = "🎤MW/MW-Audio-Tools" 
    OUTPUT_NODE = False

    def pause_or_continue(self, text, unique_id: str, split_tag="", paste_edit_text="", merge_text=False, enable_pause_gate=True, prompt=None, extra_pnginfo=None, **kwargs):
        node_id = unique_id
        signal_file = get_signal_file_path(node_id)
        
        # Calculate current hash based on all relevant inputs
        current_input_hash = compute_node_input_hash(text, split_tag, paste_edit_text, merge_text)
        stored_input_hash = PAUSED_STRING_DATA.get(node_id, {}).get('input_hash')
        # Pre-calculate outputs, which will be returned if not pausing
        str_1, str_2, str_3 = text, "", paste_edit_text
        if merge_text:
            str_1 = text + paste_edit_text
            str_3 = text + paste_edit_text
        elif split_tag.strip() and text.find(split_tag.strip()) >= 0: # Use >=0 for 'found'
            parts = text.split(split_tag, 1)
            str_1, str_2 = parts[0], parts[1]
            # str_3 remains paste_edit_text if not merged

        # If pause gate is disabled, just pass through
        if not enable_pause_gate:
            print(f"[INFO Node {node_id}] Pause gate is disabled. Passing through outputs.")
            return {"result": (str_1, str_2, str_3)}
        
        # --- Pause Gate Logic ---
        should_continue_from_file = signal_file.exists()
        if should_continue_from_file:
            # Case 1: User explicitly clicked "Continue" via HTTP signal.
            # This node takes priority and continues.
            print(f"[INFO Node {node_id}] Explicit continue signal received for THIS node.")
            
            # Clean up signal file
            try:
                if signal_file.exists(): 
                    signal_file.unlink()
                    print(f"[INFO Node {node_id}] Signal file unlinked.")
            except OSError as e:
                print(f"[ERROR Node {node_id}] Error deleting signal file: {e}")
            
            # Clear stored state for this node as it's completing its pause cycle
            if node_id in PAUSED_STRING_DATA:
                del PAUSED_STRING_DATA[node_id]
                print(f"[INFO Node {node_id}] PAUSED_STRING_DATA entry deleted.")
            return {"result": (str_1, str_2, str_3)}

        else: # Case 2: No explicit continue signal for THIS node.
            if node_id not in PAUSED_STRING_DATA:
                # Case 2a: First time hitting this node in this prompt execution, AND no signal file.
                # This means it should pause and store its current input state.
                PAUSED_STRING_DATA[node_id] = { 
                    'input_hash': current_input_hash,
                    'output_1': str_1,
                    'output_2': str_2,
                    'output_3': str_3,
                }
                print(f"[INFO Node {node_id}] Pausing for the first time. Storing inputs and outputs. PAUSED_STRING_DATA[{node_id}] now present.")
                if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                    PromptServer.instance.send_sync("string_pause_enable_button", { "node_id": node_id })

                return {"result": (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))}
            
            else:
                # Case 2b: Node was previously paused, and no signal file.
                # Check if the relevant input data (text, split_tag, etc.) has changed since last pause.
                if current_input_hash != stored_input_hash:
                    # Input data has changed. We need to re-pause with the new data.
                    PAUSED_STRING_DATA[node_id] = { 
                        'input_hash': current_input_hash,
                        'output_1': str_1,
                        'output_2': str_2,
                        'output_3': str_3,
                    }
                    print(f"[INFO Node {node_id}] Input data CHANGED ({current_input_hash} != {stored_input_hash}). Re-pausing with new data.")
                    if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                        PromptServer.instance.send_sync("string_pause_enable_button", { "node_id": node_id })

                    return {"result": (ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None))}
                else:
                    # Input data has NOT changed. This node should "pass through" its previously stored output.
                    # This is crucial for allowing downstream nodes in a serial chain to execute.
                    resumed_str_1 = PAUSED_STRING_DATA[node_id]['output_1']
                    resumed_str_2 = PAUSED_STRING_DATA[node_id]['output_2']
                    resumed_str_3 = PAUSED_STRING_DATA[node_id]['output_3']

                    print(f"[INFO Node {node_id}] Input data UNCHANGED. Passing through previous stored outputs.")
                    if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                        PromptServer.instance.send_sync("string_pause_enable_button", { "node_id": node_id }) # Keep button enabled

                    return {"result": (resumed_str_1, resumed_str_2, resumed_str_3)}