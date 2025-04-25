import uuid
from pathlib import Path
from aiohttp import web
import folder_paths
from server import PromptServer
from comfy_execution.graph import ExecutionBlocker
import json
import time

TEMP_SUBFOLDER = "string-editor-persistent-state"

def get_persistent_temp_file_path(node_id: str) -> Path:
    temp_dir = Path(folder_paths.get_temp_directory())
    state_dir = temp_dir / TEMP_SUBFOLDER
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"state_{node_id}.json"

if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
    routes = PromptServer.instance.routes
    @routes.post('/string_editor/signal_continue')
    async def signal_continue_handler(request: web.Request) -> web.Response:
        try:
            if request.content_type != 'application/json':
                return web.json_response({"status": "error", "message": "Invalid content type"}, status=400)
            data = await request.json()
            node_id = data.get('node_id')
            edited_string = data.get('edited_string')
            if not node_id:
                return web.json_response({"status": "error", "message": "node_id missing"}, status=400)
            if edited_string is None:
                edited_string = ""
            temp_file = get_persistent_temp_file_path(node_id)
            try:
                state_data = {"edited_string": edited_string, "last_modified": time.time()}
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(state_data, f)
                return web.json_response({"status": "ok", "message": "State saved, ready for re-queue."})
            except IOError as e:
                return web.json_response({"status": "error", "message": f"Failed to write state file: {e}"}, status=500)
        except json.JSONDecodeError:
            return web.json_response({"status": "error", "message": "Invalid JSON format"}, status=400)
        except Exception:
            return web.json_response({"status": "error", "message": "Internal server error"}, status=500)

class StringEditorPersistentTempFileNode:
    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return str(uuid.uuid4())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"forceInput": True}),
                "pause": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("edited_string",)
    FUNCTION = "edit_or_check_and_output"
    CATEGORY = "🎤MW/MW-Audio-Tools"
    OUTPUT_NODE = True

    def edit_or_check_and_output(self,
                    input_string: str,
                    pause: bool,
                    unique_id: str,
                    prompt=None,
                    extra_pnginfo=None,
                    **kwargs
                    ):
        node_id = unique_id
        temp_file = get_persistent_temp_file_path(node_id)
        # --- REMOVE class variable self.current_input_string - it's not needed ---
        # if self.current_input_string is None:
        #     self.current_input_string = input_string
        # --------------------------------------------------------------------

        # Initialize string_to_display with the CURRENT input
        current_input_string = str(input_string)
        string_to_display = current_input_string # <<< Start with current input

        should_block = True
        output_string = None # This will hold the value read from file if successful

        try:
            if temp_file.exists():
                with open(temp_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    # Read the value from the file
                    value_from_file = state_data.get("edited_string", current_input_string) # Fallback needed?
                    # *** THIS IS THE KEY CHANGE ***
                    string_to_display = value_from_file # <<< Update string_to_display with file content
                    output_string = value_from_file   # <<< Store the value intended for output
                    # *****************************
                try:
                    temp_file.unlink()
                    should_block = False
                except OSError as e:
                    # If delete fails, we must block to avoid losing the edit
                    print(f"[ERROR] Node {node_id}: Failed to delete state file {temp_file}: {e}. Forcing block.") # Added print/log
                    should_block = True
                    output_string = None # Cannot guarantee output if delete failed
                    string_to_display = value_from_file # Still display what we read before delete failed
            else:
                # File doesn't exist (first run or after successful continue)
                string_to_display = current_input_string # Use current input
                state_data = {"edited_string": string_to_display, "last_modified": time.time()}
                try:
                    # Create the file to prepare for blocking
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(state_data, f)
                except IOError as e:
                    print(f"[ERROR] Node {node_id}: Failed to create initial state file {temp_file}: {e}") # Added print/log
                    pass # Proceed to block anyway
                should_block = True
        except (IOError, json.JSONDecodeError) as e:
            print(f"[ERROR] Node {node_id}: Error processing state file {temp_file}: {e}. Blocking.") # Added print/log
            string_to_display = current_input_string # Fallback display
            should_block = True
            output_string = None
            try:
                temp_file.unlink() # Clean up potentially corrupt file
            except OSError:
                pass

        # Determine if UI should be enabled
        ui_should_be_enabled = pause and should_block

        # Always send the UI update with the determined string_to_display
        if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
            PromptServer.instance.send_sync("string_editor_persistent_tempfile_display", {
                "node_id": node_id,
                "string_to_edit": string_to_display, # <<< Send the correct value
                "enable_ui": ui_should_be_enabled
            })
            # Apply delay only if pause=True and we are *not* blocking (i.e., file processed)
            if pause and not should_block:
                time.sleep(0.02)

        # Final return logic
        if not pause:
            PromptServer.instance.send_sync("string_editor_persistent_tempfile_display", {
                "node_id": node_id,
                "string_to_edit": input_string, # <<< Send the correct value
                "enable_ui": ui_should_be_enabled
            })
            time.sleep(0.02)
            # If pause is off, return the current input string directly
            return {"result": (input_string,)}
        else:
            # If pause is on, return Blocker or the value read from the file
            if should_block:
                return {"result": (ExecutionBlocker(None),)}
            else:
                # Ensure we have a valid output string (should be from file)
                final_output = output_string if output_string is not None else current_input_string # Fallback just in case
                return {"result": (final_output,)}