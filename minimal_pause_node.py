import uuid
from pathlib import Path
from aiohttp import web
import hashlib # For robust hashing of data
import torch   # For hashing torch tensors
import json
import folder_paths 
from server import PromptServer
from comfy_execution.graph import ExecutionBlocker
TEMP_SUBFOLDER = "minimal-pause-signals" 

# Structure: { node_id: { 'data': any, 'data_hash': str } }
PAUSED_DATA = {} 

class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False

any_type = AlwaysEqualProxy("*")

# --- Helper functions ---
def get_signal_file_path(node_id: str) -> Path:
    """Gets the path for this node's signal file."""
    temp_dir = Path(folder_paths.get_temp_directory())
    signal_dir = temp_dir / TEMP_SUBFOLDER
    signal_dir.mkdir(parents=True, exist_ok=True)
    return signal_dir / f"continue_{node_id}.signal"

def compute_data_hash(data) -> str:
    """
    Computes a robust hash for the input data.
    Prioritizes specific handling for common ComfyUI types like torch.Tensor.
    Falls back to a string representation hash for other types.
    Returns 'NOT_HASHABLE' if hashing fails significantly.
    """
    try:
        if isinstance(data, torch.Tensor):
            # For torch.Tensor, convert to numpy array bytes for robust hashing
            # This is generally the most reliable way to hash tensor content.
            # Handle empty tensors to avoid tobytes() errors if data.numel() == 0
            if data.numel() == 0:
                return hashlib.sha256(f"EMPTY_TENSOR_SHAPE:{data.shape}_DTYPE:{data.dtype}".encode('utf-8')).hexdigest()
            else:
                return hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest()
        elif isinstance(data, (str, int, float, bool)):
            # Basic types can be directly converted to string and encoded
            return hashlib.sha256(str(data).encode('utf-8')).hexdigest()
        elif isinstance(data, (list, tuple)):
            # For lists/tuples, convert to a consistent JSON string representation
            # This handles nested structures and ensures consistent order for keys if dicts are nested
            # Note: This requires all elements to be JSON serializable.
            return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        elif isinstance(data, dict):
            # For dictionaries, convert to a consistent JSON string representation
            return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode('utf-8')).hexdigest()
        else:
            # Fallback for other arbitrary objects: try to use repr() or str()
            # Be aware: repr() can include memory addresses, making hashes inconsistent across runs.
            # str() might not be unique enough for complex objects.
            return hashlib.sha256(repr(data).encode('utf-8')).hexdigest()
            
    except Exception as e:
        # Catch any unexpected errors during hashing and log them
        print(f"[WARN] Failed to compute hash for data type {type(data)}: {e}. Returning 'NOT_HASHABLE'.")
        return 'NOT_HASHABLE'
    
# --- HTTP Route ---
if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
    routes = PromptServer.instance.routes
    @routes.post('/minimal_pause/signal_continue')
    async def signal_continue_handler(request: web.Request) -> web.Response:
        """Creates the signal file to indicate continuation."""
        try:
            if request.content_type != 'application/json':
                return web.json_response({"status": "error", "message": "Invalid content type"}, status=400)
            data = await request.json()
            node_id = data.get('node_id')

            if not node_id:
                return web.json_response({"status": "error", "message": "node_id missing"}, status=400)

            signal_file = get_signal_file_path(node_id)
            # --- START DEBUG HTTP ---
            # --- END DEBUG HTTP ---
            try:
                signal_file.touch(exist_ok=True) # Create the signal file
                # --- START DEBUG HTTP ---
                
                # --- END DEBUG HTTP ---
                
                if node_id not in PAUSED_DATA:
                    print(f"[INFO HTTP Handler] Node {node_id}: Not found in PAUSED_DATA (might have restarted or never paused in memory).")
                else:
                    print(f"[INFO HTTP Handler] Node {node_id}: Found in PAUSED_DATA when signal received.")

                return web.json_response({"status": "ok", "message": "Continue signal received and file created."})
            except IOError as e:
                print(f"[ERROR HTTP Handler] Node {node_id}: Error creating signal file: {e}")
                return web.json_response({"status": "error", "message": f"Failed to create signal file: {e}"}, status=500)

        except Exception as e:
            print(f"[ERROR HTTP Handler] General error in signal_continue_handler: {e}")
            return web.json_response({"status": "error", "message": "Internal server error"}, status=500)

            
class MinimalPauseNode:
    """A node that pauses workflow execution based on a filesystem signal, triggered by an HTTP signal,
    or passes through if upstream data is unchanged."""

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # A unique ID for each execution to ensure UI update, but not for changing internal logic
        return str(uuid.uuid4())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { "any": (any_type, {}), },
            "hidden": { "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID", },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any",)
    FUNCTION = "pause_or_continue"
    CATEGORY = "🎤MW/MW-Audio-Tools"
    OUTPUT_NODE = True

    def pause_or_continue(self, any, unique_id: str, prompt=None, extra_pnginfo=None, **kwargs):
        node_id = unique_id
        signal_file = get_signal_file_path(node_id) 

        current_data_hash = compute_data_hash(any)
        stored_data_hash = PAUSED_DATA.get(node_id, {}).get('data_hash')
        should_continue_from_file = signal_file.exists()
        if should_continue_from_file:
            # Case 1: User explicitly clicked "Continue" via HTTP signal.
            # This is the strongest signal; always resume and pass data.
            resumed_data = None
            if node_id in PAUSED_DATA:
                resumed_data = PAUSED_DATA[node_id]['data']
                print(f"[INFO Node {node_id}] Resuming (explicit continue) with original stored data from PAUSED_DATA.")
            else:
                # Fallback: If PAUSED_DATA cleared, use the current input `any`.
                resumed_data = any
                print(f"[WARN Node {node_id}] Resuming (explicit continue) - in-memory data lost. Using current input `any`.")

            # Clean up the signal file and in-memory data
            try:
                if signal_file.exists(): 
                    signal_file.unlink()
                    print(f"[INFO Node {node_id}] Signal file unlinked.")
                else:
                    print(f"[WARN Node {node_id}] Signal file not found during unlink attempt.")
            except OSError as e:
                print(f"[ERROR Node {node_id}] Error deleting signal file on resume: {e}")
            
            if node_id in PAUSED_DATA:
                del PAUSED_DATA[node_id]
                print(f"[INFO Node {node_id}] PAUSED_DATA entry deleted.")
            return {"result": (resumed_data,)}

        else: # Case 2: No explicit continue signal. Decide based on data change.
            
            if node_id not in PAUSED_DATA:
                # Case 2a: First time hitting this node in this prompt execution, AND no signal file.
                # This means it should pause. Store current data and its hash.
                PAUSED_DATA[node_id] = { 'data': any, 'data_hash': current_data_hash }
                print(f"[INFO Node {node_id}] Pausing for the first time. Storing data and hash: {current_data_hash}. PAUSED_DATA[{node_id}] now present.")
                if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                    PromptServer.instance.send_sync("minimal_pause_enable_button", { "node_id": node_id })
                
                return {"result": (ExecutionBlocker(None),)}
            else:
                # Case 2b: Node was previously paused (node_id is in PAUSED_DATA), and no signal file.
                # Check if the input data has changed since last pause.
                if current_data_hash != stored_data_hash:
                    # Input data has changed. We need to re-pause with the new data.
                    PAUSED_DATA[node_id] = { 'data': any, 'data_hash': current_data_hash } # Update stored data and hash
                    print(f"[INFO Node {node_id}] Input data CHANGED ({current_data_hash} != {stored_data_hash}). Re-pausing with new data.")
                    if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                        PromptServer.instance.send_sync("minimal_pause_enable_button", { "node_id": node_id })
                    
                    return {"result": (ExecutionBlocker(None),)}
                else:
                    # Input data has NOT changed. This node should "pass through" its previously stored output.
                    # This is crucial for allowing downstream nodes to execute when upstream is stable.
                    resumed_data = PAUSED_DATA[node_id]['data']
                    print(f"[INFO Node {node_id}] Input data UNCHANGED. Passing through previous output to allow downstream execution.")
                    if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                        # Re-send enable signal in case UI missed it or state changed (still paused, but passing through)
                        PromptServer.instance.send_sync("minimal_pause_enable_button", { "node_id": node_id })
                    
                    return {"result": (resumed_data,)}