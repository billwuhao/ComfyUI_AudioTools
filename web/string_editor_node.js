// filename: ComfyUI/web/extensions/string_editor_persistent_tempfile_node.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Comfy.StringEditorPersistentTempFileNode",

	async nodeCreated(node) {
		if (node.constructor.comfyClass !== "StringEditorPersistentTempFileNode") {
			return;
		}
        console.log("StringEditorPersistentTempFileNode: nodeCreated", node.id);

		const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        // Removed justify-content, let natural flow happen
        // Removed height: 100%
        container.style.padding = "2px";
        container.style.margin = "0";
        container.style.gap = "2px"; // Smaller gap
        // --- Control Overflow ---
        container.style.overflow = "hidden"; // Prevent container itself from overflowing node bounds
        // -----------------------

		// const label = document.createElement("label");
        // label.textContent = "Edit Text & Continue (Persistent):";
        // label.style.fontSize = "0.8em";
        // label.style.color = "var(--descrip-text)";
        // label.style.marginBottom = "3px";
        // label.style.flexShrink = "0";
        // container.appendChild(label); // Label directly in container

		const textarea = document.createElement("textarea");
        textarea.rows = 50; // Keep large default rows
        textarea.placeholder = "Workflow paused. Edit text and click Continue.";
        // --- Control Textarea Size/Overflow ---
        // Let textarea be naturally sized by rows initially, but limit max height
        textarea.style.minHeight = "160px"; // Reasonable minimum
        textarea.style.maxHeight = "600px"; // Set a sensible maximum height (adjust as needed)
        textarea.style.overflowY = "auto"; // Add scrollbar if content exceeds max height
        textarea.style.resize = "vertical";
        textarea.style.padding = "1px";
        textarea.style.border = "1px solid var(--border-color)";
        textarea.style.backgroundColor = "var(--comfy-input-bg)";
        textarea.style.color = "var(--input-text)";
        textarea.style.width = "calc(100% - 1px)"; // Ensure it fits horizontally within padding
        textarea.disabled = true;
        node.stringEditorTextarea = textarea;
        container.appendChild(textarea); // Textarea directly in container

		const button = document.createElement("button");
        button.textContent = "Continue Workflow";
        button.style.padding = "3px 10px";
        button.style.marginTop = "5px";
        button.style.cursor = "pointer";
        button.style.border = "1px solid var(--border-color)";
        button.style.backgroundColor = "var(--comfy-input-bg)";
        button.style.color = "var(--input-text)";
        button.style.flexShrink = "0"; // Prevent shrinking
        button.style.borderRadius = "15px";
        button.disabled = true;
        node.stringEditorButton = button;
        container.appendChild(button); // Button directly in container

        const widget = node.addDOMWidget(`string_editor_persistent_widget_${node.id}`, "dom", container, {});

        // --- computeSize (Simpler - let content mostly decide) ---
        widget.computeSize = function(width) {
             // Estimate minimum based on non-textarea elements + small textarea amount
             const minHeight = 250; // Minimum needed for label, button, small textarea
             // Get the container's *current* scrollHeight as a proxy for desired height
             let contentHeight = container.scrollHeight;
             const maxHeight = 800; // Absolute max node height (adjust)

             // Use scrollHeight, but cap it and ensure minimum
             let targetHeight = Math.max(minHeight, contentHeight + 10); // Add padding
             targetHeight = Math.min(maxHeight, targetHeight); // Apply max height

             console.log(`[computeSize] Request W: ${width}, scrollH: ${container.scrollHeight}, Target H: ${targetHeight}`);
             return [width, Math.ceil(targetHeight)];
        }

        // --- Resize Observer (Still useful for manual drag) ---
        let resizeTimeout;
        new ResizeObserver(() => {
              clearTimeout(resizeTimeout);
              resizeTimeout = setTimeout(() => {
                  try {
                    // When textarea manually resizes, tell the node its overall size might need recalculating
                    node.computeSize(); // Trigger node's own size update based on widget
                    node.setDirtyCanvas(true, true);
                  } catch(e) { console.error("ResizeObserver error:", e); }
              }, 150);
        }).observe(textarea);


        // --- Button Click Handler (Keep TempFile logic) ---
        button.addEventListener("click", async () => {
            // ...(Same logic: call /signal_continue, then app.queuePrompt() on success)...
            const editedString = textarea.value; const nodeId = node.id;
            console.log(`StringEditorPersistent: 'Continue' clicked for node ${nodeId}.`);
            button.disabled = true; textarea.disabled = true;
            try {
                const response = await api.fetchApi("/string_editor/signal_continue", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ node_id: nodeId, edited_string: editedString }), });
                if (!response.ok) { const errorData = await response.json(); const errorMsg = `Error saving state: ${errorData.message || response.statusText} (Status: ${response.status})`; console.error(`StringEditorPersistent: ${errorMsg} for node ${nodeId}`); alert(errorMsg); button.textContent = "Continue Workflow"; button.disabled = false; textarea.disabled = false; }
                else { console.log(`StringEditorPersistent: State saved. Re-queueing prompt.`); app.queuePrompt(); }
            } catch (error) { console.error(`StringEditorPersistent: Network error during continue for node ${nodeId}:`, error); alert(`Network error: ${error.message}`); button.textContent = "Continue Workflow"; button.disabled = false; textarea.disabled = false; }
        });

		// Initial size calculation
        node.computeSize();
        node.setDirtyCanvas(true, true);

	}, // end nodeCreated

	async setup(app) {
        // ...(setup listener remains the same - listens for string_editor_persistent_tempfile_display)...
        console.log("StringEditorPersistentTempFileNode: Setup.");
		api.addEventListener("string_editor_persistent_tempfile_display", (event) => { // Listen for NEW event name
			const { node_id, string_to_edit } = event.detail;
            console.log(`StringEditorPersistent: Received 'display' event for node ${node_id}`);
			const node = app.graph.getNodeById(node_id);
			if (node && node.stringEditorTextarea && node.stringEditorButton) {
				console.log(`StringEditorPersistent: Updating UI for node ${node_id}`);
				node.stringEditorTextarea.value = string_to_edit;
                node.stringEditorTextarea.disabled = false;
                node.stringEditorButton.textContent = "Continue Workflow";
				node.stringEditorButton.disabled = false;
                node.computeSize(); // Recalculate size after content update
                node.setDirtyCanvas(true, true);
			} else { console.warn(`StringEditorPersistent: Could not find node/widgets for ID ${node_id}`); }
		});
	} // end setup
});

console.log("StringEditorPersistentTempFileNode: Extension registered.");