// filename: ComfyUI/web/extensions/string_viewer_node.js

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Register the extension
app.registerExtension({
	name: "Comfy.StringViewerNode", // Unique extension name (可以保留，但与实际 comfyClass 不同)

	async nodeCreated(node) {
		// Check if this is our node type - **CORRECTED comfyClass CHECK**
		// Based on latest logs, the comfyClass is "StringDisplayNode"
		if (node.constructor.comfyClass !== "StringDisplayNode") { // <<< CHANGED BACK TO "StringDisplayNode"
            console.log(`StringDisplayNode: Not a StringDisplayNode, skipping DOM widget creation for node ${node.id} with comfyClass ${node.constructor.comfyClass}`);
			return;
		}
        // ** THIS LOG SHOULD NOW APPEAR **
        console.log(`StringDisplayNode: nodeCreated executed for node ${node.id} (${node.constructor.comfyClass})`);

		// --- Create Container Div ---
		const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.padding = "4px";
        container.style.margin = "0";
        container.style.gap = "0px";
        container.style.overflow = "hidden";

        container.style.width = "100%";
        container.style.height = "100%";
        container.style.boxSizing = "border-box";

        // ** DEBUG BORDER **
        container.style.border = "1px dashed #388E3C"; // Visualize container area
        console.log("StringDisplayNode: Created container div");

		// --- Create Textarea ---
		const textarea = document.createElement("textarea");
        textarea.placeholder = "Waiting for string input...";
        textarea.value = "";
        textarea.readOnly = true;

        textarea.style.flexGrow = 1;
        textarea.style.width = "100%";
        textarea.style.height = "100%";

        textarea.style.boxSizing = "border-box";
        textarea.style.resize = "none";

        textarea.style.padding = "4px";
        textarea.style.border = "1px solid var(--border-color)";
        textarea.style.backgroundColor = "var(--comfy-input-bg)";
        textarea.style.color = "var(--input-text)";
        textarea.style.whiteSpace = "pre-wrap";
        textarea.style.wordBreak = "break-word";

        // ** DEBUG BORDER **
        // textarea.style.border = "1px dashed blue"; // Visualize textarea area

        console.log("StringDisplayNode: Created textarea");

		// --- Append Textarea to Container ---
        container.appendChild(textarea);
        console.log("StringDisplayNode: Appended textarea to container");

        // --- Add Container as DOM Widget ---
        const widget = node.addDOMWidget("string_viewer_widget", "dom", container, {
             computeSize: function(width) {
                 const containerVerticalPadding = 8;
                 const textareaVerticalSpace = 8 + 2;
                 const estimatedMinTextareaContentHeight = 3 * 14;
                 const estimatedMinTextareaHeight = estimatedMinTextareaContentHeight + textareaVerticalSpace;
                 const defaultWidgetHeight = containerVerticalPadding + estimatedMinTextareaHeight;
                 const minNodeHeight = 150;

                 const targetHeight = Math.max(minNodeHeight, defaultWidgetHeight);

                 console.log(`[computeSize] W: ${width}, TargetH: ${targetHeight}`);

                return [width, targetHeight];
            }
        });

        console.log("StringDisplayNode: Added DOM widget");

        node.stringDisplayTextarea = textarea; // Store reference

        node.computeSize();
        node.setDirtyCanvas(true, true);

        console.log("StringDisplayNode: Computed size and set dirty");

	},

	async setup(app) {
        console.log("StringDisplayNode: Setup.");

		api.addEventListener("string_display_update", (event) => {
			const { node_id, string_to_display } = event.detail;
            // console.log(`StringDisplayNode: Received 'display' event for node ${node_id}`);

			const node = app.graph.getNodeById(node_id);

			if (node && node.stringDisplayTextarea) {
				// console.log(`StringDisplayNode: Updating UI for node ${node_id}`);
				node.stringDisplayTextarea.value = string_to_display;
                node.stringDisplayTextarea.placeholder = "";

                node.computeSize();
                node.setDirtyCanvas(true, true);
			} else {
				console.warn(`StringDisplayNode: Could not find node/widgets (textarea property missing?) for ID ${node_id}. Event received before node created or node deleted?`);
			}
		});
	}
});

console.log("StringDisplayNode: Extension registered.");