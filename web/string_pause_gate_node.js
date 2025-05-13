// filename: ComfyUI/web/extensions/minimal_pause_node.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Comfy.StringPauseGateNode",

	async nodeCreated(node) {
		if (node.constructor.comfyClass !== "StringPauseGateNode") {
			return;
		}
        console.log("StringPauseGateNode: nodeCreated", node.id);

        // --- Create the simple Button Widget ---
        const container = document.createElement("div");
        container.style.padding = "1px";
        container.style.width = "100%";
        // 将 'bottom' 改为 'center' 来实现水平居中
        container.style.textAlign = "center";

        const button = document.createElement("button");
        button.textContent = "Continue";
        button.style.padding = "3px 8px";
        button.style.cursor = "pointer";
        button.style.border = "1px solid var(--border-color)";
        button.style.backgroundColor = "#228B22"; // 绿色
        button.style.color = "var(--input-text)";
        button.style.borderRadius = "10px"; // Slightly rounded
        button.disabled = true; // Start disabled
        node.continueButton = button; // Store reference

        container.appendChild(button);

        const widget = node.addDOMWidget(`minimal_pause_widget_${node.id}`, "dom", container, {
             // No serialize needed usually for simple buttons
        });

        // --- computeSize (Minimal Height) ---
        widget.computeSize = function(width) {
            // Enough height for the button and padding
            // Assuming your button + padding needs about 30-40px height
            const minHeight = 35; // Adjust this value based on actual button size + padding
            // computeSize should return [width, height]
            return [width, minHeight];
        }

        // --- Button Click Handler ---
        button.addEventListener("click", async () => {
            const nodeId = node.id;
            console.log(`StringPauseGateNode: 'Continue' clicked for node ${nodeId}.`);
            button.textContent = "Continue"; // Optional feedback
            button.disabled = true; // Disable while processing

            try {
                // Call the backend endpoint to create the signal file
                const response = await api.fetchApi("/string_pause/signal_continue", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ node_id: nodeId }), // Only need node_id
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    const errorMsg = `Error signaling continue: ${errorData.message || response.statusText} (Status: ${response.status})`;
                    console.error(`StringPauseGateNode: ${errorMsg} for node ${nodeId}`);
                    alert(errorMsg);
                    button.textContent = "Continue"; // Restore button text on error
                    button.disabled = false; // Re-enable
                } else {
                    // Backend confirmed signal - Re-queue the prompt
                    console.log(`StringPauseGateNode: Signal sent for node ${nodeId}. Re-queueing prompt.`);
                    app.queuePrompt(); // Trigger next run
                    // Keep button disabled as workflow restarts
                }
            } catch (error) {
                console.error(`StringPauseGateNode: Network error during continue for node ${nodeId}:`, error);
                alert(`Network error during continue: ${error.message}`);
                 button.textContent = "Continue"; // Restore on network error
                 button.disabled = false; // Re-enable
            }
        });

		// Initial size
        node.size = widget.computeSize(node.size[0]);
        node.setDirtyCanvas(true, true);
	}, // end nodeCreated

	async setup(app) {
        console.log("StringPauseGateNode: Setup.");
		// Listen for the signal from Python to enable the button
		api.addEventListener("string_pause_enable_button", (event) => {
			const { node_id } = event.detail;
            console.log(`StringPauseGateNode: Received 'enable_button' event for node ${node_id}`);
			const node = app.graph.getNodeById(node_id);
			if (node && node.continueButton) {
				console.log(`StringPauseGateNode: Enabling button for node ${node_id}`);
                node.continueButton.textContent = "Continue";
				node.continueButton.disabled = false; // Enable the button
                // Optional: Maybe make the node flash or something?
                node.setDirtyCanvas(true, false); // Minor redraw might be needed
			} else {
                console.warn(`StringPauseGateNode: Could not find node or button for ID ${node_id} during 'enable_button' event.`);
            }
		});
	} // end setup
});

console.log("StringPauseGateNode: Extension registered.");