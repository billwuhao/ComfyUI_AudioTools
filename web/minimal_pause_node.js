// filename: ComfyUI/web/extensions/minimal_pause_node.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "Comfy.MinimalPauseNode",

	async nodeCreated(node) {
		if (node.constructor.comfyClass !== "MinimalPauseNode") {
			return;
		}
        console.log("MinimalPauseNode: nodeCreated", node.id);

		// --- Create the simple Button Widget ---
		const container = document.createElement("div");
        container.style.padding = "1px";
        container.style.width = "100%";
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
            console.log(`MinimalPauseNode: 'Continue' clicked for node ${nodeId}.`);
            button.textContent = "Continue"; // Optional feedback
            button.disabled = true; // Disable while processing

            try {
                // Call the backend endpoint to create the signal file
                const response = await api.fetchApi("/minimal_pause/signal_continue", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ node_id: nodeId }), // Only need node_id
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    const errorMsg = `Error signaling continue: ${errorData.message || response.statusText} (Status: ${response.status})`;
                    console.error(`MinimalPauseNode: ${errorMsg} for node ${nodeId}`);
                    alert(errorMsg);
                    button.textContent = "Continue"; // Restore button text on error
                    button.disabled = false; // Re-enable
                } else {
                    // Backend confirmed signal - Re-queue the prompt
                    console.log(`MinimalPauseNode: Signal sent for node ${nodeId}. Re-queueing prompt.`);
                    app.queuePrompt(); // Trigger next run
                    button.disabled = false; // Re-enable the button
                    // Workflow restarts, button is now re-enabled for potential subsequent manual pauses if the node is hit again.
                }
            } catch (error) {
                console.error(`MinimalPauseNode: Network error during continue for node ${nodeId}:`, error);
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
        console.log("MinimalPauseNode: Setup.");
		// Listen for the signal from Python to enable the button
		api.addEventListener("minimal_pause_enable_button", (event) => {
			const { node_id } = event.detail;
            console.log(`MinimalPauseNode: Received 'enable_button' event for node ${node_id}`);
			const node = app.graph.getNodeById(node_id);
			if (node && node.continueButton) {
				console.log(`MinimalPauseNode: Enabling button for node ${node_id}`);
                node.continueButton.textContent = "Continue";
				node.continueButton.disabled = false; // Enable the button
                // Optional: Maybe make the node flash or something?
                node.setDirtyCanvas(true, false); // Minor redraw might be needed
			} else {
                console.warn(`MinimalPauseNode: Could not find node or button for ID ${node_id} during 'enable_button' event.`);
            }
		});
	} // end setup
});

console.log("MinimalPauseNode: Extension registered.");