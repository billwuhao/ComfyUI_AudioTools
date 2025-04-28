import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Helper function specifically for SaveAudioMW to add/update player
function addOrUpdateSaveAudioPlayer(node, audioUrl) {
    const widgetName = "save_audio_player_button";
    const initialText = "▶️ Play";
    let audioPlayerInstance = null; // Keep track of the player instance

    // --- Cleanup existing widget and player ---
    const existingWidgetIndex = node.widgets?.findIndex(w => w.name === widgetName);
    if (existingWidgetIndex !== -1 && existingWidgetIndex !== undefined) {
        const oldWidget = node.widgets[existingWidgetIndex];
        console.log(`SaveAudioMW (${node.id}): Found existing player widget. Cleaning up...`);
        const buttonElement = oldWidget.element?.querySelector('button');
        if (buttonElement && buttonElement.audioPlayer) {
            buttonElement.audioPlayer.pause();
            buttonElement.audioPlayer.src = ''; // Release resource
            buttonElement.audioPlayer = null;   // Remove reference
        }
        // Remove the DOM element
        const oldDomElement = node.content?.querySelector(`[data-widget-name="${widgetName}"]`);
        if (oldDomElement) oldDomElement.remove();
        node.widgets.splice(existingWidgetIndex, 1);
        console.log(`   - Cleanup complete for widget ${widgetName}.`);
    }

    // --- Create new widget and player ---
    const container = document.createElement("div");
    container.style.padding = "1px";
    container.style.width = "100%";
    container.style.textAlign = "center";
    container.dataset.widgetName = widgetName; // For easier lookup during cleanup

    const button = document.createElement("button");
    button.textContent = initialText;
    button.style.padding = "3px 8px";
    button.style.cursor = "pointer";
    button.style.border = "1px solid var(--border-color)";
    button.style.backgroundColor = "#4CAF50"; // Slightly different green
    button.style.color = "var(--input-text)";
    button.style.borderRadius = "10px";
    button.name = widgetName;

    container.appendChild(button);

    // Create new Audio object
    console.log(`SaveAudioMW (${node.id}): Creating new Audio object for URL: ${audioUrl}`);
    audioPlayerInstance = new Audio(audioUrl);
    button.audioPlayer = audioPlayerInstance; // Attach instance to button

    // Event listeners for the audio player
    audioPlayerInstance.onended = () => {
        console.log(`SaveAudioMW (${node.id}): Audio finished playing.`);
        button.textContent = initialText;
    };
    audioPlayerInstance.onerror = (e) => {
        console.error(`SaveAudioMW (${node.id}): Audio error:`, e);
        if (audioPlayerInstance.error) {
            console.error(`   Error code: ${audioPlayerInstance.error.code}, message: ${audioPlayerInstance.error.message}`);
        }
        button.textContent = "Error";
        button.disabled = true;
        button.audioPlayer = null; // Clear broken instance
    };

    // Button click handler
    button.addEventListener("click", () => {
        if (button.audioPlayer) {
            if (!button.audioPlayer.paused) {
                button.audioPlayer.pause();
                button.textContent = initialText;
            } else {
                button.audioPlayer.play().then(() => {
                    button.textContent = "⏸️ Pause";
                }).catch(e => {
                    console.error(`SaveAudioMW (${node.id}): Error starting playback:`, e);
                    button.textContent = "Error"; // Indicate playback error
                });
            }
        } else {
            console.warn(`SaveAudioMW (${node.id}): Play button clicked, but audioPlayer instance is missing.`);
        }
    });

    // Add the DOM widget
    const widget = node.addDOMWidget(widgetName, "dom", container, {});
    widget.element = container; // Make sure element is accessible

    // Recalculate size and redraw
    node.computeSize();
    node.setDirtyCanvas(true, true);
    console.log(`SaveAudioMW (${node.id}): Added new player widget.`);
}


app.registerExtension({
    name: "Comfy.SaveAudioWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "SaveAudioMW") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments); // Call original first

                if (message?.audio?.[0]) {
                    const audioInfo = message.audio[0];
                    const filename = encodeURIComponent(audioInfo.filename);
                    const subfolder = audioInfo.subfolder ? encodeURIComponent(audioInfo.subfolder) : "";
                    const type = audioInfo.type;
                    const format = audioInfo.filename.split('.').pop().toLowerCase();
                    let audioUrl = `/view?filename=${filename}&type=${type}${subfolder ? '&subfolder=' + subfolder : ''}&format=audio/${format}`;

                    console.log(`SaveAudioMW (${this.id}) executed. Preparing player for URL: ${audioUrl}`);

                    // Use setTimeout to ensure UI updates happen after execution finishes
                    setTimeout(() => {
                        if (app.graph.getNodeById(this.id)) { // Check if node still exists
                            addOrUpdateSaveAudioPlayer(this, audioUrl);
                        } else {
                            console.log(`SaveAudioMW (${this.id}): Node no longer exists, skipping player addition.`);
                        }
                    }, 50); // Short delay
                } else {
                    console.log(`SaveAudioMW (${this.id}) executed, but no audio message found.`);
                    // Optionally remove existing player if no audio is output
                    // addOrUpdateSaveAudioPlayer(this, null); // Pass null to trigger cleanup only
                }
            };

            // Add cleanup on node removal
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                console.log(`SaveAudioMW (${this.id}): Node removed. Cleaning up player.`);
                const widgetName = "save_audio_player_button";
                const existingWidgetIndex = this.widgets?.findIndex(w => w.name === widgetName);
                if (existingWidgetIndex !== -1 && existingWidgetIndex !== undefined) {
                     const oldWidget = this.widgets[existingWidgetIndex];
                     const buttonElement = oldWidget.element?.querySelector('button');
                     if (buttonElement && buttonElement.audioPlayer) {
                         buttonElement.audioPlayer.pause();
                         buttonElement.audioPlayer.src = '';
                         buttonElement.audioPlayer = null;
                     }
                     const oldDomElement = this.content?.querySelector(`[data-widget-name="${widgetName}"]`);
                     if (oldDomElement) oldDomElement.remove();
                     this.widgets.splice(existingWidgetIndex, 1);
                     console.log(`   - Cleaned up save audio player widget during node removal.`);
                }
                onRemoved?.apply(this, arguments); // Call original
            };
        }
    }
});

console.log("Comfy.SaveAudioWidget extension registered.");