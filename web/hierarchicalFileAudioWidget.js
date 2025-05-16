// // alert("Hierarchical Audio Widget JS Loaded!");
// console.log("Hierarchical Audio Widget JS file is being processed.");

// // File: js/hierarchicalFileAudioWidget.js
// import { app } from "../../scripts/app.js";
// import { api } from "../../scripts/api.js";

// // Helper function to create tree elements
// function createTreeElement(item, onFileSelect, currentPath) {
//     const element = document.createElement("li");
//     element.style.cursor = "pointer";
//     element.style.marginLeft = "10px"; // Basic indentation

//     const nameSpan = document.createElement("span");
//     nameSpan.textContent = (item.type === "folder" ? "📁 " : "🎵 ") + item.name;
//     element.appendChild(nameSpan);

//     if (item.type === "folder") {
//         nameSpan.style.fontWeight = "bold";
//         const childrenContainer = document.createElement("ul");
//         childrenContainer.style.display = "none"; // Collapsed by default
//         childrenContainer.style.listStyleType = "none";
//         childrenContainer.style.paddingLeft = "15px";

//         if (item.children && item.children.length > 0) {
//             item.children.forEach(child => {
//                 childrenContainer.appendChild(createTreeElement(child, onFileSelect, item.path));
//             });
//         } else {
//             const emptyMsg = document.createElement("li");
//             emptyMsg.textContent = "(empty)";
//             emptyMsg.style.fontStyle = "italic";
//             emptyMsg.style.color = "#888";
//             childrenContainer.appendChild(emptyMsg);
//         }
//         element.appendChild(childrenContainer);

//         element.addEventListener("click", (e) => {
//             e.stopPropagation(); // Prevent event from bubbling up to parent folders
//             const isVisible = childrenContainer.style.display !== "none";
//             childrenContainer.style.display = isVisible ? "none" : "block";
//             nameSpan.textContent = (isVisible ? "📁 " : "📂 ") + item.name; // Change icon
//         });
//     } else { // File
//         element.addEventListener("click", (e) => {
//             e.stopPropagation();
//             onFileSelect(item.path); // Pass the relative path of the file
//         });
//     }
//     return element;
// }

// if (app && typeof app.registerExtension === 'function') {
//     console.log("[MW_AUDIO_WIDGET] app.registerExtension IS a function. Attempting to register.");
//     app.registerExtension({
//         name: "MW.HierarchicalFileAudioWidget", // Unique name for your extension
//         // 添加 setup 函数
//         setup(appInstance) {
//             console.log("[MW_AUDIO_WIDGET] EXTENSION SETUP FUNCTION CALLED!");
//         },
//         async getCustomWidgets(appInstance) {
//             console.log("[MW_AUDIO_WIDGET] getCustomWidgets CALLED. Preparing to return object. (USING NORMAL FUNCTION)");
    
//             const widgetsToRegister = {
//                 "HIERARCHICAL_FILE_AUDIO": function(node, inputName, inputData, appRef) { // 改为普通函数
//                     // 使用 'this' 关键字时要小心，在这个上下文中它可能不是你期望的
//                     console.log(`[MW_AUDIO_WIDGET] NORMAL FUNCTION FACTORY for HIERARCHICAL_FILE_AUDIO CALLED! Input: ${inputName}`);
//                     // ... (其余的工厂函数逻辑保持不变，使用空的或简化的版本进行测试)
//                     let el = document.createElement("span");
//                     el.textContent = "Normal Function Factory Called";
//                     return { type: "STRING", name: inputName, value: "", element: el, computeSize: (w)=>[w,20] };
//                 }
//             };
    
//             console.log("[MW_AUDIO_WIDGET] Object being returned by getCustomWidgets:", widgetsToRegister);
//             return widgetsToRegister;
//         }
//     });
//     console.log("[MW_AUDIO_WIDGET] Extension registration attempt FINISHED.");
// } else {
//     console.error("[MW_AUDIO_WIDGET] ERROR: app or app.registerExtension is not available!");
// }