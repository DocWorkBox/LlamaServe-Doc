import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { installFreeMemoryBridge } from "./free_memory_bridge.mjs";


app.registerExtension({
    name: "LlamaServe-Doc.FreeMemoryCleanup",
    setup() {
        installFreeMemoryBridge(api);
    },
});
