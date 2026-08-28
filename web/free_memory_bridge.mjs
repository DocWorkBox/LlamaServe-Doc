const BRIDGE_MARKER = Symbol.for("llamaserve-doc.free-memory-bridge");


function requestPayload(options) {
    const body = options?.body;
    if (typeof body === "string") {
        try {
            return JSON.parse(body);
        } catch {
            return null;
        }
    }
    return body && typeof body === "object" ? body : null;
}


export function isComfyMemoryCleanup(route, options) {
    if (route !== "/free") return false;
    const payload = requestPayload(options);
    return Boolean(payload?.free_memory || payload?.unload_models);
}


export function installFreeMemoryBridge(api) {
    if (!api || api[BRIDGE_MARKER]) return false;
    const originalFetchApi = api.fetchApi.bind(api);

    api.fetchApi = async function llamaServeFetchApi(route, options) {
        const response = await originalFetchApi(route, options);
        if (isComfyMemoryCleanup(route, options)) {
            try {
                await originalFetchApi("/llamaserve_doc/stop", { method: "POST" });
            } catch (error) {
                console.warn("LlamaServe-Doc could not stop its server during memory cleanup", error);
            }
        }
        return response;
    };
    api[BRIDGE_MARKER] = true;
    return true;
}
