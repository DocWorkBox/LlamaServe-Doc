import assert from "node:assert/strict";
import test from "node:test";

import { installFreeMemoryBridge } from "../web/free_memory_bridge.mjs";


test("ComfyUI memory cleanup also requests LlamaServe stop", async () => {
    const calls = [];
    const api = {
        async fetchApi(route, options) {
            calls.push({ route, options });
            return { ok: true };
        },
    };

    installFreeMemoryBridge(api);
    await api.fetchApi("/free", {
        method: "POST",
        body: JSON.stringify({ unload_models: true, free_memory: true }),
    });

    assert.deepEqual(
        calls.map((call) => call.route),
        ["/free", "/llamaserve_doc/stop"],
    );
    assert.equal(calls[1].options.method, "POST");
});


test("ordinary API requests do not stop LlamaServe", async () => {
    const calls = [];
    const api = {
        async fetchApi(route, options) {
            calls.push({ route, options });
            return { ok: true };
        },
    };

    installFreeMemoryBridge(api);
    await api.fetchApi("/queue", { method: "GET" });

    assert.deepEqual(calls.map((call) => call.route), ["/queue"]);
});
