import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";


const helperPath = resolve(
    import.meta.dirname,
    "..",
    "web",
    "h3_reference_mentions.mjs",
);

assert.ok(existsSync(helperPath), "the H3 @-mention helper must exist");

const {
    collectReferenceMentions,
    findMentionQuery,
    insertReferenceMention,
} = await import(pathToFileURL(helperPath));


const input = (name, connected = true) => ({
    name: `group.${name}`,
    label: name,
    link: connected ? 1 : null,
});


test("lists only connected references using backend canonical numbering", () => {
    const mentions = collectReferenceMentions([
        input("ref_image_2"),
        input("ref_image_3", false),
        input("ref_video_2"),
        input("ref_video_audio_2"),
        input("ref_audio_0"),
        input("ref_audio_1", false),
    ]);

    assert.deepEqual(
        mentions.map(({ source, label }) => ({ source, label })),
        [
            { source: "ref_image_2", label: "<Picture 1>" },
            { source: "ref_video_audio_2", label: "<Audio 1>" },
            { source: "ref_video_2", label: "<Video 1>" },
            { source: "ref_audio_0", label: "<Audio 2>" },
        ],
    );
});


test("does not offer a paired soundtrack without its matching video", () => {
    const mentions = collectReferenceMentions([
        input("ref_video_0", false),
        input("ref_video_audio_0"),
    ]);

    assert.deepEqual(mentions, []);
});


test("ignores bypassed source nodes when assigning reference numbers", () => {
    const graph = {
        links: new Map([
            [3, { origin_id: 4 }],
            [4, { origin_id: 4 }],
            [7, { origin_id: 7 }],
        ]),
        getNodeById(id) {
            return { mode: id === 4 ? 4 : 0 };
        },
    };
    const mentions = collectReferenceMentions([
        { ...input("ref_video_0"), link: 3 },
        { ...input("ref_video_audio_0"), link: 4 },
        { ...input("ref_audio_0"), link: 7 },
    ], graph);

    assert.deepEqual(
        mentions.map(({ source, label }) => ({ source, label })),
        [{ source: "ref_audio_0", label: "<Audio 1>" }],
    );
});


test("recognizes connected references when the frontend exposes qualified labels", () => {
    const mentions = collectReferenceMentions([
        {
            name: "ref_images.ref_image_0",
            label: "ref_images.ref_image_0",
            link: 7,
        },
    ]);

    assert.deepEqual(
        mentions.map(({ source, label }) => ({ source, label })),
        [{ source: "ref_image_0", label: "<Picture 1>" }],
    );
});


test("uses the connected source node preview in the mention menu", () => {
    const graph = {
        links: new Map([[7, { origin_id: 42 }]]),
        getNodeById(id) {
            assert.equal(id, 42);
            return { imgs: [{ src: "/view?filename=reference.png&type=input" }] };
        },
    };
    const mentions = collectReferenceMentions([
        {
            name: "ref_images.ref_image_0",
            label: "ref_images.ref_image_0",
            link: 7,
        },
    ], graph);

    assert.equal(mentions[0].thumbnail, "/view?filename=reference.png&type=input");
});


test("finds the active @ query and replaces it with the selected official label", () => {
    const value = "将 @视频 替换为主参考";
    const caret = value.indexOf(" ", 2);
    const query = findMentionQuery(value, caret);

    assert.deepEqual(query, { start: 2, end: caret, query: "视频" });
    assert.deepEqual(
        insertReferenceMention(value, query, "<Video 1>"),
        {
            value: "将 <Video 1> 替换为主参考",
            caret: 12,
        },
    );
});
