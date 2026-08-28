const SLOT_PATTERN = /^(ref_image|ref_video_audio|ref_video|ref_audio)_(\d+)$/;


function isConnected(input) {
    return (
        (input?.link !== null && input?.link !== undefined) ||
        (Array.isArray(input?.links) && input.links.length > 0) ||
        (Array.isArray(input?.linkIds) && input.linkIds.length > 0)
    );
}


function sourceName(input) {
    const candidate = input?.label || String(input?.name || "").split(".").at(-1);
    return SLOT_PATTERN.test(candidate) ? candidate : null;
}


function connectedIndexMap(inputs, prefix) {
    return new Map(
        (inputs || [])
            .filter(isConnected)
            .map(sourceName)
            .map((name) => [name, name?.match(SLOT_PATTERN)])
            .filter(([, match]) => match?.[1] === prefix)
            .map(([name, match]) => [Number(match[2]), name])
            .sort(([left], [right]) => left - right),
    );
}


export function collectReferenceMentions(inputs) {
    const images = connectedIndexMap(inputs, "ref_image");
    const videos = connectedIndexMap(inputs, "ref_video");
    const pairedAudio = connectedIndexMap(inputs, "ref_video_audio");
    const audio = connectedIndexMap(inputs, "ref_audio");
    const mentions = [];

    let pictureNumber = 0;
    for (const source of images.values()) {
        pictureNumber += 1;
        mentions.push({
            source,
            label: `<Picture ${pictureNumber}>`,
            kind: "image",
            description: "参考图片",
        });
    }

    let videoNumber = 0;
    let audioNumber = 0;
    for (const [index, source] of videos) {
        if (pairedAudio.has(index)) {
            audioNumber += 1;
            mentions.push({
                source: pairedAudio.get(index),
                label: `<Audio ${audioNumber}>`,
                kind: "audio",
                description: "视频配套音频",
            });
        }
        videoNumber += 1;
        mentions.push({
            source,
            label: `<Video ${videoNumber}>`,
            kind: "video",
            description: "参考视频",
        });
    }

    for (const source of audio.values()) {
        audioNumber += 1;
        mentions.push({
            source,
            label: `<Audio ${audioNumber}>`,
            kind: "audio",
            description: "独立参考音频",
        });
    }
    return mentions;
}


export function findMentionQuery(value, caret) {
    const end = Math.max(0, Math.min(Number(caret) || 0, value.length));
    const match = value.slice(0, end).match(/@([^\s@<>]*)$/u);
    if (!match) return null;
    return {
        start: end - match[0].length,
        end,
        query: match[1],
    };
}


export function insertReferenceMention(value, mention, label) {
    const before = value.slice(0, mention.start);
    const after = value.slice(mention.end);
    const separator = /^\s/u.test(after) ? "" : " ";
    const nextValue = `${before}${label}${separator}${after}`;
    const caret = before.length + label.length + (separator ? 1 : 0) + (/^\s/u.test(after) ? 1 : 0);
    return { value: nextValue, caret };
}
