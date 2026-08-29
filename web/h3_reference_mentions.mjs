const SLOT_PATTERN = /^(ref_image|ref_video_audio|ref_video|ref_audio)_(\d+)$/;


function connectedLinkId(input) {
    if (input?.link !== null && input?.link !== undefined) return input.link;
    for (const links of [input?.links, input?.linkIds]) {
        if (Array.isArray(links) && links.length) return links[0];
        if (links instanceof Set && links.size) return links.values().next().value;
    }
    return null;
}


function isConnected(input) {
    return connectedLinkId(input) !== null;
}


function sourceName(input) {
    for (const value of [input?.label, input?.name]) {
        const candidate = String(value || "").split(".").at(-1);
        if (SLOT_PATTERN.test(candidate)) return candidate;
    }
    return null;
}


function graphLink(graph, linkId) {
    return (
        graph?.getLink?.(linkId) ||
        graph?.links?.get?.(linkId) ||
        graph?._links?.get?.(linkId) ||
        graph?.links?.[linkId] ||
        null
    );
}


function sourceThumbnail(input, graph, inputViewUrl) {
    const link = graphLink(graph, connectedLinkId(input));
    const sourceNode = link
        ? (graph?.getNodeById?.(link.origin_id) || graph?._nodes_by_id?.[link.origin_id])
        : null;
    if (!sourceNode) return "";

    const imageIndex = sourceNode.imageIndex ?? sourceNode.overIndex ?? 0;
    const image = sourceNode.imgs?.[imageIndex] || sourceNode.imgs?.[0];
    if (image?.src) return String(image.src);

    const videoPreview = sourceNode.widgets?.find((widget) => widget.name === "videopreview");
    if (videoPreview?.videoEl?.poster) return String(videoPreview.videoEl.poster);

    const imageWidget = sourceNode.widgets?.find((widget) => widget.name === "image");
    if (imageWidget?.value && typeof inputViewUrl === "function") {
        return inputViewUrl(String(imageWidget.value));
    }
    return "";
}


function connectedIndexMap(inputs, prefix, graph, inputViewUrl) {
    return new Map(
        (inputs || [])
            .filter(isConnected)
            .map((input) => [input, sourceName(input)])
            .map(([input, name]) => [input, name, name?.match(SLOT_PATTERN)])
            .filter(([, , match]) => match?.[1] === prefix)
            .map(([input, name, match]) => [
                Number(match[2]),
                { source: name, thumbnail: sourceThumbnail(input, graph, inputViewUrl) },
            ])
            .sort(([left], [right]) => left - right),
    );
}


export function collectReferenceMentions(inputs, graph = null, inputViewUrl = null) {
    const images = connectedIndexMap(inputs, "ref_image", graph, inputViewUrl);
    const videos = connectedIndexMap(inputs, "ref_video", graph, inputViewUrl);
    const pairedAudio = connectedIndexMap(inputs, "ref_video_audio", graph, inputViewUrl);
    const audio = connectedIndexMap(inputs, "ref_audio", graph, inputViewUrl);
    const mentions = [];

    let pictureNumber = 0;
    for (const item of images.values()) {
        pictureNumber += 1;
        mentions.push({
            source: item.source,
            label: `<Picture ${pictureNumber}>`,
            kind: "image",
            description: "参考图片",
            thumbnail: item.thumbnail,
        });
    }

    let videoNumber = 0;
    let audioNumber = 0;
    for (const [index, item] of videos) {
        if (pairedAudio.has(index)) {
            audioNumber += 1;
            mentions.push({
                source: pairedAudio.get(index).source,
                label: `<Audio ${audioNumber}>`,
                kind: "audio",
                description: "视频配套音频",
                thumbnail: "",
            });
        }
        videoNumber += 1;
        mentions.push({
            source: item.source,
            label: `<Video ${videoNumber}>`,
            kind: "video",
            description: "参考视频",
            thumbnail: item.thumbnail,
        });
    }

    for (const item of audio.values()) {
        audioNumber += 1;
        mentions.push({
            source: item.source,
            label: `<Audio ${audioNumber}>`,
            kind: "audio",
            description: "独立参考音频",
            thumbnail: "",
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
