import { app } from "../../scripts/app.js";
import {
    collectReferenceMentions,
    findMentionQuery,
    insertReferenceMention,
} from "./h3_reference_mentions.mjs";


const NODE_CLASS = "LlamaServeDocH3OmniGenerate";
const CONTROLLER_KEY = "__llamaServeDocReferenceAutocomplete";


function textareaFor(widget) {
    const element = widget?.element || widget?.inputEl;
    if (element instanceof HTMLTextAreaElement) return element;
    if (element?.$el instanceof HTMLElement) {
        return element.$el.querySelector("textarea");
    }
    if (element instanceof HTMLElement) {
        return element.querySelector("textarea");
    }
    return null;
}


function matchesQuery(item, query) {
    if (!query) return true;
    const haystack = [item.label, item.source, item.description, item.kind]
        .join(" ")
        .toLocaleLowerCase();
    return haystack.includes(query.toLocaleLowerCase());
}


function createPopup() {
    const popup = document.createElement("div");
    popup.className = "llamaserve-h3-reference-autocomplete";
    popup.setAttribute("role", "listbox");
    Object.assign(popup.style, {
        position: "fixed",
        display: "none",
        zIndex: "100000",
        minWidth: "300px",
        maxWidth: "460px",
        maxHeight: "260px",
        overflowY: "auto",
        padding: "6px",
        border: "1px solid var(--border-color, #555)",
        borderRadius: "8px",
        background: "var(--comfy-menu-bg, #242424)",
        boxShadow: "0 10px 28px rgba(0, 0, 0, 0.45)",
        color: "var(--input-text, #eee)",
        font: "12px/1.35 system-ui, sans-serif",
    });
    document.body.appendChild(popup);
    return popup;
}


function attachAutocomplete(node, attempt = 0) {
    if (node[CONTROLLER_KEY]) return;
    const widget = node.widgets?.find((candidate) => candidate.name === "raw_prompt");
    const textarea = textareaFor(widget);
    if (!widget || !textarea) {
        if (attempt < 20) {
            requestAnimationFrame(() => attachAutocomplete(node, attempt + 1));
        }
        return;
    }

    const popup = createPopup();
    let mention = null;
    let visibleItems = [];
    let selectedIndex = 0;

    const hide = () => {
        popup.style.display = "none";
        popup.replaceChildren();
        mention = null;
        visibleItems = [];
        selectedIndex = 0;
    };

    const positionPopup = () => {
        const bounds = textarea.getBoundingClientRect();
        const width = Math.max(300, Math.min(460, bounds.width));
        popup.style.width = `${width}px`;
        popup.style.left = `${Math.max(8, Math.min(bounds.left, window.innerWidth - width - 8))}px`;
        const below = bounds.bottom + 4;
        popup.style.top = `${Math.min(below, window.innerHeight - popup.offsetHeight - 8)}px`;
    };

    const updateSelection = () => {
        [...popup.querySelectorAll("button")].forEach((button, index) => {
            const active = index === selectedIndex;
            button.setAttribute("aria-selected", String(active));
            button.style.background = active
                ? "var(--comfy-input-bg, #3b3b3b)"
                : "transparent";
        });
    };

    const choose = (item) => {
        if (!mention) return;
        const next = insertReferenceMention(textarea.value, mention, item.label);
        widget.value = next.value;
        textarea.value = next.value;
        textarea.dispatchEvent(new Event("input", { bubbles: true }));
        textarea.dispatchEvent(new Event("change", { bubbles: true }));
        textarea.focus();
        textarea.setSelectionRange(next.caret, next.caret);
        node.setDirtyCanvas?.(true, true);
        hide();
    };

    const render = () => {
        mention = findMentionQuery(textarea.value, textarea.selectionStart);
        if (!mention) {
            hide();
            return;
        }

        visibleItems = collectReferenceMentions(node.inputs).filter((item) =>
            matchesQuery(item, mention.query),
        );
        selectedIndex = Math.min(selectedIndex, Math.max(0, visibleItems.length - 1));
        popup.replaceChildren();

        if (!visibleItems.length) {
            const message = document.createElement("div");
            message.textContent = collectReferenceMentions(node.inputs).length
                ? "没有匹配的已连接媒体"
                : "请先连接图片、视频或音频参考";
            Object.assign(message.style, { padding: "8px 10px", opacity: "0.75" });
            popup.appendChild(message);
        } else {
            visibleItems.forEach((item, index) => {
                const button = document.createElement("button");
                button.type = "button";
                button.setAttribute("role", "option");
                Object.assign(button.style, {
                    display: "grid",
                    gridTemplateColumns: "110px 1fr",
                    gap: "10px",
                    width: "100%",
                    padding: "7px 9px",
                    border: "0",
                    borderRadius: "5px",
                    color: "inherit",
                    textAlign: "left",
                    cursor: "pointer",
                });
                const label = document.createElement("strong");
                label.textContent = item.label;
                const detail = document.createElement("span");
                detail.textContent = `${item.source} · ${item.description}`;
                detail.style.opacity = "0.72";
                button.append(label, detail);
                button.addEventListener("mouseenter", () => {
                    selectedIndex = index;
                    updateSelection();
                });
                button.addEventListener("mousedown", (event) => event.preventDefault());
                button.addEventListener("click", () => choose(item));
                popup.appendChild(button);
            });
        }

        popup.style.display = "block";
        positionPopup();
        updateSelection();
    };

    const onKeyDown = (event) => {
        if (popup.style.display === "none") return;
        if (event.key === "Escape") {
            event.preventDefault();
            hide();
            return;
        }
        if (!visibleItems.length) return;
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            const direction = event.key === "ArrowDown" ? 1 : -1;
            selectedIndex = (selectedIndex + direction + visibleItems.length) % visibleItems.length;
            updateSelection();
            popup.querySelectorAll("button")[selectedIndex]?.scrollIntoView({ block: "nearest" });
            return;
        }
        if (event.key === "Enter" || event.key === "Tab") {
            event.preventDefault();
            choose(visibleItems[selectedIndex]);
        }
    };

    const onBlur = () => setTimeout(hide, 120);
    const onWindowChange = () => popup.style.display !== "none" && positionPopup();
    textarea.addEventListener("input", render);
    textarea.addEventListener("click", render);
    textarea.addEventListener("keyup", render);
    textarea.addEventListener("keydown", onKeyDown);
    textarea.addEventListener("blur", onBlur);
    window.addEventListener("resize", onWindowChange);
    window.addEventListener("scroll", onWindowChange, true);

    const destroy = () => {
        textarea.removeEventListener("input", render);
        textarea.removeEventListener("click", render);
        textarea.removeEventListener("keyup", render);
        textarea.removeEventListener("keydown", onKeyDown);
        textarea.removeEventListener("blur", onBlur);
        window.removeEventListener("resize", onWindowChange);
        window.removeEventListener("scroll", onWindowChange, true);
        popup.remove();
        delete node[CONTROLLER_KEY];
    };
    node[CONTROLLER_KEY] = { destroy };
    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        destroy();
        return originalOnRemoved?.apply(this, arguments);
    };
}


app.registerExtension({
    name: "LlamaServe-Doc.H3ReferenceAutocomplete",
    nodeCreated(node) {
        if (node.comfyClass === NODE_CLASS) attachAutocomplete(node);
    },
    loadedGraphNode(node) {
        if (node.comfyClass === NODE_CLASS) attachAutocomplete(node);
    },
});
