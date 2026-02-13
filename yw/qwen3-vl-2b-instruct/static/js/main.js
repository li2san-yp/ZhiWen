const API_BASE = "http://172.27.235.239:4543";

const chatEl = document.getElementById("chat");
const promptEl = document.getElementById("prompt");
const sendStopBtn = document.getElementById("sendStopBtn");
const healthBtn = document.getElementById("healthBtn");
const clearBtn = document.getElementById("clearBtn");
const statusEl = document.getElementById("status");
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const previewImg = document.getElementById("previewImg");
const removeImgBtn = document.getElementById("removeImg");

const state = {
    messages: [],
    imageDataUrl: null,
    controller: null,
    streaming: false
};

function setStatus(message, isError) {
    statusEl.textContent = message || "";
    statusEl.className = isError ? "status error" : "status";
}

function sanitizeAssistantText(text) {
    if (!text) return "";
    const raw = String(text);

    const assistantRegex = /(?:^|\n)\s*(assistant|Assistant|模型)\s*:?/g;
    let lastIndex = -1;
    let match;
    while ((match = assistantRegex.exec(raw)) !== null) {
    lastIndex = match.index + match[0].length;
    }

    let trimmed = lastIndex >= 0 ? raw.slice(lastIndex) : raw;

    const userSplit = trimmed.split(/(?:^|\n)\s*user\s*:?/i);
    trimmed = userSplit[userSplit.length - 1];

    return trimmed.trimStart();
}

function addMessage(role, text, imageUrl) {
    const msg = document.createElement("div");
    msg.className = role === "user" ? "message user" : "message";

    const content = document.createElement("div");
    content.className = role === "assistant" ? "markdown" : "";
    content.textContent = text || "";
    msg.appendChild(content);

    if (imageUrl) {
    const img = document.createElement("img");
    img.src = imageUrl;
    msg.appendChild(img);
    }

    chatEl.appendChild(msg);
    chatEl.scrollTop = chatEl.scrollHeight;
    return { msg, content };
}

function renderMarkdown(element, text) {
    const safeText = sanitizeAssistantText(text);
    element.innerHTML = marked.parse(safeText || "");
}

function buildContent(text, imageDataUrl) {
    const content = [];
    if (text && text.trim()) {
    content.push({ type: "text", text: text.trim() });
    }
    if (imageDataUrl) {
    content.push({ type: "image", image: imageDataUrl });
    }
    return content;
}

function resetImage() {
    state.imageDataUrl = null;
    imageInput.value = "";
    preview.style.display = "none";
    previewImg.src = "";
}

function setStreaming(isStreaming) {
    state.streaming = isStreaming;
    sendStopBtn.textContent = isStreaming ? "停止" : "发送";
    sendStopBtn.classList.toggle("stop", isStreaming);
    sendStopBtn.classList.toggle("send", !isStreaming);
}

imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
    state.imageDataUrl = reader.result;
    previewImg.src = state.imageDataUrl;
    preview.style.display = "flex";
    };
    reader.readAsDataURL(file);
});

removeImgBtn.addEventListener("click", () => {
    resetImage();
});

clearBtn.addEventListener("click", () => {
    state.messages = [];
    chatEl.innerHTML = "";
    setStatus("已清空对话", false);
});

healthBtn.addEventListener("click", async () => {
    try {
    const res = await fetch(API_BASE + "/api/health");
    if (!res.ok) throw new Error("health failed");
    const data = await res.json();
    setStatus("服务正常，模型: " + data.model + "，设备: " + data.device, false);
    } catch (err) {
    setStatus("无法连接到服务", true);
    }
});

promptEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendStopBtn.click();
    }
});

async function sendMessage() {
    const text = promptEl.value.trim();
    if (!text && !state.imageDataUrl) {
    setStatus("请输入内容或上传图片", true);
    return;
    }

    setStatus("", false);
    setStreaming(true);

    addMessage("user", text, state.imageDataUrl);
    state.messages.push({
    role: "user",
    content: buildContent(text, state.imageDataUrl)
    });

    promptEl.value = "";
    resetImage();

    const typing = addMessage("assistant", "", null);
    typing.content.textContent = "正在生成...";

    const controller = new AbortController();
    state.controller = controller;

    try {
    const res = await fetch(API_BASE + "/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        messages: state.messages,
        max_new_tokens: 512,
        temperature: 0.7,
        top_p: 0.9
        }),
        signal: controller.signal
    });

    if (!res.ok) {
        const errText = await res.text();
        throw new Error("HTTP " + res.status + ": " + errText);
    }

    if (!res.body) {
        throw new Error("浏览器不支持流式读取");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullText = "";
    typing.content.textContent = "";
    let lastRenderAt = 0;
    let hasRendered = false;
    const renderIntervalMs = 50;

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        fullText += chunk;
        const now = Date.now();
        if (now - lastRenderAt >= renderIntervalMs) {
        renderMarkdown(typing.content, fullText);
        lastRenderAt = now;
        hasRendered = true;
        } else {
        if (!hasRendered) {
            typing.content.textContent = sanitizeAssistantText(fullText);
        }
        }
        chatEl.scrollTop = chatEl.scrollHeight;
    }

    renderMarkdown(typing.content, fullText);
    state.messages.push({
        role: "assistant",
        content: [{ type: "text", text: sanitizeAssistantText(fullText) }]
    });
    } catch (err) {
    if (err.name === "AbortError") {
        setStatus("已停止生成", false);
    } else {
        typing.content.textContent = "";
        setStatus("请求失败: " + err.message, true);
    }
    } finally {
    state.controller = null;
    setStreaming(false);
    }
}

sendStopBtn.addEventListener("click", () => {
    if (state.streaming) {
    if (state.controller) {
        state.controller.abort();
        state.controller = null;
    }
    setStreaming(false);
    setStatus("已停止生成", false);
    return;
    }

    sendMessage();
});
