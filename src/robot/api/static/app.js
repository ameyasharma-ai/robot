/**
 * Robot Core — Frontend Controller
 * Connects to Robot Core WebSocket & REST API:
 * - Real-time state-driven facial expressions & natural blinking
 * - Mouth audio-reactive animation synced with speaking
 * - World State & Camera Snapshot auto-refresh
 * - Live pipeline telemetry metrics (STT, LLM, TTS, E2E)
 * - View switching: Face Only, Dual Mode, Console Only
 */

class RobotUI {
    constructor() {
        this.ws = null;
        this.currentMode = "idle";
        this.blinkInterval = null;
        this.mouthInterval = null;
        this.cameraInterval = null;
        this.worldStateInterval = null;

        // DOM Elements
        this.faceStage = document.getElementById("faceStage");
        this.faceCaption = document.getElementById("faceCaption");
        this.statusPill = document.getElementById("globalStatusPill");
        this.statusText = document.getElementById("globalStatusText");
        this.personaSelect = document.getElementById("personaSelect");
        this.personaName = document.getElementById("personaName");
        this.bubbleTag = document.getElementById("bubbleTag");
        this.bubbleContent = document.getElementById("bubbleContent");
        this.terminalFeed = document.getElementById("terminalFeed");
        this.mouthBars = document.querySelectorAll(".mouth-bar");
        this.cameraImg = document.getElementById("cameraImg");
        this.worldStateCode = document.getElementById("worldStateCode");

        // 3D Character Elements
        this.characterContainer = document.getElementById("characterContainer");
        this.characterEngine = null;
        this.charStateBadge = document.getElementById("charStateBadge");
        this.charGazeBadge = document.getElementById("charGazeBadge");
        this.charActionBadge = document.getElementById("charActionBadge");
        this.dbgCharState = document.getElementById("dbgCharState");
        this.dbgCharMotion = document.getElementById("dbgCharMotion");
        this.dbgCharFps = document.getElementById("hudCharFps");

        // Metric DOM
        this.hudTtft = document.getElementById("hudTtft");
        this.hudE2e = document.getElementById("hudE2e");
        this.hudTokens = document.getElementById("hudTokens");
        this.metricVad = document.getElementById("metricVad");
        this.metricStt = document.getElementById("metricStt");
        this.metricContext = document.getElementById("metricContext");
        this.metricTtft = document.getElementById("metricTtft");
        this.metricTts = document.getElementById("metricTts");
        this.metricE2e = document.getElementById("metricE2e");
        this.barVad = document.getElementById("barVad");
        this.barStt = document.getElementById("barStt");
        this.barContext = document.getElementById("barContext");
        this.barTtft = document.getElementById("barTtft");
        this.barTts = document.getElementById("barTts");
        this.barE2e = document.getElementById("barE2e");

        // Chat Workspace State
        this.chatSessions = [];
        this.activeChatSessionId = null;
        this.chatStreaming = false;
        this.chatAbortController = null;

        this.initViewControls();
        this.initPersonalitySelector();
        this.initLlmProviderToggle();
        this.init3DCharacter();
        this.initTextChat();
        this.initChatWorkspace();
        this.initBlinking();
        this.initWebsocket();
        this.initImageGen();
        this.startPollers();
    }

    initTextChat() {
        const input = document.getElementById("chatInput");
        const btn = document.getElementById("chatSendBtn");
        const voiceBtn = document.getElementById("btnVoiceToggle");
        const voiceIcon = document.getElementById("voiceToggleIcon");
        const voiceLabel = document.getElementById("voiceToggleLabel");
        this.voiceEnabled = true;

        if (voiceBtn) {
            const updateVoiceUI = (enabled) => {
                this.voiceEnabled = enabled;
                if (enabled) {
                    voiceBtn.classList.remove("muted");
                    voiceBtn.classList.add("active");
                    if (voiceIcon) voiceIcon.textContent = "🎙️";
                    if (voiceLabel) voiceLabel.textContent = "VOICE ON";
                    voiceBtn.title = "Voice interaction is ON. Click to mute mic and switch to text-only mode.";
                } else {
                    voiceBtn.classList.remove("active");
                    voiceBtn.classList.add("muted");
                    if (voiceIcon) voiceIcon.textContent = "🔇";
                    if (voiceLabel) voiceLabel.textContent = "VOICE MUTED";
                    voiceBtn.title = "Voice interaction is MUTED. Text chat only. Click to enable mic.";
                }
            };

            voiceBtn.addEventListener("click", async () => {
                const targetState = !this.voiceEnabled;
                updateVoiceUI(targetState);
                try {
                    const res = await fetch("/voice-mode", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ enabled: targetState }),
                    });
                    if (res.ok) {
                        const data = await res.json();
                        updateVoiceUI(data.voice_enabled);
                        this.logEvent("AUDIO", `Voice interaction ${data.voice_enabled ? "ENABLED" : "MUTED (Text-only chat active)"}`, "system");
                    }
                } catch (e) {
                    console.error("Failed to toggle voice mode:", e);
                }
            });
            this.updateVoiceUI = updateVoiceUI;
        }

        if (!input || !btn) return;

        let isSubmitting = false;

        const sendMessage = async () => {
            const text = input.value.trim();
            if (!text || isSubmitting) return;

            isSubmitting = true;
            btn.classList.add("sending");
            input.disabled = true;

            try {
                // Show user message in bubble immediately
                if (this.bubbleTag) {
                    this.bubbleTag.textContent = "USER (TEXT)";
                    this.bubbleTag.style.color = "var(--primary-cyan)";
                }
                if (this.bubbleContent) {
                    this.bubbleContent.textContent = text;
                }
                if (typeof this.logEvent === "function") {
                    this.logEvent("USER", text, "user");
                    this._lastUserLogged = text;
                }

                // Clear input so user can type next message
                input.value = "";
                if (typeof this.setRobotMode === "function") {
                    this.setRobotMode("thinking");
                }
                this.updatePipelineStatus({ CONTEXT: "RUNNING", LLM: "RUNNING" });
            } catch (err) {
                console.error("UI update failed before fetch:", err);
            }

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20000);

            try {
                const resp = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text }),
                    signal: controller.signal,
                });
                clearTimeout(timeoutId);

                if (!resp.ok) {
                    throw new Error(`Server returned HTTP ${resp.status}`);
                }

                const data = await resp.json();
                const rawResponse = data.response || data.error || "No response received.";
                const response = this.cleanHumanText(rawResponse) || rawResponse;

                // Show robot response
                if (this.bubbleTag) {
                    this.bubbleTag.textContent = "ROBOT RESPONSE";
                    this.bubbleTag.style.color = "var(--accent-emerald)";
                }
                if (this.bubbleContent) {
                    this.bubbleContent.textContent = response;
                }
                if (typeof this.logEvent === "function") {
                    this.logEvent("ROBOT", response, "robot");
                    this._lastRobotLogged = response;
                }
                if (typeof this.setRobotMode === "function") {
                    this.setRobotMode("speaking");
                }
                this.updatePipelineStatus({ CONTEXT: "COMPLETED", LLM: "COMPLETED", AVATAR: "RUNNING" });

                // Trigger lip-sync on 3D engine safely
                try {
                    if (typeof this._forEachEngine === "function") {
                        this._forEachEngine((eng) => {
                            if (eng && typeof eng.speakStart === "function") {
                                eng.speakStart();
                                setTimeout(() => {
                                    if (eng && typeof eng.speakStop === "function") eng.speakStop();
                                }, Math.min(8000, response.length * 80));
                            }
                        });
                    }
                } catch (e) {
                    console.warn("Avatar speech trigger warning:", e);
                }

                // Return to idle after speech completes
                setTimeout(() => {
                    if (typeof this.setRobotMode === "function") this.setRobotMode("idle");
                    this.updatePipelineStatus({ AVATAR: "COMPLETED" });
                    setTimeout(() => {
                        this.updatePipelineStatus({ CONTEXT: "IDLE", LLM: "IDLE", AVATAR: "IDLE", TTS: "IDLE", TOOLS: "IDLE" });
                    }, 2000);
                }, Math.min(10000, response.length * 80 + 1000));
            } catch (err) {
                clearTimeout(timeoutId);
                const errMsg = err.name === "AbortError" ? "Request timed out after 20s" : err.message;
                if (this.bubbleContent) {
                    this.bubbleContent.textContent = `[Error]: ${errMsg}`;
                }
                if (typeof this.logEvent === "function") {
                    this.logEvent("ERROR", errMsg, "system");
                }
                if (typeof this.setRobotMode === "function") {
                    this.setRobotMode("idle");
                }
                this.updatePipelineStatus({ CONTEXT: "ERROR", LLM: "ERROR" });
                setTimeout(() => {
                    this.updatePipelineStatus({ CONTEXT: "IDLE", LLM: "IDLE" });
                }, 3000);
            } finally {
                isSubmitting = false;
                btn.classList.remove("sending");
                input.disabled = false;
                input.focus();
            }
        };

        btn.addEventListener("click", (e) => {
            e.preventDefault();
            sendMessage();
        });

        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    initChatWorkspace() {
        this.chatSidebar = document.getElementById("chatSidebar");
        this.btnToggleChatSidebar = document.getElementById("btnToggleChatSidebar");
        this.btnNewChat = document.getElementById("btnNewChat");
        this.btnRefreshSessions = document.getElementById("btnRefreshSessions");
        this.chatSessionsList = document.getElementById("chatSessionsList");
        this.chatActiveTitle = document.getElementById("chatActiveTitle");
        this.chatSessionIdPill = document.getElementById("chatSessionIdPill");
        this.chatPersonaLabel = document.getElementById("chatPersonaLabel");
        this.chatModelLabel = document.getElementById("chatModelLabel");
        this.chatMicPill = document.getElementById("chatMicPill");
        this.chatMicLabel = document.getElementById("chatMicLabel");
        this.chatMessagesContainer = document.getElementById("chatMessagesContainer");
        this.chatTextarea = document.getElementById("chatTextarea");
        this.btnChatSubmit = document.getElementById("btnChatSubmit");
        this.btnChatStopStream = document.getElementById("btnChatStopStream");
        this.chatLiveSpeed = document.getElementById("chatLiveSpeed");

        if (this.btnToggleChatSidebar && this.chatSidebar) {
            this.btnToggleChatSidebar.addEventListener("click", () => {
                this.chatSidebar.classList.toggle("collapsed");
            });
        }

        if (this.btnNewChat) {
            this.btnNewChat.addEventListener("click", () => this.createChatSession());
        }

        if (this.btnRefreshSessions) {
            this.btnRefreshSessions.addEventListener("click", () => this.loadChatSessions());
        }

        if (this.chatTextarea) {
            this.chatTextarea.addEventListener("input", () => {
                this.chatTextarea.style.height = "auto";
                this.chatTextarea.style.height = `${Math.min(220, this.chatTextarea.scrollHeight)}px`;
            });

            this.chatTextarea.addEventListener("keydown", (e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    this.sendChatMessage();
                }
            });
        }

        if (this.btnChatSubmit) {
            this.btnChatSubmit.addEventListener("click", (e) => {
                e.preventDefault();
                this.sendChatMessage();
            });
        }

        if (this.btnChatStopStream) {
            this.btnChatStopStream.addEventListener("click", (e) => {
                e.preventDefault();
                this.stopChatStream();
            });
        }

        // Quick action chips
        document.querySelectorAll(".chat-chip").forEach((chip) => {
            chip.addEventListener("click", () => {
                const prompt = chip.getAttribute("data-prompt");
                if (prompt && this.chatTextarea) {
                    this.chatTextarea.value = prompt;
                    this.chatTextarea.style.height = "auto";
                    this.chatTextarea.style.height = `${Math.min(220, this.chatTextarea.scrollHeight)}px`;
                    this.sendChatMessage();
                }
            });
        });

        this.syncChatTopBar();
    }

    syncChatTopBar() {
        if (this.chatPersonaLabel && this.personaSelect) {
            const p = this.personaSelect.value;
            this.chatPersonaLabel.textContent = p === "neo" ? "CYPHER" : p.toUpperCase();
        }
        if (this.chatModelLabel) {
            const isGemini = this.btnLlmToggle && this.btnLlmToggle.classList.contains("gemini-mode");
            this.chatModelLabel.textContent = isGemini ? "GEMINI 3.5 FLASH (1M)" : "RTX 3050 (4K)";
        }
    }

    async onEnterChatMode() {
        this.stopMouthSpeaking();
        this._forEachEngine((eng) => eng.speakStop());

        try {
            await fetch("/voice-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled: false }),
            });
            this.voiceEnabled = false;
            if (this.updateVoiceUI) this.updateVoiceUI(false);
            if (this.chatMicLabel) this.chatMicLabel.textContent = "MIC UNLOADED";
            this.logEvent("AUDIO", "Switched to Chat Workspace: Microphone unloaded and hardware released.", "system");
        } catch (e) {
            console.error("Failed to unload mic on entering chat mode:", e);
        }

        this.syncChatTopBar();

        if (!this.chatSessions || this.chatSessions.length === 0) {
            await this.loadChatSessions();
        }

        if (this.chatTextarea) {
            this.chatTextarea.focus();
        }
    }

    async onExitChatMode() {
        try {
            await fetch("/voice-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled: true }),
            });
            this.voiceEnabled = true;
            if (this.updateVoiceUI) this.updateVoiceUI(true);
            this.logEvent("AUDIO", "Returned to Voice Interface: Microphone restored.", "system");
        } catch (e) {
            console.error("Failed to restore voice mode:", e);
        }
    }

    async loadChatSessions() {
        if (!this.chatSessionsList) return;
        try {
            const res = await fetch("/chat/sessions");
            if (!res.ok) return;
            const data = await res.json();
            this.chatSessions = data.sessions || [];
            this.renderSessionsList();

            if (!this.activeChatSessionId) {
                if (this.chatSessions.length > 0) {
                    this.selectChatSession(this.chatSessions[0].id);
                } else {
                    this.createChatSession();
                }
            }
        } catch (e) {
            console.error("Failed to load chat sessions:", e);
        }
    }

    renderSessionsList() {
        if (!this.chatSessionsList) return;
        this.chatSessionsList.innerHTML = "";
        if (this.chatSessions.length === 0) {
            this.chatSessionsList.innerHTML = '<div class="chat-session-empty">No conversations yet.<br>Click "+ New Chat" to start.</div>';
            return;
        }

        this.chatSessions.forEach((s) => {
            const item = document.createElement("div");
            item.className = `chat-session-item ${s.id === this.activeChatSessionId ? "active" : ""}`;
            item.setAttribute("data-session-id", s.id);

            const title = s.title || "Conversation";
            let timeStr = "";
            const rawDate = s.updated_at || s.started_at || s.created_at;
            if (rawDate) {
                const date = new Date(rawDate);
                const now = new Date();
                const isToday = date.toDateString() === now.toDateString();
                const yesterday = new Date(now);
                yesterday.setDate(yesterday.getDate() - 1);
                const isYesterday = date.toDateString() === yesterday.toDateString();
                if (isToday) {
                    timeStr = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                } else if (isYesterday) {
                    timeStr = "Yesterday";
                } else {
                    timeStr = date.toLocaleDateString([], { month: "short", day: "numeric" });
                }
            }

            item.innerHTML = `
                <div class="session-info">
                    <div class="session-title" title="${this.escapeHtml(title)}">${this.escapeHtml(title)}</div>
                    <div class="session-date">${timeStr}</div>
                </div>
                <button class="session-del-btn" title="Delete conversation">✕</button>
            `;

            item.addEventListener("click", (e) => {
                if (e.target.closest(".session-del-btn")) return;
                this.selectChatSession(s.id);
            });

            const delBtn = item.querySelector(".session-del-btn");
            if (delBtn) {
                delBtn.addEventListener("click", (e) => {
                    e.stopPropagation();
                    this.deleteChatSession(s.id);
                });
            }

            this.chatSessionsList.appendChild(item);
        });
    }

    async createChatSession() {
        try {
            const res = await fetch("/chat/sessions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title: "New Conversation" })
            });
            if (res.ok) {
                const data = await res.json();
                this.activeChatSessionId = data.session_id;
                await this.loadChatSessions();
                this.selectChatSession(data.session_id);
            }
        } catch (e) {
            console.error("Failed to create chat session:", e);
        }
    }

    async selectChatSession(sessionId) {
        this.activeChatSessionId = sessionId;
        this.renderSessionsList();

        if (this.chatSessionIdPill) {
            this.chatSessionIdPill.textContent = `SESSION: ${sessionId.substring(0, 8)}...`;
        }

        try {
            const res = await fetch(`/chat/sessions/${sessionId}`);
            if (!res.ok) return;
            const data = await res.json();

            if (this.chatActiveTitle) {
                this.chatActiveTitle.textContent = data.title || "Conversation";
            }

            if (!this.chatMessagesContainer) return;
            this.chatMessagesContainer.innerHTML = "";

            if (!data.messages || data.messages.length === 0) {
                this.renderChatWelcome();
                return;
            }

            data.messages.forEach((msg) => {
                this.appendHistoricalMessage(msg);
            });
            this.scrollChatToBottom();
        } catch (e) {
            console.error("Failed to select chat session:", e);
        }
    }

    async deleteChatSession(sessionId) {
        if (!confirm("Are you sure you want to delete this conversation thread?")) return;
        try {
            const res = await fetch(`/chat/sessions/${sessionId}`, { method: "DELETE" });
            if (res.ok) {
                if (this.activeChatSessionId === sessionId) {
                    this.activeChatSessionId = null;
                }
                await this.loadChatSessions();
            }
        } catch (e) {
            console.error("Failed to delete session:", e);
        }
    }

    renderChatWelcome() {
        if (!this.chatMessagesContainer) return;
        const persona = (this.personaSelect ? this.personaSelect.value : "neo").toUpperCase();
        this.chatMessagesContainer.innerHTML = `
            <div class="chat-welcome-hero">
                <div class="hero-logo">🤖</div>
                <h2 class="hero-title">${this.escapeHtml(persona)} AI WORKSPACE</h2>
                <p class="hero-desc">
                    Isolated conversational threads with shared global memory across all instances and voice turns.
                    Microphone is unloaded to preserve audio privacy and hardware bandwidth.
                </p>
                <div class="hero-features">
                    <div class="hero-feat"><span class="feat-icon">⚡</span> Real-time token streaming with live tok/s</div>
                    <div class="hero-feat"><span class="feat-icon">🛠️</span> Interactive tool execution telemetry cards</div>
                    <div class="hero-feat"><span class="feat-icon">🧠</span> Shared persistent vector facts & context memory</div>
                </div>
            </div>
        `;
    }

    appendHistoricalMessage(msg) {
        if (!this.chatMessagesContainer) return;
        const isUser = msg.role === "user";

        // Render system bootstrap / arrival prompts gracefully as subtle badges
        if (isUser && msg.content && (msg.content.startsWith("[System") || msg.content.startsWith("[SYSTEM"))) {
            const sysRow = document.createElement("div");
            sysRow.className = "chat-msg-row msg-system-event";
            sysRow.innerHTML = `
                <div class="chat-system-badge">
                    <span class="system-icon">⚡</span>
                    <span class="system-text">${this.escapeHtml(msg.content)}</span>
                </div>
            `;
            this.chatMessagesContainer.appendChild(sysRow);
            return;
        }

        const row = document.createElement("div");
        row.className = `chat-msg-row ${isUser ? "msg-user" : "msg-assistant"}`;

        let toolHtml = "";
        if (!isUser && msg.tool_name) {
            let resStr = "";
            if (typeof msg.tool_result === "object") {
                resStr = JSON.stringify(msg.tool_result, null, 2);
            } else {
                resStr = String(msg.tool_result || "Tool execution completed.");
            }
            toolHtml = `
                <div class="chat-tool-card completed">
                    <div class="tool-hdr">
                        <span class="tool-icon">⚡</span>
                        <span class="tool-title">TOOL EXECUTED: <strong>${this.escapeHtml(msg.tool_name)}</strong></span>
                    </div>
                    <div class="tool-result-box">
                        <pre class="tool-result-content">${this.escapeHtml(resStr)}</pre>
                    </div>
                </div>
            `;
        }

        let timeStr = "";
        if (msg.timestamp) {
            const d = new Date(msg.timestamp);
            timeStr = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }

        const senderLabel = isUser ? "YOU" : (msg.personality ? msg.personality.toUpperCase() : (this.personaSelect ? this.personaSelect.value.toUpperCase() : "ROBOT"));
        const avatarIcon = isUser ? "👤" : "🤖";
        const contentHtml = isUser ? this.escapeHtml(msg.content) : this.renderChatMarkdown(msg.content);

        row.innerHTML = `
            <div class="chat-msg-avatar">${avatarIcon}</div>
            <div class="chat-msg-content">
                <div class="chat-msg-header">
                    <span class="chat-msg-sender">${senderLabel}</span>
                    <span class="chat-msg-time">${timeStr}</span>
                </div>
                ${toolHtml}
                <div class="chat-msg-body">${contentHtml}</div>
            </div>
        `;
        this.chatMessagesContainer.appendChild(row);
    }

    async sendChatMessage(promptText) {
        if (!this.chatTextarea) return;
        const text = (promptText || this.chatTextarea.value).trim();
        if (!text || this.chatStreaming) return;

        if (!this.activeChatSessionId) {
            await this.createChatSession();
        }

        // Remove welcome hero if present
        const hero = this.chatMessagesContainer?.querySelector(".chat-welcome-hero");
        if (hero) hero.remove();

        this.chatStreaming = true;
        if (this.btnChatSubmit) this.btnChatSubmit.disabled = true;
        if (this.btnChatStopStream) this.btnChatStopStream.style.display = "inline-flex";
        this.chatTextarea.value = "";
        this.chatTextarea.style.height = "auto";

        // Append User bubble
        const userRow = document.createElement("div");
        userRow.className = "chat-msg-row msg-user";
        userRow.innerHTML = `
            <div class="chat-msg-avatar">👤</div>
            <div class="chat-msg-content">
                <div class="chat-msg-header">
                    <span class="chat-msg-sender">YOU</span>
                    <span class="chat-msg-time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
                <div class="chat-msg-body">${this.escapeHtml(text)}</div>
            </div>
        `;
        this.chatMessagesContainer.appendChild(userRow);
        this.scrollChatToBottom();

        // Create Assistant Streaming row
        const streamId = `stream_${Date.now()}`;
        const currentPersona = (this.personaSelect ? this.personaSelect.value : "neo").toUpperCase();
        const isGemini = this.btnLlmToggle && this.btnLlmToggle.classList.contains("gemini-mode");
        const currentModel = isGemini ? "GEMINI (1M)" : "LOCAL QWEN (4K)";

        const assistantRow = document.createElement("div");
        assistantRow.className = "chat-msg-row msg-assistant";
        assistantRow.id = `row_${streamId}`;
        assistantRow.innerHTML = `
            <div class="chat-msg-avatar">🤖</div>
            <div class="chat-msg-content">
                <div class="chat-msg-header">
                    <span class="chat-msg-sender">${currentPersona}</span>
                    <span class="chat-msg-badge">${currentModel}</span>
                    <span class="chat-msg-time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
                <div class="chat-tool-container" id="toolBox_${streamId}"></div>
                <div class="chat-msg-body">
                    <span class="chat-text-stream" id="streamText_${streamId}"></span>
                    <span class="chat-cursor" id="streamCursor_${streamId}"></span>
                </div>
                <div class="chat-msg-telemetry" id="streamTelem_${streamId}" style="display: none;"></div>
            </div>
        `;
        this.chatMessagesContainer.appendChild(assistantRow);
        this.scrollChatToBottom();

        const textSpan = document.getElementById(`streamText_${streamId}`);
        const cursorSpan = document.getElementById(`streamCursor_${streamId}`);
        const toolBox = document.getElementById(`toolBox_${streamId}`);
        const telemDiv = document.getElementById(`streamTelem_${streamId}`);

        this.chatAbortController = new AbortController();
        let streamedRawText = "";
        let tokenCount = 0;
        let streamStartTime = performance.now();

        try {
            const res = await fetch("/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    session_id: this.activeChatSessionId,
                    personality: this.personaSelect ? this.personaSelect.value : undefined
                }),
                signal: this.chatAbortController.signal
            });

            if (!res.ok) {
                throw new Error(`HTTP error ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop(); // keep partial

                for (const chunk of lines) {
                    if (!chunk.trim()) continue;

                    // Backend sends: "data: {\"event\": \"token\", ...}\n\n"
                    // Extract the JSON from the data: field
                    const dataMatch = chunk.match(/^data:\s*(.+)$/m);
                    const dataStr = dataMatch ? dataMatch[1].trim() : chunk.trim();

                    let payload = {};
                    try { payload = JSON.parse(dataStr); } catch (e) { continue; }

                    // Event type is embedded inside the JSON payload
                    const eventType = payload.event || "message";

                    if (eventType === "start") {
                        streamStartTime = performance.now();
                        if (this.chatLiveSpeed) {
                            this.chatLiveSpeed.style.display = "inline";
                        }
                    } else if (eventType === "token") {
                        if (payload.token) {
                            streamedRawText += payload.token;
                            tokenCount++;
                            const elapsedSec = (performance.now() - streamStartTime) / 1000;
                            const speed = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : "0.0";
                            if (this.chatLiveSpeed) this.chatLiveSpeed.textContent = `${speed} tok/s`;
                            if (textSpan) textSpan.innerHTML = this.renderChatMarkdown(streamedRawText);
                            this.scrollChatToBottom();
                        }
                    } else if (eventType === "tool_call") {
                        if (toolBox) {
                            const card = document.createElement("div");
                            card.className = "chat-tool-card running";
                            card.id = `toolCard_${payload.tool}`;
                            card.innerHTML = `
                                <div class="tool-hdr">
                                    <span class="tool-spinner"></span>
                                    <span class="tool-title">⚡ Calling Tool: <strong>${this.escapeHtml(payload.tool)}</strong></span>
                                </div>
                                <pre class="tool-args">${this.escapeHtml(JSON.stringify(payload.args || {}, null, 2))}</pre>
                            `;
                            toolBox.appendChild(card);
                            this.scrollChatToBottom();
                        }
                    } else if (eventType === "tool_result") {
                        if (toolBox) {
                            const existingCard = document.getElementById(`toolCard_${payload.tool}`);
                            let resStr = typeof payload.result === "object" ? JSON.stringify(payload.result, null, 2) : String(payload.result);
                            const completedCardHtml = `
                                <div class="tool-hdr">
                                    <span class="tool-icon">✓</span>
                                    <span class="tool-title">Tool Executed: <strong>${this.escapeHtml(payload.tool)}</strong></span>
                                </div>
                                <div class="tool-result-box">
                                    <pre class="tool-result-content">${this.escapeHtml(resStr)}</pre>
                                </div>
                            `;
                            if (existingCard) {
                                existingCard.className = "chat-tool-card completed";
                                existingCard.innerHTML = completedCardHtml;
                            } else {
                                const newCard = document.createElement("div");
                                newCard.className = "chat-tool-card completed";
                                newCard.innerHTML = completedCardHtml;
                                toolBox.appendChild(newCard);
                            }
                            this.scrollChatToBottom();
                        }
                    } else if (eventType === "done") {
                        if (cursorSpan) cursorSpan.remove();
                        if (telemDiv) {
                            telemDiv.style.display = "flex";
                            const durationSec = payload.duration_ms ? (payload.duration_ms / 1000) : 0;
                            telemDiv.innerHTML = `
                                <span class="telem-item">⚡ ${payload.tokens || tokenCount} tokens</span>
                                <span class="telem-item">🚀 ${(payload.tok_per_sec || 0).toFixed(1)} tok/s</span>
                                <span class="telem-item">⏱️ ${durationSec.toFixed(2)}s</span>
                                <span class="telem-item">🎯 TTFT: ${(payload.ttft_ms || 0).toFixed(0)} ms</span>
                            `;
                        }
                        if (this.chatLiveSpeed) {
                            this.chatLiveSpeed.style.display = "none";
                        }
                        this.loadChatSessions();
                    } else if (eventType === "error") {
                        if (textSpan) {
                            textSpan.innerHTML += `<div class="chat-stream-error">[Error: ${this.escapeHtml(payload.error || "Unknown stream error")}]</div>`;
                        }
                    }
                }
            }
        } catch (err) {
            if (err.name === "AbortError") {
                if (textSpan) textSpan.innerHTML += ` <span class="chat-stream-stopped">[Generation stopped]</span>`;
            } else {
                if (textSpan) textSpan.innerHTML += ` <span class="chat-stream-error">[Error: ${this.escapeHtml(err.message)}]</span>`;
            }
        } finally {
            if (cursorSpan) cursorSpan.remove();
            this.chatStreaming = false;
            this.chatAbortController = null;
            if (this.btnChatSubmit) this.btnChatSubmit.disabled = false;
            if (this.btnChatStopStream) this.btnChatStopStream.style.display = "none";
            if (this.chatLiveSpeed) this.chatLiveSpeed.textContent = "0.0 tok/s";
            this.scrollChatToBottom();
            if (this.chatTextarea) this.chatTextarea.focus();
        }
    }

    stopChatStream() {
        if (this.chatAbortController) {
            this.chatAbortController.abort();
        }
    }

    scrollChatToBottom() {
        if (this.chatMessagesContainer) {
            this.chatMessagesContainer.scrollTop = this.chatMessagesContainer.scrollHeight;
        }
    }

    cleanHumanText(text) {
        if (!text) return "";
        let cleaned = String(text);
        // Strip tool_call XML blocks
        cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "");
        cleaned = cleaned.replace(/<\/?(?:tool_call|function|parameter)[^>]*>/gi, "");
        // Strip explicit motion markers like [motion: wave], [motion: raise_hand left], [motion: lean_in]
        cleaned = cleaned.replace(/\[\s*(?:motion|action|emotion|character\.)\s*:[^\]]*\]/gi, "");
        // Strip general bracketed tags e.g. [motion: wave] or [character.execute_action ...]
        cleaned = cleaned.replace(/\[(?:motion|action|emotion)?:?\s*(?:wave|nod|point|kick|walk|turn|sit|stand|lean_in|raise_hand|stop)[^\]]*\]/gi, "");
        // Strip stage directions in asterisks e.g. *waves*, *smiles*, *nods*
        cleaned = cleaned.replace(/\*(?:waves?|raises?\s+(?:her\s+|his\s+|right\s+|left\s+)?hand|points?|kicks?|nods?|smiles?|stops?|leans?|looks?\s+around|shrugs?|pauses?|blinks?|winks?)[^*]*\*/gi, "");
        // Strip stray unclosed motion tags at end of stream like [motion: or [motion
        cleaned = cleaned.replace(/\[\s*(?:motion|action|emotion)?:?[^\]\n]*$/i, "");
        return cleaned.trim();
    }

    renderChatMarkdown(text) {
        if (!text) return "";
        
        // Strip motion tags and internal avatar directives from human-visible text
        let cleaned = this.cleanHumanText(text);
        if (!cleaned) return "";

        // Extract and preserve code blocks
        const codeBlocks = [];
        cleaned = cleaned.replace(/```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g, (match, lang, code) => {
            const id = `__CODE_BLOCK_${codeBlocks.length}__`;
            const langLabel = lang ? lang.toUpperCase() : "CODE";
            codeBlocks.push(
                `<div class="chat-code-block">` +
                `<div class="code-block-header">` +
                `<span class="code-lang-tag">${this.escapeHtml(langLabel)}</span>` +
                `<button class="code-copy-btn" onclick="navigator.clipboard.writeText(this.closest('.chat-code-block').querySelector('code').innerText).then(() => { this.textContent = 'Copied!'; setTimeout(() => this.textContent = 'Copy', 1500); })">Copy</button>` +
                `</div>` +
                `<pre class="code-pre"><code class="language-${this.escapeHtml(lang)}">${this.escapeHtml(code.trim())}</code></pre>` +
                `</div>`
            );
            return id;
        });

        // HTML escape the text outside code blocks
        let escaped = this.escapeHtml(cleaned);

        // Headers
        escaped = escaped.replace(/^### (.*$)/gim, '<h4 class="chat-md-h4">$1</h4>');
        escaped = escaped.replace(/^## (.*$)/gim, '<h3 class="chat-md-h3">$1</h3>');
        escaped = escaped.replace(/^# (.*$)/gim, '<h2 class="chat-md-h2">$1</h2>');

        // Blockquotes
        escaped = escaped.replace(/^\> (.*$)/gim, '<blockquote class="chat-md-quote">$1</blockquote>');

        // Unordered lists (- item or * item)
        escaped = escaped.replace(/^\s*[-*]\s+(.*$)/gim, '<li class="chat-md-li">$1</li>');
        escaped = escaped.replace(/(<li class="chat-md-li">[\s\S]*?<\/li>)/gi, '<ul class="chat-md-ul">$1</ul>');
        // Fix contiguous ul tags
        escaped = escaped.replace(/<\/ul>\s*<ul class="chat-md-ul">/g, "");

        // Inline code: `code`
        escaped = escaped.replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>');

        // Bold: **text**
        escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic: *text* (excluding already formatted list items and bold)
        escaped = escaped.replace(/(^|[^\*])\*([^*]+)\*([^\*]|$)/g, '$1<em>$2</em>$3');

        // Convert double newlines to paragraph breaks and single newlines to <br>
        escaped = escaped.replace(/\n\n+/g, '</p><p>');
        escaped = escaped.replace(/\n/g, '<br>');
        escaped = `<p>${escaped}</p>`;

        // Cleanup empty paragraphs
        escaped = escaped.replace(/<p>\s*<\/p>/g, "");

        // Restore code blocks
        codeBlocks.forEach((block, idx) => {
            escaped = escaped.replace(`__CODE_BLOCK_${idx}__`, block);
            // Also handle if paragraph wrapped the placeholder
            escaped = escaped.replace(`<p>${block}</p>`, block);
        });

        return escaped;
    }

    escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    init3DCharacter() {
        // ─── Animated 3D Character Engine (Cyber variant) ──────────────────
        if (typeof CharacterEngine !== "undefined" && this.characterContainer) {
            this.characterEngine = new CharacterEngine(this.characterContainer, {
                variant: "animated",
                modelUrl: "/static/models/avatar.glb",
            });
            this.logEvent("3D-CORE", "Animated 3D character initialized (WebGL2).", "system");

            // Update debug metrics every 500ms
            setInterval(() => {
                const eng = this._activeCharEngine();
                if (eng && this.dbgCharFps) {
                    this.dbgCharFps.textContent = `FPS: ${eng.fps} (${eng.frameTimeMs.toFixed(1)}ms)`;
                }
            }, 500);
        }

        // Character Action Debug Buttons — dispatch to the animated engine
        document.querySelectorAll(".dbg-act-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const act = btn.getAttribute("data-act");
                const side = btn.getAttribute("data-side") || null;
                const emo = btn.getAttribute("data-emo") || null;
                this.executeSemanticAction({ action: act, side: side, emotion: emo });
            });
        });
    }

    /** Returns the active animated CharacterEngine */
    _activeCharEngine() {
        return this.characterEngine;
    }

    /** Execute a callback on the active CharacterEngine */
    _forEachEngine(fn) {
        if (this.characterEngine) fn(this.characterEngine);
    }

    async executeSemanticAction(actionObj) {
        // Dispatch to ALL initialized character engines (animated + realistic)
        this._forEachEngine((eng) => eng.executeAction(actionObj));
        if (this.charActionBadge) {
            this.charActionBadge.textContent = `MOTION: ${(actionObj.action || "").toUpperCase()}`;
        }
        if (this.dbgCharMotion) {
            this.dbgCharMotion.textContent = `${(actionObj.action || "").toUpperCase()} ${(actionObj.side || "")}`.trim();
        }
        this.logEvent("ACTION", `Action executed: ${JSON.stringify(actionObj)}`, "system");

        // Also post to REST endpoint
        try {
            await fetch("/character/action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(actionObj),
            });
        } catch (e) {}
    }

    initViewControls() {
        const btnStop = document.getElementById("btnStopExecution");
        const btnDual = document.getElementById("btnViewDual");
        const btnDev = document.getElementById("btnViewDev");
        const btnChat = document.getElementById("btnViewChat");
        const btnFs = document.getElementById("btnFullscreen");
        const btnClear = document.getElementById("btnClearLog");

        if (btnStop) {
            btnStop.addEventListener("click", async () => {
                try {
                    await fetch("/interrupt", { method: "POST" });
                } catch (e) {}
                if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send(JSON.stringify({ type: "interrupt" }));
                }
                if (this.characterEngine && this.characterEngine.animation) {
                    this.characterEngine.animation.interrupt();
                }
                this.setRobotMode("idle");
                this.logEvent("INTERRUPT", "Manual stop triggered by user", "warning");
            });
        }

        if (btnDual) btnDual.addEventListener("click", () => this.setView("dual"));
        if (btnDev) btnDev.addEventListener("click", () => this.setView("dev"));
        if (btnChat) btnChat.addEventListener("click", () => this.setView("chat"));

        btnFs.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        });

        btnClear.addEventListener("click", () => {
            this.terminalFeed.innerHTML = "";
        });

        // ─── Camera Box Resizing Controls ──────────────────────────────
        const cameraBox = document.getElementById("cameraBox");
        const btnReset = document.getElementById("btnStreamReset");
        const btnExpand = document.getElementById("btnStreamExpand");
        const resizeHandle = document.getElementById("resizeHandle");

        if (btnReset && cameraBox) {
            btnReset.addEventListener("click", (e) => {
                e.stopPropagation();
                cameraBox.style.width = "240px";
                cameraBox.style.height = "180px";
            });
        }

        if (btnExpand && cameraBox) {
            let isExpanded = false;
            btnExpand.addEventListener("click", (e) => {
                e.stopPropagation();
                if (!isExpanded) {
                    cameraBox.style.width = "480px";
                    cameraBox.style.height = "360px";
                    btnExpand.textContent = "⤡";
                    isExpanded = true;
                } else {
                    cameraBox.style.width = "240px";
                    cameraBox.style.height = "180px";
                    btnExpand.textContent = "⤢";
                    isExpanded = false;
                }
            });
        }

        // Custom mouse drag handle for smooth resizing
        if (resizeHandle && cameraBox) {
            let isDragging = false;
            let startX, startY, startWidth, startHeight;

            resizeHandle.addEventListener("mousedown", (e) => {
                isDragging = true;
                startX = e.clientX;
                startY = e.clientY;
                startWidth = parseInt(document.defaultView.getComputedStyle(cameraBox).width, 10);
                startHeight = parseInt(document.defaultView.getComputedStyle(cameraBox).height, 10);
                document.documentElement.addEventListener("mousemove", onMouseMove, false);
                document.documentElement.addEventListener("mouseup", onMouseUp, false);
                e.preventDefault();
            });

            const onMouseMove = (e) => {
                if (!isDragging) return;
                const newWidth = Math.max(160, Math.min(700, startWidth + (e.clientX - startX)));
                const newHeight = Math.max(120, Math.min(520, startHeight + (e.clientY - startY)));
                cameraBox.style.width = `${newWidth}px`;
                cameraBox.style.height = `${newHeight}px`;
            };

            const onMouseUp = () => {
                isDragging = false;
                document.documentElement.removeEventListener("mousemove", onMouseMove, false);
                document.documentElement.removeEventListener("mouseup", onMouseUp, false);
            };
        }
    }

    async initPersonalitySelector() {
        if (!this.personaSelect) return;

        // Fetch available personalities and current active
        try {
            const res = await fetch("/personalities");
            if (res.ok) {
                const data = await res.json();
                if (data.personalities && Array.isArray(data.personalities)) {
                    this.personaSelect.innerHTML = "";
                    data.personalities.forEach((p) => {
                        const opt = document.createElement("option");
                        opt.value = p;
                        const label = p === "neo" ? "CYPHER (Matrix Hacker Girl)"
                                    : p === "luna" ? "LUNA (Devoted Girlfriend)"
                                    : p === "aria" ? "ARIA (Empathetic / Warm)"
                                    : p === "glados" ? "GLADOS (Sarcastic / Clinical)"
                                    : p.toUpperCase();
                        opt.textContent = label;
                        this.personaSelect.appendChild(opt);
                    });
                }
                const savedPersona = localStorage.getItem("robot_active_personality");
                if (savedPersona && savedPersona !== data.active && data.personalities && data.personalities.includes(savedPersona)) {
                    await fetch(`/personality/${savedPersona}`, { method: "POST" });
                    data.active = savedPersona;
                }
                if (data.active) {
                    this.personaSelect.value = data.active;
                    const activeDisplay = data.active === "neo" ? "CYPHER" : data.active.toUpperCase();
                    if (this.personaName) this.personaName.textContent = activeDisplay;
                }
            }
        } catch (e) {}

        this.personaSelect.addEventListener("change", async (e) => {
            const chosen = e.target.value;
            try {
                localStorage.setItem("robot_active_personality", chosen);
                const switchRes = await fetch(`/personality/${chosen}`, { method: "POST" });
                if (switchRes.ok) {
                    const disp = chosen === "neo" ? "CYPHER" : chosen.toUpperCase();
                    this.logEvent("PERSONA", `Personality switched to ${disp}`, "system");
                    if (this.personaName) this.personaName.textContent = disp;
                    this.syncChatTopBar();
                }
            } catch (err) {
                this.logEvent("ERROR", `Failed to switch personality: ${err.message}`, "error");
            }
        });
    }

    async initLlmProviderToggle() {
        const btn = document.getElementById("btnLlmToggle");
        const icon = document.getElementById("llmToggleIcon");
        const label = document.getElementById("llmToggleLabel");
        const sub = document.getElementById("llmToggleSub");
        const badge = document.getElementById("llmModeBadge");

        const panelStrip = document.getElementById("activeModelStrip");
        const panelBadge = document.getElementById("activeModelBadge");
        const panelName = document.getElementById("activeModelName");
        const panelBtn = document.getElementById("btnSwapModelPanel");
        const panelBtnText = document.getElementById("swapModelBtnText");
        const panelBtnIcon = document.getElementById("swapBtnIcon");

        this.updateLlmToggleUi = (prov, mode, isTransitioning = false) => {
            const isGemini = prov === "gemini";

            if (isTransitioning) {
                if (btn) btn.classList.add("switching");
                if (panelBtn) panelBtn.disabled = true;
                if (label) label.textContent = isGemini ? "STARTING LOCAL..." : "CONNECTING GEMINI...";
                if (sub) sub.textContent = isGemini ? "ALLOCATING VRAM" : "FREEING 3.4GB VRAM";
                if (badge) badge.textContent = "WAIT";
                if (panelName) panelName.textContent = isGemini ? "Starting Local Qwen (Allocating VRAM)..." : "Connecting Gemini (Freeing VRAM)...";
                if (panelBtnText) panelBtnText.textContent = "SWAPPING...";
                return;
            }

            if (btn) btn.classList.remove("switching");
            if (panelBtn) panelBtn.disabled = false;

            if (isGemini) {
                if (btn) {
                    btn.classList.remove("local-mode");
                    btn.classList.add("gemini-mode");
                }
                if (panelStrip) panelStrip.classList.add("gemini-active");
                if (icon) icon.textContent = "✨";
                if (label) label.textContent = "GEMINI (CLOUD)";
                if (sub) sub.textContent = "1M CONTEXT • HIGH INTEL";
                if (badge) badge.textContent = "GEMINI";
                if (panelBadge) panelBadge.textContent = "CLOUD API";
                if (panelName) panelName.textContent = "Gemini 3.5 Flash Lite • 1M Context";
                if (panelBtnText) panelBtnText.textContent = "SWAP TO LOCAL";
                if (panelBtnIcon) panelBtnIcon.textContent = "⚡";
                if (btn) btn.title = "Active: Cloud Gemini 3.5 Flash Lite (1M Context, High Intelligence). Click to swap to Local Qwen (RTX 3050).";
            } else {
                if (btn) {
                    btn.classList.remove("gemini-mode");
                    btn.classList.add("local-mode");
                }
                if (panelStrip) panelStrip.classList.remove("gemini-active");
                if (icon) icon.textContent = "⚡";
                if (label) label.textContent = "LOCAL (RTX 3050)";
                if (sub) sub.textContent = "LOW LATENCY • 4K";
                if (badge) badge.textContent = "LOCAL";
                if (panelBadge) panelBadge.textContent = "LOCAL GPU";
                if (panelName) panelName.textContent = "Qwen 3.5 4B • RTX 3050 • 4K";
                if (panelBtnText) panelBtnText.textContent = "SWAP TO GEMINI";
                if (panelBtnIcon) panelBtnIcon.textContent = "✨";
                if (btn) btn.title = "Active: Local Qwen 3.5 4B on RTX 3050 (Low Latency). Click to swap to Cloud Gemini (1M Context).";
            }
            if (typeof this.syncChatTopBar === "function") this.syncChatTopBar();
        };

        // Fetch current status on load
        try {
            const res = await fetch("/api/llm/status");
            if (res.ok) {
                const data = await res.json();
                this.updateLlmToggleUi(data.provider, data.mode);
            }
        } catch (e) {}

        const handleSwapAction = async () => {
            const isCurrentlyGemini = (panelStrip && panelStrip.classList.contains("gemini-active")) || (btn && btn.classList.contains("gemini-mode"));
            const target = isCurrentlyGemini ? "local" : "gemini";
            this.updateLlmToggleUi(isCurrentlyGemini ? "gemini" : "local", null, true);
            this.logEvent("LLM", `Swapping LLM provider to ${target.toUpperCase()}...`, "system");
            try {
                const res = await fetch("/api/llm/provider", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ provider: target })
                });
                const data = await res.json();
                if (data.status === "ok") {
                    this.updateLlmToggleUi(data.provider, data.mode);
                    const msg = data.provider === "gemini" 
                        ? "Swapped to Cloud Gemini 3.6 Flash (1M Context)! Local llama-server killed & 3.4GB VRAM released."
                        : "Swapped to Local Qwen 3.5 4B on RTX 3050! llama-server spun up & slot cache primed.";
                    this.logEvent("LLM", msg, "success");
                } else {
                    this.updateLlmToggleUi(isCurrentlyGemini ? "gemini" : "local");
                    this.logEvent("ERROR", `Failed to switch LLM provider: ${data.message || data.error}`, "error");
                }
            } catch (err) {
                this.updateLlmToggleUi(isCurrentlyGemini ? "gemini" : "local");
                this.logEvent("ERROR", `Network error switching LLM provider: ${err.message}`, "error");
            }
        };

        if (btn) btn.addEventListener("click", handleSwapAction);
        if (panelBtn) panelBtn.addEventListener("click", handleSwapAction);
    }

    setView(mode) {
        document.body.className = `mode-${mode}`;
        document.querySelectorAll(".view-btn").forEach(btn => btn.classList.remove("active"));
        if (mode === "dual") document.getElementById("btnViewDual")?.classList.add("active");
        if (mode === "dev") document.getElementById("btnViewDev")?.classList.add("active");
        if (mode === "chat") {
            document.getElementById("btnViewChat")?.classList.add("active");
            this.onEnterChatMode();
        } else {
            this.onExitChatMode();
        }
    }

    initBlinking() {
        // Humanoid natural asynchronous blinking
        const doBlink = () => {
            const eyes = document.querySelectorAll(".eye");
            eyes.forEach(e => e.classList.add("blink"));
            setTimeout(() => {
                eyes.forEach(e => e.classList.remove("blink"));
            }, 120);

            // Double blink chance
            if (Math.random() < 0.25) {
                setTimeout(() => {
                    eyes.forEach(e => e.classList.add("blink"));
                    setTimeout(() => eyes.forEach(e => e.classList.remove("blink")), 110);
                }, 240);
            }

            const nextBlink = Math.random() * 3500 + 2000;
            this.blinkInterval = setTimeout(doBlink, nextBlink);
        };
        this.blinkInterval = setTimeout(doBlink, 2000);
    }

    setRobotMode(mode) {
        if (!mode) return;
        const normalizedMode = mode.toLowerCase();
        if (this.currentMode === normalizedMode) return;
        this.currentMode = normalizedMode;

        // Reset stage state classes if faceStage exists
        if (this.faceStage) {
            this.faceStage.classList.remove("state-idle", "state-listening", "state-thinking", "state-speaking");
            this.faceStage.classList.add(`state-${normalizedMode}`);
        }

        // Update Status Pill
        if (this.statusPill) this.statusPill.className = `status-pill status-${normalizedMode}`;
        if (this.statusText) this.statusText.textContent = normalizedMode.toUpperCase();

        // Update Face Caption & 3D Character Posture / Expression
        if (this.charStateBadge) this.charStateBadge.textContent = `STATE: ${normalizedMode.toUpperCase()}`;
        if (this.dbgCharState) this.dbgCharState.textContent = normalizedMode.toUpperCase();

        switch (normalizedMode) {
            case "listening":
                if (this.faceCaption) this.faceCaption.textContent = "LISTENING FOR VOICE";
                this.stopMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.setEmotion("neutral");
                    eng.lookAt(0, 1.45, 2.0);
                });
                break;
            case "thinking":
                if (this.faceCaption) this.faceCaption.textContent = "PROCESSING & REASONING";
                this.stopMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.setEmotion("thinking");
                    eng.lookAt(0.3, 1.6, 2.2);
                });
                break;
            case "speaking":
                if (this.faceCaption) this.faceCaption.textContent = "SYNTHESIZING & SPEAKING";
                this.startMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.speakStart();
                    eng.setEmotion("happy");
                });
                break;
            default:
                if (this.faceCaption) this.faceCaption.textContent = "OBSERVING ENVIRONMENT";
                this.stopMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.speakStop();
                    eng.setEmotion("neutral");
                });
                break;
        }
    }

    startMouthSpeaking() {
        if (this.mouthInterval) clearInterval(this.mouthInterval);
        this.mouthInterval = setInterval(() => {
            this.mouthBars.forEach((bar, idx) => {
                const height = Math.floor(Math.random() * 38) + 8;
                bar.style.height = `${height}px`;
            });
        }, 75);
    }

    stopMouthSpeaking() {
        if (this.mouthInterval) {
            clearInterval(this.mouthInterval);
            this.mouthInterval = null;
        }
        this.mouthBars.forEach(b => b.style.height = "10px");
    }

    logEvent(tag, msg, type = "system") {
        const time = new Date().toTimeString().split(" ")[0];
        const entry = document.createElement("div");
        entry.className = `log-entry log-${type}`;
        entry.innerHTML = `<span class="timestamp">[${time}]</span><span class="tag">[${tag}]</span><span class="msg">${msg}</span>`;
        this.terminalFeed.appendChild(entry);
        this.terminalFeed.scrollTop = this.terminalFeed.scrollHeight;

        // Cap log entries
        while (this.terminalFeed.children.length > 100) {
            this.terminalFeed.removeChild(this.terminalFeed.firstChild);
        }
    }

    initWebsocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/events`;

        const connect = () => {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                this.logEvent("WS", "Real-time telemetry event stream connected.", "system");
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleServerEvent(data);
                } catch (e) {
                    console.error("WS Parse error", e);
                }
            };

            this.ws.onclose = () => {
                setTimeout(connect, 2000);
            };
        };

        connect();
    }

    handleServerEvent(data) {
        if (data.event === "llm.provider_changed" && this.updateLlmToggleUi) {
            this.updateLlmToggleUi(data.provider, data.mode);
        }

        if (data.event === "pipeline_updated" || data.pipeline) {
            this.updatePipelineStatus(data.pipeline);
        }

        if (data.mode) {
            this.setRobotMode(data.mode);
        }

        if (data.user_text) {
            if (this.bubbleTag) {
                this.bubbleTag.textContent = "USER SPEECH";
                this.bubbleTag.style.color = "var(--primary-cyan)";
            }
            if (this.bubbleContent) {
                this.bubbleContent.textContent = data.user_text;
            }
            if (this._lastUserLogged !== data.user_text) {
                this.logEvent("USER", data.user_text, "user");
                this._lastUserLogged = data.user_text;
            }
            this.updatePipelineStatus({ MIC: "COMPLETED", VAD: "COMPLETED", STT: "COMPLETED", CONTEXT: "RUNNING", LLM: "RUNNING" });
        }

        if (data.event === "robot_speech_ended" || data.robot_speech_ended) {
            this.stopMouthSpeaking();
            this._forEachEngine((eng) => eng.speakStop());
            this.setRobotMode("idle");
            this.updatePipelineStatus({ TTS: "COMPLETED", AVATAR: "IDLE" });
        }

        if (data.music_playing) {
            this.stopMouthSpeaking();
            this._forEachEngine((eng) => eng.speakStop());
            if (this.currentMode === "speaking") {
                this.setRobotMode("idle");
            }
        }

        if (data.robot_phrase) {
            const cleanPhrase = this.cleanHumanText(data.robot_phrase) || data.robot_phrase;
            if (this.bubbleTag) {
                this.bubbleTag.textContent = "ROBOT RESPONSE";
                this.bubbleTag.style.color = "var(--accent-emerald)";
            }
            if (this.bubbleContent) {
                this.bubbleContent.textContent = cleanPhrase;
            }
            if (this._lastRobotLogged !== cleanPhrase) {
                this.logEvent("ROBOT", cleanPhrase, "robot");
                this._lastRobotLogged = cleanPhrase;
            }
            this.updatePipelineStatus({ LLM: "COMPLETED", TTS: "RUNNING", AVATAR: "RUNNING" });

            // Trigger lip-sync on ALL 3D character engines with safe timeout
            const phraseDuration = Math.max(1200, Math.min(10000, data.robot_phrase.length * 80));
            this._forEachEngine((eng) => eng.speakStart());
            if (this._phraseTimeout) clearTimeout(this._phraseTimeout);
            this._phraseTimeout = setTimeout(() => {
                this._forEachEngine((eng) => eng.speakStop());
                this.stopMouthSpeaking();
                if (this.currentMode === "speaking") {
                    this.setRobotMode("idle");
                }
                this.updatePipelineStatus({ TTS: "COMPLETED", AVATAR: "IDLE" });
            }, phraseDuration);
        }

        if (data.barge_in) {
            this.logEvent("BARGE-IN", "Interrupted bot playback via voice activity!", "barge");
            this._forEachEngine((eng) => eng.stopMotion());
            this.setRobotMode("listening");
            this.updatePipelineStatus({ MIC: "RUNNING", VAD: "RUNNING", LLM: "CANCELLED", TTS: "CANCELLED", AVATAR: "IDLE" });
        }

        // Live Tool-Conversation Coordinator State
        if (data.event === "coordinator_state" || data.coordinator_state) {
            const st = data.state || data.coordinator_state;
            const badge = document.getElementById("coordinatorBadge");
            if (badge) {
                badge.style.display = "inline-block";
                badge.textContent = `COORDINATOR: ${st}`;
                if (st === "HOLDING_CONVERSATION" || st === "TOOL_RUNNING") {
                    badge.className = "badge badge-pulsing";
                } else if (st === "COMPLETED") {
                    badge.className = "badge badge-accent";
                }
            }
            const parallelBox = document.getElementById("parallelStreams");
            if (parallelBox) {
                if (st === "HOLDING_CONVERSATION" || st === "TOOL_RUNNING" || st === "TRANSITIONING_TO_RESULT") {
                    parallelBox.style.display = "flex";
                } else if (st === "COMPLETED" || st === "IDLE" || st === "CANCELLED") {
                    setTimeout(() => { if (parallelBox) parallelBox.style.display = "none"; }, 3000);
                }
            }
            const toolStatusEl = document.getElementById("toolStreamStatus");
            if (toolStatusEl && data.tool) {
                toolStatusEl.textContent = `${data.tool.toUpperCase()}: ${st}`;
            }
        }

        // Real-time Semantic Character Actions dispatched from Robot Core / LLM
        if (data.event === "character.action" || data.action) {
            const act = data.action;
            if (act) {
                this._forEachEngine((eng) => eng.executeAction(act));
                if (this.charActionBadge) {
                    this.charActionBadge.textContent = `MOTION: ${(act.action || "").toUpperCase()}`;
                }
                if (this.dbgCharMotion) {
                    this.dbgCharMotion.textContent = `${(act.action || "").toUpperCase()} ${(act.side || "")}`.trim();
                }
                this.logEvent("CHAR", `Executing motion: ${act.action} (${act.side || "center"})`, "system");
            }
        }

        if (data.event === "character.interrupt" || data.interrupt) {
            this._forEachEngine((eng) => eng.stopMotion());
            if (this.charActionBadge) this.charActionBadge.textContent = "MOTION: STOPPED";
            if (this.dbgCharMotion) this.dbgCharMotion.textContent = "INTERRUPTED (IDLE)";
            this.logEvent("CHAR", "Motion interrupted. Blending to neutral idle.", "barge");
        }

        if (data.event === "tool_executing") {
            const toolBadge = document.getElementById("activeToolBadge");
            if (toolBadge) {
                toolBadge.textContent = `TOOL: ${(data.tool || "RUNNING").toUpperCase()}`;
                toolBadge.style.display = "inline-block";
                toolBadge.style.background = "rgba(0, 242, 254, 0.2)";
                toolBadge.style.borderColor = "var(--primary-cyan)";
            }
            this.logEvent("TOOL", `Executing: ${data.tool}`, "system");
            this.updatePipelineStatus({ TOOLS: "RUNNING" });
        }

        if (data.event === "tool_completed" || data.event === "tool_error") {
            const toolBadge = document.getElementById("activeToolBadge");
            if (toolBadge) {
                if (data.event === "tool_completed") {
                    toolBadge.textContent = `TOOL: DONE`;
                    toolBadge.style.background = "rgba(0, 245, 160, 0.2)";
                    toolBadge.style.borderColor = "var(--accent-emerald)";
                } else {
                    toolBadge.textContent = `TOOL: ERR`;
                    toolBadge.style.background = "rgba(255, 87, 87, 0.2)";
                    toolBadge.style.borderColor = "#ff5757";
                }
                setTimeout(() => {
                    if (toolBadge.textContent.includes("DONE") || toolBadge.textContent.includes("ERR")) {
                        toolBadge.style.display = "none";
                    }
                }, 3000);
            }
            this.updatePipelineStatus({ TOOLS: data.event === "tool_completed" ? "COMPLETED" : "ERROR" });
        }

        if (data.metrics) {
            this.updateMetrics(data.metrics);
        }

        if (data.event === "observability.event") {
            this.handleObservabilityEvent(data);
        }
    }

    updateMetrics(metrics) {
        if (metrics.vad !== undefined && this.metricVad) {
            this.metricVad.textContent = `${Math.round(metrics.vad)} ms`;
            if (this.barVad) this.barVad.style.width = `${Math.min(100, (metrics.vad / 500) * 100)}%`;
        }
        if (metrics.stt !== undefined && this.metricStt) {
            this.metricStt.textContent = `${Math.round(metrics.stt)} ms`;
            if (this.barStt) this.barStt.style.width = `${Math.min(100, (metrics.stt / 1000) * 100)}%`;
        }
        if (metrics.context !== undefined && this.metricContext) {
            this.metricContext.textContent = `${Math.round(metrics.context)} ms`;
            if (this.barContext) this.barContext.style.width = `${Math.min(100, (metrics.context / 300) * 100)}%`;
        }
        if (metrics.ttft !== undefined && this.metricTtft) {
            this.metricTtft.textContent = `${Math.round(metrics.ttft)} ms`;
            if (this.hudTtft) this.hudTtft.textContent = `${Math.round(metrics.ttft)} ms`;
            if (this.barTtft) this.barTtft.style.width = `${Math.min(100, (metrics.ttft / 1500) * 100)}%`;
        }
        if (metrics.synth !== undefined && this.metricTts) {
            this.metricTts.textContent = `${Math.round(metrics.synth)} ms`;
            if (this.barTts) this.barTts.style.width = `${Math.min(100, (metrics.synth / 500) * 100)}%`;
        }
        if (metrics.e2e !== undefined && this.metricE2e) {
            this.metricE2e.textContent = `${Math.round(metrics.e2e)} ms`;
            if (this.hudE2e) this.hudE2e.textContent = `${Math.round(metrics.e2e)} ms`;
            if (this.barE2e) this.barE2e.style.width = `${Math.min(100, (metrics.e2e / 2000) * 100)}%`;
        }
        if (this.hudTokens) {
            if (metrics.tokens !== undefined) {
                this.hudTokens.textContent = `${metrics.tokens}`;
            } else if (metrics.tokens_in !== undefined && metrics.tokens_out !== undefined) {
                this.hudTokens.textContent = `${metrics.tokens_in} in / ${metrics.tokens_out} out`;
            } else if (metrics.tokens_in !== undefined) {
                this.hudTokens.textContent = `${metrics.tokens_in} in / 0 out`;
            }
        }
    }

    startPollers() {
        // Poll status every 1.5 seconds for robust state synchronization
        setInterval(async () => {
            try {
                const res = await fetch("/status");
                if (res.ok) {
                    const data = await res.json();
                    if (data.mode) this.setRobotMode(data.mode);
                    if ((data.is_speaking === false || data.mode === "idle") && this.currentMode === "speaking") {
                        this.setRobotMode("idle");
                        this.stopMouthSpeaking();
                        this._forEachEngine((eng) => eng.speakStop());
                    }
                    if (data.music_playing) {
                        this.stopMouthSpeaking();
                        this._forEachEngine((eng) => eng.speakStop());
                        if (this.currentMode === "speaking") {
                            this.setRobotMode("idle");
                        }
                    }
                    if (data.voice_enabled !== undefined && this.updateVoiceUI && this.voiceEnabled !== data.voice_enabled) {
                        this.updateVoiceUI(data.voice_enabled);
                    }
                    if (data.personality) {
                        const pUpper = data.personality.toUpperCase();
                        if (this.personaName) this.personaName.textContent = pUpper;
                        if (this.personaSelect && document.activeElement !== this.personaSelect && this.personaSelect.value !== data.personality) {
                            this.personaSelect.value = data.personality;
                        }
                    }
                    if (data.services) {
                        const dots = {
                            stt: document.querySelector(".h-stt"),
                            llm: document.querySelector(".h-llm"),
                            tts: document.querySelector(".h-tts"),
                            mem: document.querySelector(".h-mem"),
                        };
                        if (dots.stt) dots.stt.className = `h-dot h-stt ${data.services.stt ? "h-up" : "h-down"}`;
                        if (dots.llm) dots.llm.className = `h-dot h-llm ${data.services.llm ? "h-up" : "h-down"}`;
                        if (dots.tts) dots.tts.className = `h-dot h-tts ${data.services.tts ? "h-up" : "h-down"}`;
                        if (dots.mem) dots.mem.className = `h-dot h-mem ${data.services.memory ? "h-up" : "h-down"}`;
                    }
                    if (data.telemetry) {
                        const t = data.telemetry;
                        const metrics = {};
                        metrics.vad = 150; // standard silence threshold
                        if (t.STT_FINAL_LATENCY && t.STT_FINAL_LATENCY.last) metrics.stt = t.STT_FINAL_LATENCY.last;
                        if (t.CONTEXT_BUILD_LATENCY && t.CONTEXT_BUILD_LATENCY.last) metrics.context = t.CONTEXT_BUILD_LATENCY.last;
                        if (t.LLM_TTFT && t.LLM_TTFT.last) metrics.ttft = t.LLM_TTFT.last;
                        if (t.TTS_SYNTHESIS_LATENCY && t.TTS_SYNTHESIS_LATENCY.last) metrics.synth = t.TTS_SYNTHESIS_LATENCY.last;
                        else if (t.TTS_FIRST_AUDIO_LATENCY && t.TTS_FIRST_AUDIO_LATENCY.last) metrics.synth = t.TTS_FIRST_AUDIO_LATENCY.last;
                        if (t.END_OF_SPEECH_TO_FIRST_AUDIO && t.END_OF_SPEECH_TO_FIRST_AUDIO.last) metrics.e2e = t.END_OF_SPEECH_TO_FIRST_AUDIO.last;
                        
                        const tIn = (t.CONTEXT_INPUT_TOKENS && t.CONTEXT_INPUT_TOKENS.last) || (t.CONTEXT_TOKENS && t.CONTEXT_TOKENS.last) || 0;
                        const tOut = (t.LLM_OUTPUT_TOKENS && t.LLM_OUTPUT_TOKENS.last) || 0;
                        if (tIn || tOut) {
                            metrics.tokens = `${tIn} in / ${tOut} out`;
                        }
                        this.updateMetrics(metrics);
                    }
                }
            } catch (e) {}
        }, 1200);

        // Poll Traces & Pipeline State every 1.5 seconds if WebSocket drops or during active turn
        setInterval(async () => {
            try {
                const res = await fetch("/observability/traces?limit=1");
                if (res.ok) {
                    const data = await res.json();
                    if (data.traces && data.traces.length > 0) {
                        this.renderTraceTree(data.traces[0]);
                        const total = data.traces[0].total_latency_ms;
                        const durEl = document.getElementById("traceTotalDuration");
                        if (durEl && total) durEl.textContent = `DURATION: ${total.toFixed(1)} ms`;
                    }
                    if (data.pipeline) {
                        this.updatePipelineStatus(data.pipeline);
                    }
                }
            } catch (e) {}
        }, 1500);

        // Poll World State & Perception at 4Hz (250ms) for responsive face tracking
        setInterval(async () => {
            try {
                const res = await fetch("/world-state");
                if (res.ok) {
                    const ws = await res.json();
                    if (!ws.error) {
                        if (this.worldStateCode) this.worldStateCode.textContent = JSON.stringify(ws, null, 2);
                        const elLight = document.getElementById("hudLighting");
                        if (elLight) elLight.textContent = `LIGHT: ${(ws.lighting || "NORMAL").toUpperCase()}`;
                        const elMotion = document.getElementById("hudMotion");
                        if (elMotion) elMotion.textContent = `MOTION: ${(ws.motion || "NONE").toUpperCase()}`;
                        const elPeople = document.getElementById("hudPeople");
                        if (elPeople) elPeople.textContent = `PEOPLE: ${ws.people_count || 0}`;

                        // Drive 3D Character Gaze & Look-At IK from OpenCV detected user position
                        if (ws.people && ws.people.length > 0 && this.characterEngine) {
                            const user = ws.people[0];
                            const nx = user.norm_x !== undefined ? user.norm_x : 0.5;
                            const ny = user.norm_y !== undefined ? user.norm_y : 0.5;
                            this.characterEngine.setOpenCVUserFace(nx, ny);
                            if (this.charGazeBadge) {
                                this.charGazeBadge.textContent = `GAZE: USER (${user.position.toUpperCase()})`;
                            }
                        } else if (this.characterEngine && this.charGazeBadge) {
                            this.charGazeBadge.textContent = "GAZE: CAMERA";
                        }
                    }
                }
            } catch (e) {}
        }, 1500);

        // Camera Stream Auto-Recovery: /camera/stream provides continuous 60 FPS MJPEG.
        // If the stream ever drops, reconnect gracefully.
        if (this.cameraImg) {
            this.cameraImg.onerror = () => {
                setTimeout(() => {
                    this.cameraImg.src = `/camera/stream?t=${Date.now()}`;
                }, 1000);
            };
        }

        setInterval(async () => {
            try {
                const res = await fetch("/observability/hardware");
                if (res.ok) {
                    const hw = await res.json();
                    if (hw.cpu_percent !== undefined) {
                        document.getElementById("hwCpuVal").textContent = `${hw.cpu_percent.toFixed(1)} %`;
                    }
                    if (hw.ram_used_gb !== undefined) {
                        const total = hw.ram_total_gb ? ` / ${hw.ram_total_gb.toFixed(1)}` : "";
                        document.getElementById("hwRamVal").textContent = `${hw.ram_used_gb.toFixed(1)}${total} GB`;
                    } else if (hw.system_ram_gb !== undefined) {
                        document.getElementById("hwRamVal").textContent = `${hw.system_ram_gb.toFixed(1)} GB`;
                    }
                    if (hw.nvidia && hw.nvidia.vram_used_mb !== undefined) {
                        document.getElementById("hwVramVal").textContent = `${Math.round(hw.nvidia.vram_used_mb)} MB`;
                    } else if (hw.gpu_vram_mb !== undefined) {
                        document.getElementById("hwVramVal").textContent = `${Math.round(hw.gpu_vram_mb)} MB`;
                    }
                    if (hw.nvidia && hw.nvidia.temperature_c !== undefined) {
                        document.getElementById("hwTempVal").textContent = `${Math.round(hw.nvidia.temperature_c)} °C`;
                    } else if (hw.gpu_temp_c !== undefined) {
                        document.getElementById("hwTempVal").textContent = `${Math.round(hw.gpu_temp_c)} °C`;
                    }
                    if (hw.amd && hw.amd.utilization_percent !== undefined) {
                        document.getElementById("hwAmdVal").textContent = `${Math.round(hw.amd.utilization_percent)} %`;
                    } else if (hw.amd_utilization !== undefined) {
                        document.getElementById("hwAmdVal").textContent = `${Math.round(hw.amd_utilization)} %`;
                    }
                }
            } catch (e) {}
        }, 2000);

        setInterval(async () => {
            try {
                const res = await fetch("/images/jobs");
                if (res.ok) {
                    const data = await res.json();
                    if (data.jobs) {
                        this.updateImageJobsUI(data.jobs);
                    }
                }
            } catch (e) {}
        }, 1000);
    }

    handleObservabilityEvent(evt) {
        if (evt.type === "trace_updated" && evt.trace) {
            this.renderTraceTree(evt.trace);
            const total = evt.trace.total_latency_ms;
            const durEl = document.getElementById("traceTotalDuration");
            if (durEl && total) durEl.textContent = `DURATION: ${total.toFixed(1)} ms`;
        }
        if (evt.type === "pipeline_updated" && evt.pipeline) {
            this.updatePipelineStatus(evt.pipeline);
        }
    }

    updatePipelineStatus(pipeline) {
        if (!pipeline) return;
        const nodes = ["mic", "vad", "stt", "context", "llm", "tools", "tts", "avatar", "image"];
        for (let i = 0; i < nodes.length; i++) {
            const key = nodes[i];
            const capKey = key.charAt(0).toUpperCase() + key.slice(1);
            const el = document.getElementById(`node${capKey}`);
            const arrow = document.getElementById(`arrow${capKey}`);
            
            // Only update nodes present in pipeline object if partial update
            if (pipeline[key.toUpperCase()] === undefined && pipeline[key] === undefined) {
                continue;
            }
            
            if (el) {
                const rawStatus = pipeline[key.toUpperCase()] !== undefined ? pipeline[key.toUpperCase()] : pipeline[key];
                const status = String(rawStatus || "IDLE").toLowerCase();
                el.className = "pipe-node";
                const statusEl = el.querySelector(".node-status");
                if (statusEl) statusEl.textContent = String(rawStatus || "IDLE").toUpperCase();

                if (status === "running" || status === "active") {
                    el.classList.add("running");
                    if (arrow) arrow.classList.add("running");
                } else {
                    if (arrow) arrow.classList.remove("running");
                    if (status === "completed" || status === "done") {
                        el.classList.add("completed");
                    } else if (status === "cancelled") {
                        el.classList.add("cancelled");
                    } else if (status === "error" || status === "failed") {
                        el.classList.add("error");
                    }
                }
            }
        }
    }

    renderTraceTree(trace) {
        const box = document.getElementById("traceTreeBox");
        if (!trace || !trace.spans || trace.spans.length === 0) {
            box.innerHTML = '<div class="trace-empty-hint">Waiting for voice turn or tool execution...</div>';
            return;
        }

        let html = '<ul class="trace-tree">';
        const renderSpan = (span, level = 0) => {
            const duration = span.duration_ms ? `${span.duration_ms.toFixed(1)} ms` : (span.end_time ? "0 ms" : "...");
            const indent = level * 15;
            let statusClass = "span-ok";
            if (!span.end_time) statusClass = "span-active";
            if (span.error) statusClass = "span-error";
            
            html += `<li class="span-item" style="padding-left: ${indent}px;">
                        <span class="span-name ${statusClass}">${span.name}</span>
                        <span class="span-dur">${duration}</span>
                    </li>`;
            
            if (span.children) {
                for (const child of span.children) renderSpan(child, level + 1);
            }
        };

        for (const span of trace.spans) renderSpan(span);
        html += '</ul>';
        box.innerHTML = html;
        document.getElementById("activeTraceBadge").textContent = `TRACE: ${trace.trace_id.substring(0,8)}`;
    }

    initImageGen() {
        const btnGen = document.getElementById("btnGenImage");
        const btnCancel = document.getElementById("btnCancelImage");
        const input = document.getElementById("imgPromptInput");

        const submitImage = async () => {
            if (!input) return;
            const prompt = input.value.trim();
            if (!prompt) return;

            const resSelect = document.getElementById("imgResSelect");
            const stepsInput = document.getElementById("imgStepsInput");
            const [width, height] = (resSelect ? resSelect.value : "512x512").split("x").map(Number);
            const steps = stepsInput ? (parseInt(stepsInput.value, 10) || 4) : 4;

            if (btnGen) {
                btnGen.disabled = true;
                btnGen.textContent = "Generating...";
                btnGen.classList.add("generating");
            }
            this.updatePipelineStatus({ IMAGE: "RUNNING" });

            try {
                const res = await fetch("/images/generate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: prompt, width: width, height: height, steps: steps })
                });
                const data = await res.json();
                if (res.ok && !data.error) {
                    input.value = "";
                    this.logEvent("IMAGE", `Generation job submitted: "${prompt}" (${width}x${height}, ${steps} steps)`, "system");
                    try {
                        const pollRes = await fetch("/images/jobs");
                        if (pollRes.ok) {
                            const jobsData = await pollRes.json();
                            if (jobsData.jobs) this.updateImageJobsUI(jobsData.jobs);
                        }
                    } catch (e) {}
                } else {
                    const err = data.error || `HTTP ${res.status}`;
                    this.logEvent("IMAGE", `Image generation error: ${err}`, "system");
                    if (btnGen) {
                        btnGen.disabled = false;
                        btnGen.textContent = "Generate";
                        btnGen.classList.remove("generating");
                    }
                    this.updatePipelineStatus({ IMAGE: "ERROR" });
                }
            } catch (e) {
                console.error("Image gen failed", e);
                this.logEvent("IMAGE", `Failed to send generation request: ${e.message}`, "system");
                if (btnGen) {
                    btnGen.disabled = false;
                    btnGen.textContent = "Generate";
                    btnGen.classList.remove("generating");
                }
                this.updatePipelineStatus({ IMAGE: "ERROR" });
            }
        };

        if (btnGen) {
            btnGen.addEventListener("click", (e) => {
                e.preventDefault();
                submitImage();
            });
        }

        if (input) {
            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    submitImage();
                }
            });
        }

        if (btnCancel) {
            btnCancel.addEventListener("click", async (e) => {
                e.preventDefault();
                if (this.activeImageJob) {
                    await fetch(`/images/cancel/${this.activeImageJob}`, { method: "POST" });
                    this.logEvent("IMAGE", `Cancelled job ${this.activeImageJob}`, "system");
                    this.updatePipelineStatus({ IMAGE: "CANCELLED" });
                }
            });
        }
    }

    updateImageJobsUI(jobs) {
        const badge = document.getElementById("imgJobBadge");
        const btnGen = document.getElementById("btnGenImage");
        const btnCancel = document.getElementById("btnCancelImage");
        const previewImg = document.getElementById("imgPreviewImg");
        const placeholder = document.getElementById("imgPlaceholder");
        const genStatus = document.getElementById("imgGeneratingStatus");
        const metaBlock = document.getElementById("imgMetaBlock");

        let activeJob = jobs.find(j => j.status === "queued" || j.status === "running" || j.status === "QUEUED" || j.status === "RUNNING");
        let completedJob = [...jobs].reverse().find(j => j.status === "completed" || j.status === "COMPLETED");
        if (!activeJob && jobs.length > 0 && (jobs[jobs.length - 1].status === "queued" || jobs[jobs.length - 1].status === "running")) {
            activeJob = jobs[jobs.length - 1];
        }

        this.activeImageJob = activeJob ? activeJob.job_id : null;
        
        if (activeJob) {
            if (badge) badge.textContent = `STATUS: ${activeJob.status.toUpperCase()}`;
            if (btnCancel) btnCancel.disabled = false;
            if (btnGen) {
                btnGen.disabled = true;
                btnGen.textContent = "Generating...";
                btnGen.classList.add("generating");
            }
            this.updatePipelineStatus({ IMAGE: "RUNNING" });
            
            // Show generating status
            if (placeholder) placeholder.style.display = "none";
            if (previewImg) previewImg.style.display = "none";
            if (metaBlock) metaBlock.style.display = "none";
            if (genStatus) genStatus.style.display = "block";
            
            const stepEl = document.getElementById("imgGenStep");
            if (stepEl) stepEl.textContent = `Step ${Math.floor((activeJob.progress / 100) * activeJob.steps)}/${activeJob.steps}`;
            const beEl = document.getElementById("imgGenBackend");
            if (beEl) beEl.textContent = `Backend: ${activeJob.backend || '--'}`;
            const devEl = document.getElementById("imgGenDevice");
            if (devEl) devEl.textContent = `Device: ${activeJob.device || '--'}`;
            const elapsed = activeJob.started_at ? ((Date.now() / 1000) - activeJob.started_at).toFixed(1) : 0.0;
            const elapEl = document.getElementById("imgGenElapsed");
            if (elapEl) elapEl.textContent = `Elapsed: ${elapsed}s`;
            
            const pbar = document.getElementById("imgProgressBar");
            const pwrap = document.getElementById("imgProgressWrap");
            if (pbar && pwrap) {
                pwrap.style.display = "block";
                pbar.style.width = `${activeJob.progress}%`;
            }
        } else {
            if (badge) badge.textContent = "STATUS: READY";
            if (btnCancel) btnCancel.disabled = true;
            if (btnGen) {
                btnGen.disabled = false;
                btnGen.textContent = "Generate";
                btnGen.classList.remove("generating");
            }
            if (genStatus) genStatus.style.display = "none";
            
            const pwrap = document.getElementById("imgProgressWrap");
            if (pwrap) pwrap.style.display = "none";
            
            if (completedJob && completedJob.output_path) {
                if (placeholder) placeholder.style.display = "none";
                if (previewImg) {
                    previewImg.style.display = "block";
                    previewImg.src = completedJob.output_path;
                }
                
                if (metaBlock) {
                    metaBlock.style.display = "block";
                    document.getElementById("metaModel").textContent = completedJob.model || '--';
                    document.getElementById("metaBackend").textContent = completedJob.backend || '--';
                    document.getElementById("metaDevice").textContent = completedJob.device || '--';
                    document.getElementById("metaTime").textContent = `${(completedJob.duration_s || 0).toFixed(2)}s`;
                    document.getElementById("metaRes").textContent = `${completedJob.width}x${completedJob.height}`;
                    document.getElementById("metaSteps").textContent = completedJob.steps || '--';
                }
            } else {
                if (placeholder) placeholder.style.display = "block";
                if (previewImg) previewImg.style.display = "none";
                if (metaBlock) metaBlock.style.display = "none";
            }
        }
    }
}

/**
 * Matrix Rain Ambient Visual Effect
 */
class MatrixRain {
    constructor() {
        this.canvas = document.getElementById("matrixRainCanvas");
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext("2d");
        this.characters = "0123456789ABCDEFｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ9876543210<>{}[]=/*+~";
        this.fontSize = 14;
        this.columns = 0;
        this.drops = [];
        this.resize();
        window.addEventListener("resize", () => this.resize());
        this.render = this.render.bind(this);
        requestAnimationFrame(this.render);
    }

    resize() {
        if (!this.canvas) return;
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.columns = Math.floor(this.canvas.width / this.fontSize);
        this.drops = [];
        for (let i = 0; i < this.columns; i++) {
            this.drops[i] = Math.floor(Math.random() * -50);
        }
    }

    render() {
        if (!this.ctx || !this.canvas) return;
        // Translucent fade over canvas to produce trails
        this.ctx.fillStyle = "rgba(2, 5, 3, 0.08)";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = "#00ff66";
        this.ctx.font = `${this.fontSize}px 'JetBrains Mono', monospace`;

        for (let i = 0; i < this.drops.length; i++) {
            const char = this.characters.charAt(Math.floor(Math.random() * this.characters.length));
            const x = i * this.fontSize;
            const y = this.drops[i] * this.fontSize;

            // Highlight leading glyph in white/bright green
            if (Math.random() > 0.88) {
                this.ctx.fillStyle = "#ffffff";
            } else {
                this.ctx.fillStyle = "#00ff66";
            }

            this.ctx.fillText(char, x, y);

            if (y > this.canvas.height && Math.random() > 0.975) {
                this.drops[i] = 0;
            }
            this.drops[i]++;
        }

        setTimeout(() => {
            requestAnimationFrame(this.render);
        }, 33); // ~30 FPS for gentle background ambient
    }
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    window.matrixRain = new MatrixRain();
    window.robotUI = new RobotUI();
});
