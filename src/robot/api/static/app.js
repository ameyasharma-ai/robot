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
        this.personaName = document.getElementById("chatPersonaLabel") || document.getElementById("personaName");
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
        this.chatRecording = false;

        // Cognitive Reasoning & Thinking Mode State
        const savedThinking = localStorage.getItem("robot_thinking_enabled");
        this.thinkingEnabled = savedThinking !== null ? (savedThinking === "true") : true;

        this.initViewControls();
        this.initPersonalitySelector();
        this.initLlmProviderToggle();
        this.initLocalLlmModelControls();
        this.initSttModelControls();
        this.initTtsModelControls();
        this.init3DCharacter();
        this.initTextChat();
        this.initChatWorkspace();
        this.initBlinking();
        this.initWebsocket();
        this.initImageGen();
        this.initAudioDeck();
        this.initMemoryDeck();
        this.initPipelineTelemetryDeck();
        this.startPollers();
        this.fetchThinkingState();
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

        // Wire Avatar View Thinking Toggle and Holographic Matrix Collapse Header
        const avatarThinkingBtn = document.getElementById("btnAvatarThinkingToggle");
        if (avatarThinkingBtn) {
            avatarThinkingBtn.addEventListener("click", (e) => {
                e.preventDefault();
                this.toggleThinkingMode();
            });
        }
        const avatarThoughtHdr = document.getElementById("avatarThoughtHeader");
        if (avatarThoughtHdr) {
            avatarThoughtHdr.addEventListener("click", () => {
                const card = document.getElementById("avatarThoughtCard");
                if (card) card.classList.toggle("collapsed");
            });
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
                if (this.thinkingEnabled) {
                    this.showAvatarThoughtLive("started");
                }
            } catch (err) {
                console.error("UI update failed before fetch:", err);
            }

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 20000);

            try {
                const resp = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ 
                        text,
                        thinking: this.thinkingEnabled
                    }),
                    signal: controller.signal,
                });
                clearTimeout(timeoutId);

                if (!resp.ok) {
                    throw new Error(`Server returned HTTP ${resp.status}`);
                }

                const data = await resp.json();
                const rawResponse = data.response || data.error || "No response received.";
                const response = this.cleanHumanText(rawResponse) || rawResponse;

                if (this.thinkingEnabled) {
                    this.showAvatarThoughtLive("completed", { duration_ms: 1200 });
                }

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
        this.chatSidebarBackdrop = document.getElementById("chatSidebarBackdrop");
        this.btnToggleChatSidebar = document.getElementById("btnToggleChatSidebar");
        this.btnNewChat = document.getElementById("btnNewChat");
        this.btnRefreshSessions = document.getElementById("btnRefreshSessions");
        this.chatSessionsList = document.getElementById("chatSessionsList");
        this.chatActiveTitle = document.getElementById("chatActiveTitle");
        this.chatSessionIdPill = document.getElementById("chatSessionIdPill");
        this.chatPersonaLabel = document.getElementById("chatPersonaLabel");
        this.chatActiveModelPill = document.getElementById("chatActiveModelPill");
        this.chatModelLabel = document.getElementById("chatModelLabel");
        this.chatMicPill = document.getElementById("chatMicPill");
        this.chatMicLabel = document.getElementById("chatMicLabel");
        this.chatMessagesContainer = document.getElementById("chatMessagesContainer");
        this.chatBoxWrapper = document.getElementById("chatBoxWrapper");
        this.chatRecordDock = document.getElementById("chatRecordDock");
        this.btnChatMicRecord = document.getElementById("btnChatMicRecord");
        this.btnChatRecordCancel = document.getElementById("btnChatRecordCancel");
        this.btnChatRecordDone = document.getElementById("btnChatRecordDone");
        this.chatRecordTimer = document.getElementById("chatRecordTimer");
        this.chatRecordStatus = document.getElementById("chatRecordStatus");
        this.chatRecordWaveform = document.getElementById("chatRecordWaveform");
        this.chatTextarea = document.getElementById("chatTextarea");
        this.btnChatSubmit = document.getElementById("btnChatSubmit");
        this.btnChatStopStream = document.getElementById("btnChatStopStream");
        this.chatLiveSpeed = document.getElementById("chatLiveSpeed");

        this.btnChatThinkingToggle = document.getElementById("btnChatThinkingToggle");
        if (this.btnChatThinkingToggle) {
            this.btnChatThinkingToggle.addEventListener("click", (e) => {
                e.preventDefault();
                this.toggleThinkingMode();
            });
        }

        if (this.btnChatMicRecord) {
            this.btnChatMicRecord.addEventListener("click", (e) => {
                e.preventDefault();
                if (this.chatRecording) {
                    this.finishChatVoiceRecord();
                } else {
                    this.startChatVoiceRecord();
                }
            });
        }

        if (this.btnChatRecordCancel) {
            this.btnChatRecordCancel.addEventListener("click", (e) => {
                e.preventDefault();
                this.cancelChatVoiceRecord();
            });
        }

        if (this.btnChatRecordDone) {
            this.btnChatRecordDone.addEventListener("click", (e) => {
                e.preventDefault();
                this.finishChatVoiceRecord();
            });
        }

        // Global hotkeys while recording: Enter = Finish/Transcribe, Escape = Cancel
        window.addEventListener("keydown", (e) => {
            if (!this.chatRecording) return;
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                this.finishChatVoiceRecord();
            } else if (e.key === "Escape") {
                e.preventDefault();
                this.cancelChatVoiceRecord();
            }
        });

        if (this.btnToggleChatSidebar && this.chatSidebar) {
            this.btnToggleChatSidebar.addEventListener("click", () => {
                if (window.innerWidth <= 768) {
                    const isOpen = this.chatSidebar.classList.toggle("mobile-open");
                    if (this.chatSidebarBackdrop) {
                        this.chatSidebarBackdrop.classList.toggle("active", isOpen);
                    }
                } else {
                    this.chatSidebar.classList.toggle("collapsed");
                }
            });
        }

        if (this.chatSidebarBackdrop && this.chatSidebar) {
            this.chatSidebarBackdrop.addEventListener("click", () => {
                this.chatSidebar.classList.remove("mobile-open");
                this.chatSidebarBackdrop.classList.remove("active");
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

        // Category Tab Filter for Chat System Tools
        const catTabs = document.querySelectorAll(".cat-tab");
        const chatChips = document.querySelectorAll(".chat-chip");
        const countBadge = document.getElementById("chatToolsCount");

        catTabs.forEach((tab) => {
            tab.addEventListener("click", () => {
                catTabs.forEach((t) => t.classList.remove("active"));
                tab.classList.add("active");
                const chosenCat = tab.getAttribute("data-cat") || "quick";
                let visibleCount = 0;

                chatChips.forEach((chip) => {
                    const rawCats = chip.getAttribute("data-cat") || "";
                    const catList = rawCats.split(/\s+/);
                    if (chosenCat === "all" || catList.includes(chosenCat)) {
                        chip.classList.remove("hidden-by-cat");
                        visibleCount++;
                    } else {
                        chip.classList.add("hidden-by-cat");
                    }
                });

                if (countBadge) {
                    countBadge.textContent = `${visibleCount} ACTIVE`;
                }
            });
        });

        // Trigger initial static filtering (defaults to Quick tools with zero scroll)
        const initialActiveTab = document.querySelector(".cat-tab.active") || catTabs[0];
        if (initialActiveTab) {
            initialActiveTab.click();
        }

        // Quick available tools chips — insert prefix into chat input without auto-submitting
        chatChips.forEach((chip) => {
            chip.addEventListener("click", () => {
                const prefix = chip.getAttribute("data-prefix") || chip.getAttribute("data-prompt") || "";
                if (this.chatTextarea) {
                    const currentVal = this.chatTextarea.value.trim();
                    if (!currentVal) {
                        this.chatTextarea.value = prefix;
                    } else if (!currentVal.startsWith(prefix)) {
                        this.chatTextarea.value = `${prefix}${currentVal}`;
                    }
                    this.chatTextarea.style.height = "auto";
                    this.chatTextarea.style.height = `${Math.min(220, this.chatTextarea.scrollHeight)}px`;
                    this.chatTextarea.focus();
                    const len = this.chatTextarea.value.length;
                    this.chatTextarea.setSelectionRange(len, len);
                }
            });
        });

        this.syncChatTopBar();
    }

    /**
     * Format a personality key to a clean, canonical human-readable display name.
     * @param {string} [personaKey] - Persona identifier ('neo', 'luna', 'aria', 'glados').
     * @returns {string} Formatted display name ('CYPHER', 'LUNA', 'ARIA', 'GLADOS', or 'ROBOT').
     */
    formatPersonaName(personaKey) {
        if (!personaKey) return "ROBOT";
        const clean = personaKey.trim().toLowerCase();
        if (clean === "neo") return "CYPHER";
        return clean.toUpperCase();
    }

    /**
     * Resolve the formatted display title of the currently loaded character/persona.
     * Checks dropdown selector, chat persona label, or active persona property.
     * @returns {string} Formatted title ('Cypher', 'Luna', 'Aria', 'GLaDOS', or 'Robot').
     */
    getActivePersonaTitle() {
        if (this.personaSelect && this.personaSelect.value) {
            const val = this.personaSelect.value.toLowerCase().trim();
            if (val === "neo" || val === "cypher") return "Cypher";
            if (val === "glados") return "GLaDOS";
            return val.charAt(0).toUpperCase() + val.slice(1);
        }
        if (this.chatPersonaLabel && this.chatPersonaLabel.textContent) {
            const txt = this.chatPersonaLabel.textContent.trim().toLowerCase();
            if (txt === "neo" || txt === "cypher") return "Cypher";
            if (txt === "glados") return "GLaDOS";
            return txt.charAt(0).toUpperCase() + txt.slice(1);
        }
        if (this.activePersona) {
            const p = this.activePersona.toLowerCase().trim();
            if (p === "neo" || p === "cypher") return "Cypher";
            if (p === "glados") return "GLaDOS";
            return p.charAt(0).toUpperCase() + p.slice(1);
        }
        return "Robot";
    }

    /**
     * Synchronize the chat interface top bar indicators (persona label, theme, active model pill, STT and TTS pickers).
     * @returns {void}
     */
    syncChatTopBar() {
        if (this.chatPersonaLabel && this.personaSelect) {
            const p = this.personaSelect.value;
            this.chatPersonaLabel.textContent = this.formatPersonaName(p);
            this.applyPersonaTheme(p);
        }
        const modelPill = document.getElementById("chatActiveModelPill") || this.chatActiveModelPill;
        const modelLabel = document.getElementById("chatModelLabel") || this.chatModelLabel;
        const modelIcon = modelPill ? modelPill.querySelector(".model-icon") : null;
        const chatOpenaiControls = document.getElementById("chatOpenaiControls");
        const chatOpenaiSelect = document.getElementById("chatOpenaiModelSelect");

        const chatWhisperLabel = document.getElementById("chatWhisperPickerLabel");
        if (chatWhisperLabel && this.activeSttModelId) {
            chatWhisperLabel.textContent = `Whisper: ${this.activeSttModelId}`;
        }
        const chatTtsLabel = document.getElementById("chatTtsPickerLabel");
        if (chatTtsLabel) {
            const activeVoiceClean = this.activeTtsVoice ? this.activeTtsVoice.replace(".onnx", "").replace("en-us-", "").replace("models/tts/", "") : "af_heart";
            chatTtsLabel.textContent = `TTS: ${activeVoiceClean}`;
        }

        const prov = this.currentLlmProvider || "local";
        const isGemini = prov === "gemini";
        const isOpenAi = prov === "openai_compatible";

        if (modelLabel) {
            modelLabel.textContent = this.formatModelBadgeText(prov);
        }
        if (modelIcon) {
            modelIcon.textContent = isOpenAi ? "🌐" : (isGemini ? "✨" : "⚡");
        }
        if (chatOpenaiControls) {
            chatOpenaiControls.style.display = isOpenAi ? "inline-flex" : "none";
        }
        if (chatOpenaiSelect && this.activeOpenaiModel) {
            chatOpenaiSelect.value = this.activeOpenaiModel;
        }

        if (modelPill) {
            modelPill.classList.toggle("pill-gemini", isGemini);
            modelPill.classList.toggle("pill-openai", isOpenAi);
            modelPill.classList.toggle("pill-local", !isGemini && !isOpenAi);
            const ctxK = Math.round((this.localContextWindow || 16384) / 1024);
            if (isOpenAi) {
                const activeM = this.activeOpenaiModel || "gpt-4o";
                modelPill.title = `Active AI Model: Custom OpenAI-Compatible (${activeM}, Prompt Caching Active). Click to cycle provider.`;
            } else if (isGemini) {
                modelPill.title = "Active AI Model: Cloud Gemini 3.5 Flash Lite (1M Context, High Intelligence). Click to cycle provider.";
            } else {
                modelPill.title = `Active AI Model: Local Qwen 3.5 4B on RTX 3050 (Low Latency, ${ctxK}K Context). Click to cycle provider.`;
            }
        }
    }

    /**
     * Display an Apple-grade Dynamic Island notification toast pill.
     * Interacts with DOM element `#dynamicIsland` using CSS aero spring transitions.
     *
     * @param {string} title - Primary headline text to display.
     * @param {string} subtitle - Secondary descriptive text or status detail.
     * @param {string} [icon='⚡'] - Visual emoji or symbolic badge.
     * @param {string} [tone='emerald'] - Color theme accent ('emerald', 'cyan', 'amber', 'rose').
     * @param {number} [durationMs=3200] - Duration in milliseconds before smooth dismissal.
     * @returns {void}
     */
    showDynamicIsland(title, subtitle, icon = '⚡', tone = 'emerald', durationMs = 3200) {
        const island = document.getElementById("dynamicIsland");
        if (!island) return;

        const iconEl = document.getElementById("islandIcon");
        const titleEl = document.getElementById("islandTitle");
        const subEl = document.getElementById("islandSubtitle");
        const dotEl = document.getElementById("islandStatusDot");

        if (iconEl) iconEl.textContent = icon;
        if (titleEl) titleEl.textContent = title;
        if (subEl) subEl.textContent = subtitle;

        if (dotEl) {
            dotEl.className = "island-status-dot";
            if (tone === "cyan") {
                dotEl.style.background = "#06b6d4";
            } else if (tone === "amber") {
                dotEl.style.background = "#f59e0b";
            } else if (tone === "rose") {
                dotEl.style.background = "#ef4444";
            } else {
                dotEl.style.background = "#10b981";
            }
            dotEl.style.boxShadow = `0 0 10px ${dotEl.style.background}`;
        }

        // Apply theme tone glow class to Dynamic Island container
        island.classList.remove("tone-emerald", "tone-cyan", "tone-amber", "tone-rose");
        island.classList.add(`tone-${tone}`);

        island.classList.add("visible");

        if (this._islandTimeout) {
            clearTimeout(this._islandTimeout);
        }

        this._islandTimeout = setTimeout(() => {
            island.classList.remove("visible");
            this._islandTimeout = null;
        }, durationMs);
    }

    /**
     * Fetch the engine's current thinking/reasoning state from the server.
     * Synchronizes local state, localStorage, and both UI buttons.
     * @returns {Promise<void>}
     */
    async fetchThinkingState() {
        try {
            const res = await fetch("/api/thinking");
            if (res.ok) {
                const data = await res.json();
                this.thinkingEnabled = Boolean(data.thinking_enabled);
                localStorage.setItem("robot_thinking_enabled", String(this.thinkingEnabled));
                this.updateThinkingUI(this.thinkingEnabled);
            }
        } catch (e) {
            console.warn("Failed to fetch thinking state from API:", e);
        }
    }

    /**
     * Toggle AI deep cognitive reasoning mode on or off.
     * Synchronizes across Chat top bar, Avatar HUD, localStorage, and engine runtime.
     * @param {boolean} [forcedState] - Optional explicit boolean state.
     * @returns {Promise<void>}
     */
    async toggleThinkingMode(forcedState) {
        const target = typeof forcedState === "boolean" ? forcedState : !this.thinkingEnabled;
        this.thinkingEnabled = target;
        localStorage.setItem("robot_thinking_enabled", String(target));
        this.updateThinkingUI(target);

        try {
            const res = await fetch("/api/thinking", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled: target }),
            });
            if (res.ok) {
                const data = await res.json();
                this.thinkingEnabled = Boolean(data.thinking_enabled);
                this.updateThinkingUI(this.thinkingEnabled);
                if (typeof this.logEvent === "function") {
                    this.logEvent("THINKING", `Cognitive thinking & reasoning mode ${this.thinkingEnabled ? "ENABLED" : "DISABLED"}`, "system");
                }
            }
        } catch (err) {
            console.error("Failed to toggle thinking mode:", err);
        }
    }

    /**
     * Update the visual states of thinking buttons across Chat and Avatar views.
     * @param {boolean} enabled - Whether thinking mode is active.
     * @returns {void}
     */
    updateThinkingUI(enabled) {
        // Chat workspace header toggle button
        const chatToggle = document.getElementById("btnChatThinkingToggle") || this.btnChatThinkingToggle;
        const chatLabel = document.getElementById("chatThinkingLabel");
        if (chatToggle) {
            if (enabled) {
                chatToggle.classList.add("active");
                chatToggle.setAttribute("aria-pressed", "true");
                chatToggle.title = "AI Deep Reasoning is ON. Click to disable thinking mode.";
                if (chatLabel) chatLabel.textContent = "THINKING ON";
            } else {
                chatToggle.classList.remove("active");
                chatToggle.setAttribute("aria-pressed", "false");
                chatToggle.title = "AI Deep Reasoning is OFF. Click to enable thinking mode.";
                if (chatLabel) chatLabel.textContent = "THINKING OFF";
            }
        }

        // Avatar input bar toggle button
        const avatarToggle = document.getElementById("btnAvatarThinkingToggle") || this.btnAvatarThinkingToggle;
        const avatarLabel = document.getElementById("avatarThinkingLabel");
        if (avatarToggle) {
            if (enabled) {
                avatarToggle.classList.add("active");
                avatarToggle.setAttribute("aria-pressed", "true");
                avatarToggle.title = "AI Deep Reasoning is ON. Click to disable thinking mode.";
                if (avatarLabel) avatarLabel.textContent = "THINKING ON";
            } else {
                avatarToggle.classList.remove("active");
                avatarToggle.setAttribute("aria-pressed", "false");
                avatarToggle.title = "AI Deep Reasoning is OFF. Click to enable thinking mode.";
                if (avatarLabel) avatarLabel.textContent = "THINKING OFF";
            }
        }
    }

    /**
     * Render live holographic neural reasoning matrix in the 3D Avatar viewport.
     * Handles lifecycle phases: 'started', 'token', 'completed', and 'hide'.
     * @param {"started" | "token" | "completed" | "hide"} action - Lifecycle event action.
     * @param {any} [data] - Associated event payload (token string, completion object).
     * @returns {void}
     */
    showAvatarThoughtLive(action, data) {
        const card = document.getElementById("avatarThoughtCard");
        const body = document.getElementById("avatarThoughtBody");
        const timer = document.getElementById("avatarThoughtTimer");
        const title = document.getElementById("avatarThoughtTitle");
        const badge = document.getElementById("charThinkingBadge");

        if (action === "started") {
            if (badge) {
                badge.style.display = "inline-flex";
                badge.classList.add("active");
            }
            if (card && body) {
                card.style.display = "block";
                card.classList.remove("collapsed");
                body.textContent = "Contemplating context and reasoning strategy...";
                body.classList.add("thinking-active");
                if (timer) timer.textContent = "0.0s";
                if (title) title.textContent = "NEURAL REASONING MATRIX";
                this._avatarThoughtStartTime = performance.now();
                if (this._avatarThoughtTimerInterval) clearInterval(this._avatarThoughtTimerInterval);
                this._avatarThoughtTimerInterval = setInterval(() => {
                    if (timer && this._avatarThoughtStartTime) {
                        const elapsed = (performance.now() - this._avatarThoughtStartTime) / 1000;
                        timer.textContent = `${elapsed.toFixed(1)}s`;
                    }
                }, 100);
            }
        } else if (action === "token") {
            if (card && card.style.display === "none") {
                this.showAvatarThoughtLive("started");
            }
            if (body && data) {
                if (body.classList.contains("thinking-active")) {
                    body.textContent = "";
                    body.classList.remove("thinking-active");
                }
                body.textContent += data;
                body.scrollTop = body.scrollHeight;
            }
        } else if (action === "completed") {
            if (this._avatarThoughtTimerInterval) {
                clearInterval(this._avatarThoughtTimerInterval);
                this._avatarThoughtTimerInterval = null;
            }
            if (data && data.text && data.text.trim()) {
                if (body) {
                    body.textContent = data.text;
                    body.classList.remove("thinking-active");
                }
                if (title) {
                    title.textContent = "COGNITIVE REASONING";
                }
            } else if (body && body.classList.contains("thinking-active")) {
                if (card) card.style.display = "none";
            }
            if (timer && data && data.duration_ms) {
                timer.textContent = `${(data.duration_ms / 1000).toFixed(1)}s`;
            }
            if (title && (!data || !data.text)) {
                title.textContent = "REASONING CONCLUDED";
            }
            if (badge) {
                setTimeout(() => {
                    if (badge && !this._avatarThoughtTimerInterval) {
                        badge.style.display = "none";
                    }
                }, 2500);
            }
            // Auto-collapse the holographic thought matrix after 10s of quiet idle
            if (this._avatarThoughtHideTimeout) clearTimeout(this._avatarThoughtHideTimeout);
            this._avatarThoughtHideTimeout = setTimeout(() => {
                if (card && !this._avatarThoughtTimerInterval) {
                    card.classList.add("collapsed");
                }
            }, 10000);
        } else if (action === "hide") {
            if (this._avatarThoughtTimerInterval) {
                clearInterval(this._avatarThoughtTimerInterval);
                this._avatarThoughtTimerInterval = null;
            }
            if (card) card.style.display = "none";
            if (badge) badge.style.display = "none";
        }
    }

    /**
     * Return formatted model badge text depending on provider and local context size.
     * @param {string} [provider] - Active LLM provider ('local', 'gemini', or 'openai_compatible').
     * @param {string} [modelName] - Optional model identifier override.
     * @returns {string} Human-readable badge text.
     */
    formatModelBadgeText(provider, modelName) {
        let prov = provider;
        if (typeof prov !== "string") {
            prov = prov ? "gemini" : (this.currentLlmProvider || "local");
        }
        if (prov === "openai_compatible" || prov === "openai") {
            const m = modelName || this.activeOpenaiModel || "gpt-4o";
            return `OPENAI (${m.toUpperCase()})`;
        }
        if (prov === "gemini") {
            const m = modelName || "1M";
            return `GEMINI (${m.toUpperCase()})`;
        }
        const ctxK = Math.round((this.localContextWindow || 16384) / 1024);
        return `LOCAL QWEN (${ctxK}K)`;
    }

    /**
     * Get badge configuration (text, CSS class, icon) for a chat message.
     * @param {string} [provider] - Active provider identifier.
     * @param {string} [modelName] - Active model identifier.
     * @returns {{ text: string, className: string, icon: string }}
     */
    getChatMsgBadgeInfo(provider, modelName) {
        let prov = provider;
        if (typeof prov !== "string") {
            prov = prov ? "gemini" : (this.currentLlmProvider || "local");
        }
        if (prov === "openai_compatible" || prov === "openai") {
            const m = modelName || this.activeOpenaiModel || "gpt-4o";
            return {
                text: `OPENAI (${m.toUpperCase()})`,
                className: "badge-openai",
                icon: "🌐 "
            };
        }
        if (prov === "gemini") {
            return {
                text: "GEMINI (1M)",
                className: "badge-gemini",
                icon: "✨ "
            };
        }
        const ctxK = Math.round((this.localContextWindow || 16384) / 1024);
        return {
            text: `LOCAL QWEN (${ctxK}K)`,
            className: "badge-local",
            icon: "⚡ "
        };
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

        try {
            await fetch("/api/view-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: "chat" }),
            });
        } catch (e) {}

        this.syncChatTopBar();

        if (!this.chatSessions || this.chatSessions.length === 0) {
            await this.loadChatSessions();
        }

        if (this.chatTextarea) {
            this.chatTextarea.focus();
        }
    }

    async onExitChatMode() {
        this.stopSpeakingMessage();
        try {
            const res = await fetch("/voice-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ enabled: true }),
            });
            if (res.ok) {
                const data = await res.json();
                this.voiceEnabled = data.voice_enabled !== undefined ? data.voice_enabled : true;
                if (this.updateVoiceUI) this.updateVoiceUI(this.voiceEnabled);
            } else {
                this.voiceEnabled = true;
                if (this.updateVoiceUI) this.updateVoiceUI(true);
            }
            this.logEvent("AUDIO", "Returned to Voice Interface: Microphone restored.", "system");
        } catch (e) {
            console.error("Failed to restore voice mode:", e);
        }

        try {
            await fetch("/api/view-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: "dual" }),
            });
        } catch (e) {}
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
        this.stopSpeakingMessage();
        this.activeChatSessionId = sessionId;
        this.renderSessionsList();

        if (window.innerWidth <= 768 && this.chatSidebar) {
            this.chatSidebar.classList.remove("mobile-open");
            if (this.chatSidebarBackdrop) this.chatSidebarBackdrop.classList.remove("active");
        }

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

            if (data.personality && this.personaSelect && this.personaSelect.value !== data.personality) {
                this.personaSelect.value = data.personality;
                this.applyPersonaTheme(data.personality);
                localStorage.setItem("robot_active_personality", data.personality);
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
            this.syncChatTopBar();
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

        // Ensure message's own personality is respected so Luna/Neo/Cypher/etc. are accurately displayed
        const activePersona = msg.personality
            ? msg.personality
            : ((this.personaSelect && this.personaSelect.value) ? this.personaSelect.value : "neo");
        const senderLabel = isUser ? "YOU" : this.formatPersonaName(activePersona);
        const avatarIcon = isUser ? "👤" : "🤖";
        const contentHtml = this.renderChatMarkdown(msg.content);
        let badgeHtml = "";
        let telemHtml = "";
        let thoughtHtml = "";

        if (!isUser) {
            if (msg.metadata && msg.metadata.thought) {
                const rawThought = msg.metadata.thought;
                const durMs = msg.metadata.thought_duration_ms || msg.metadata.duration_ms || 0;
                const durStr = durMs ? `${(durMs / 1000).toFixed(1)}s` : "";
                thoughtHtml = `
                    <div class="chat-thought-card collapsed">
                        <div class="chat-thought-header">
                            <div class="chat-thought-left">
                                <span class="thought-brain-icon">🧠</span>
                                <span class="thought-title-text">Thought Process</span>
                                ${durStr ? `<span class="thought-timer-badge">${durStr}</span>` : ""}
                            </div>
                            <button class="btn-thought-toggle" type="button" title="Expand thoughts">▾</button>
                        </div>
                        <div class="chat-thought-body">${this.escapeHtml(rawThought)}</div>
                    </div>
                `;
            }

            const msgProv = (msg.metadata && msg.metadata.provider) ? msg.metadata.provider : this.currentLlmProvider;
            const msgModel = (msg.metadata && msg.metadata.model) ? msg.metadata.model : (msgProv === "openai_compatible" ? this.activeOpenaiModel : null);
            const badgeInfo = this.getChatMsgBadgeInfo(msgProv, msgModel);
            badgeHtml = `<span class="chat-msg-badge ${badgeInfo.className}">${badgeInfo.icon}${badgeInfo.text}</span>`;

            if (msg.metadata && (msg.metadata.tokens || msg.metadata.tok_per_sec)) {
                const durSec = msg.metadata.duration_ms ? (msg.metadata.duration_ms / 1000) : 0;
                telemHtml = `
                    <div class="chat-msg-telemetry" style="display: flex;">
                        <span class="telem-item">⚡ ${msg.metadata.tokens || 0} tokens</span>
                        <span class="telem-item">🚀 ${(Number(msg.metadata.tok_per_sec) || 0).toFixed(1)} tok/s</span>
                        <span class="telem-item">⏱️ ${durSec.toFixed(2)}s</span>
                        <span class="telem-item">🎯 TTFT: ${(Number(msg.metadata.ttft_ms) || 0).toFixed(0)} ms</span>
                    </div>
                `;
            }
        }

        let actionsHtml = "";
        if (!isUser && msg.content) {
            actionsHtml = `
                <div class="chat-msg-actions">
                    <button class="chat-action-btn btn-msg-speak" type="button" title="Read aloud (Speak response)">
                        <span class="action-icon">🔊</span>
                        <span class="action-label">Speak</span>
                    </button>
                    <button class="chat-action-btn btn-msg-copy" type="button" title="Copy response text">
                        <span class="action-icon">📋</span>
                        <span class="action-label">Copy</span>
                    </button>
                </div>
            `;
        }

        row.innerHTML = `
            <div class="chat-msg-avatar">${avatarIcon}</div>
            <div class="chat-msg-content">
                <div class="chat-msg-header">
                    <span class="chat-msg-sender">${senderLabel}</span>
                    ${badgeHtml}
                    <span class="chat-msg-time">${timeStr}</span>
                </div>
                ${thoughtHtml}
                ${toolHtml}
                <div class="chat-msg-body">${contentHtml}</div>
                ${actionsHtml}
                ${telemHtml}
            </div>
        `;
        this.chatMessagesContainer.appendChild(row);

        const thoughtHdr = row.querySelector(".chat-thought-header");
        if (thoughtHdr) {
            thoughtHdr.addEventListener("click", () => {
                const card = row.querySelector(".chat-thought-card");
                if (card) card.classList.toggle("collapsed");
            });
        }

        if (!isUser && msg.content) {
            const speakBtn = row.querySelector(".btn-msg-speak");
            if (speakBtn) {
                speakBtn.addEventListener("click", () => {
                    this.toggleSpeakMessage(speakBtn, msg.content, msg.personality);
                });
            }
            const copyBtn = row.querySelector(".btn-msg-copy");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    this.copyMessageText(copyBtn, msg.content);
                });
            }
        }
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
                <div class="chat-msg-body">${this.renderChatMarkdown(text)}</div>
            </div>
        `;
        this.chatMessagesContainer.appendChild(userRow);
        this.scrollChatToBottom();

        // Create Assistant Streaming row
        const streamId = `stream_${Date.now()}`;
        const activePersonaKey = this.personaSelect ? this.personaSelect.value : "neo";
        const currentPersona = this.formatPersonaName(activePersonaKey);
        const badgeInfo = this.getChatMsgBadgeInfo(this.currentLlmProvider, this.activeOpenaiModel);

        const assistantRow = document.createElement("div");
        assistantRow.className = "chat-msg-row msg-assistant";
        assistantRow.id = `row_${streamId}`;
        assistantRow.innerHTML = `
            <div class="chat-msg-avatar">🤖</div>
            <div class="chat-msg-content">
                <div class="chat-msg-header">
                    <span class="chat-msg-sender">${currentPersona}</span>
                    <span class="chat-msg-badge ${badgeInfo.className}">${badgeInfo.icon}${badgeInfo.text}</span>
                    <span class="chat-msg-time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
                <div class="chat-thought-container" id="thoughtBox_${streamId}"></div>
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
        const thoughtBox = document.getElementById(`thoughtBox_${streamId}`);
        const toolBox = document.getElementById(`toolBox_${streamId}`);
        const telemDiv = document.getElementById(`streamTelem_${streamId}`);

        this.chatAbortController = new AbortController();
        let streamedRawText = "";
        let streamedThoughtText = "";
        let tokenCount = 0;
        let streamStartTime = performance.now();
        let thoughtStartTime = null;
        let thoughtIntervalId = null;
        let isStreamCompleted = false;

        if (this.thinkingEnabled) {
            thoughtStartTime = performance.now();
            if (thoughtBox) {
                const thoughtCard = document.createElement("div");
                thoughtCard.className = "chat-thought-card";
                thoughtCard.id = `thoughtCard_${streamId}`;
                thoughtCard.innerHTML = `
                    <div class="chat-thought-header">
                        <div class="chat-thought-left">
                            <span class="thought-brain-icon thinking">🧠</span>
                            <span class="thought-title-text">Thinking...</span>
                            <span class="thought-timer-badge" id="thoughtTimer_${streamId}">0.0s</span>
                        </div>
                        <button class="btn-thought-toggle" type="button" title="Collapse / Expand thoughts">▾</button>
                    </div>
                    <div class="chat-thought-body thinking-active" id="thoughtText_${streamId}">Contemplating context, tools, and response strategy...</div>
                `;
                const hdr = thoughtCard.querySelector(".chat-thought-header");
                if (hdr) {
                    hdr.addEventListener("click", () => {
                        thoughtCard.classList.toggle("collapsed");
                    });
                }
                thoughtBox.appendChild(thoughtCard);
            }
            if (thoughtIntervalId) clearInterval(thoughtIntervalId);
            thoughtIntervalId = setInterval(() => {
                const timerEl = document.getElementById(`thoughtTimer_${streamId}`);
                if (timerEl && thoughtStartTime) {
                    const el = (performance.now() - thoughtStartTime) / 1000;
                    timerEl.textContent = `${el.toFixed(1)}s`;
                }
            }, 100);

            this.showAvatarThoughtLive("started");
            this.scrollChatToBottom();
        }

        try {
            const res = await fetch("/chat/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    text: text,
                    session_id: this.activeChatSessionId,
                    personality: this.personaSelect ? this.personaSelect.value : undefined,
                    thinking: this.thinkingEnabled
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
                        if (payload.personality) {
                            const senderEl = assistantRow.querySelector(".chat-msg-sender");
                            if (senderEl) {
                                senderEl.textContent = this.formatPersonaName(payload.personality);
                            }
                        }
                        if (payload.provider) {
                            this.currentLlmProvider = payload.provider;
                            if (payload.model && (payload.provider === "openai_compatible" || payload.provider === "openai")) {
                                this.activeOpenaiModel = payload.model;
                            }
                            const badgeEl = assistantRow.querySelector(".chat-msg-badge");
                            if (badgeEl) {
                                const updatedBadge = this.getChatMsgBadgeInfo(payload.provider, payload.model || this.activeOpenaiModel);
                                badgeEl.className = `chat-msg-badge ${updatedBadge.className}`;
                                badgeEl.textContent = `${updatedBadge.icon}${updatedBadge.text}`;
                            }
                            this.syncChatTopBar();
                        }
                    } else if (eventType === "thought_started") {
                        if (!thoughtStartTime) thoughtStartTime = performance.now();
                        if (thoughtBox && !thoughtBox.querySelector(".chat-thought-card")) {
                            const thoughtCard = document.createElement("div");
                            thoughtCard.className = "chat-thought-card";
                            thoughtCard.id = `thoughtCard_${streamId}`;
                            thoughtCard.innerHTML = `
                                <div class="chat-thought-header">
                                    <div class="chat-thought-left">
                                        <span class="thought-brain-icon thinking">🧠</span>
                                        <span class="thought-title-text">Thinking...</span>
                                        <span class="thought-timer-badge" id="thoughtTimer_${streamId}">0.0s</span>
                                    </div>
                                    <button class="btn-thought-toggle" type="button" title="Collapse / Expand thoughts">▾</button>
                                </div>
                                <div class="chat-thought-body thinking-active" id="thoughtText_${streamId}">Contemplating context, tools, and response strategy...</div>
                            `;
                            const hdr = thoughtCard.querySelector(".chat-thought-header");
                            if (hdr) {
                                hdr.addEventListener("click", () => {
                                    thoughtCard.classList.toggle("collapsed");
                                });
                            }
                            thoughtBox.appendChild(thoughtCard);
                        }
                        if (!thoughtIntervalId) {
                            thoughtIntervalId = setInterval(() => {
                                const timerEl = document.getElementById(`thoughtTimer_${streamId}`);
                                if (timerEl && thoughtStartTime) {
                                    const el = (performance.now() - thoughtStartTime) / 1000;
                                    timerEl.textContent = `${el.toFixed(1)}s`;
                                }
                            }, 100);
                        }

                        this.showAvatarThoughtLive("started");
                        this.scrollChatToBottom();
                    } else if (eventType === "thought") {
                        if (payload.token) {
                            streamedThoughtText += payload.token;
                            let thoughtBody = document.getElementById(`thoughtText_${streamId}`);
                            if (!thoughtBody && thoughtBox) {
                                const thoughtCard = document.createElement("div");
                                thoughtCard.className = "chat-thought-card";
                                thoughtCard.id = `thoughtCard_${streamId}`;
                                thoughtCard.innerHTML = `
                                    <div class="chat-thought-header">
                                        <div class="chat-thought-left">
                                            <span class="thought-brain-icon thinking">🧠</span>
                                            <span class="thought-title-text">Thinking...</span>
                                            <span class="thought-timer-badge" id="thoughtTimer_${streamId}">0.0s</span>
                                        </div>
                                        <button class="btn-thought-toggle" type="button" title="Collapse / Expand thoughts">▾</button>
                                    </div>
                                    <div class="chat-thought-body" id="thoughtText_${streamId}"></div>
                                `;
                                const hdr = thoughtCard.querySelector(".chat-thought-header");
                                if (hdr) {
                                    hdr.addEventListener("click", () => {
                                        thoughtCard.classList.toggle("collapsed");
                                    });
                                }
                                thoughtBox.appendChild(thoughtCard);
                                thoughtBody = document.getElementById(`thoughtText_${streamId}`);
                            }
                            if (thoughtBody) {
                                if (thoughtBody.classList.contains("thinking-active")) {
                                    thoughtBody.classList.remove("thinking-active");
                                    thoughtBody.textContent = "";
                                }
                                thoughtBody.textContent = streamedThoughtText;
                                thoughtBody.scrollTop = thoughtBody.scrollHeight;
                            }
                            this.showAvatarThoughtLive("token", payload.token);
                            this.scrollChatToBottom();
                        }
                    } else if (eventType === "thought_completed") {
                        if (thoughtIntervalId) {
                            clearInterval(thoughtIntervalId);
                            thoughtIntervalId = null;
                        }
                        const finalDurSec = payload.duration_ms ? (payload.duration_ms / 1000).toFixed(1) : ((performance.now() - (thoughtStartTime || streamStartTime)) / 1000).toFixed(1);
                        const timerEl = document.getElementById(`thoughtTimer_${streamId}`);
                        if (timerEl) {
                            timerEl.textContent = `${finalDurSec}s`;
                        }
                        const bIcon = assistantRow.querySelector(".thought-brain-icon");
                        if (bIcon) bIcon.classList.remove("thinking");

                        if (payload.text && (!streamedThoughtText || streamedThoughtText.length < payload.text.length)) {
                            streamedThoughtText = payload.text;
                        }

                        const thoughtCard = document.getElementById(`thoughtCard_${streamId}`);
                        const thoughtBody = document.getElementById(`thoughtText_${streamId}`);

                        if (streamedThoughtText && streamedThoughtText.trim().length > 0) {
                            const words = streamedThoughtText.trim().split(/\s+/).length;
                            const tTitle = assistantRow.querySelector(".thought-title-text");
                            if (tTitle) tTitle.textContent = `Thought for ${finalDurSec}s (${words} words)`;
                            if (thoughtBody) {
                                thoughtBody.classList.remove("thinking-active");
                                thoughtBody.innerHTML = this.renderChatMarkdown(streamedThoughtText);
                            }
                        } else {
                            if (thoughtCard) thoughtCard.remove();
                        }

                        this.showAvatarThoughtLive("completed", {
                            ...payload,
                            duration_ms: parseFloat(finalDurSec) * 1000,
                            text: streamedThoughtText,
                        });
                    } else if (eventType === "token") {
                        if (thoughtIntervalId) {
                            clearInterval(thoughtIntervalId);
                            thoughtIntervalId = null;
                            const finalDurSec = ((performance.now() - (thoughtStartTime || streamStartTime)) / 1000).toFixed(1);
                            const timerEl = document.getElementById(`thoughtTimer_${streamId}`);
                            if (timerEl) timerEl.textContent = `${finalDurSec}s`;
                            const bIcon = assistantRow.querySelector(".thought-brain-icon");
                            if (bIcon) bIcon.classList.remove("thinking");

                            const thoughtCard = document.getElementById(`thoughtCard_${streamId}`);
                            const thoughtBody = document.getElementById(`thoughtText_${streamId}`);

                            if (streamedThoughtText && streamedThoughtText.trim().length > 0) {
                                const words = streamedThoughtText.trim().split(/\s+/).length;
                                const tTitle = assistantRow.querySelector(".thought-title-text");
                                if (tTitle) tTitle.textContent = `Thought for ${finalDurSec}s (${words} words)`;
                                if (thoughtBody) {
                                    thoughtBody.classList.remove("thinking-active");
                                    thoughtBody.innerHTML = this.renderChatMarkdown(streamedThoughtText);
                                }
                                this.showAvatarThoughtLive("completed", { duration_ms: parseFloat(finalDurSec) * 1000, text: streamedThoughtText });
                            } else {
                                if (thoughtCard) thoughtCard.remove();
                                this.showAvatarThoughtLive("hide");
                            }
                        }
                        if (payload.token) {
                            streamedRawText += payload.token;
                            tokenCount++;
                            const elapsedSec = (performance.now() - streamStartTime) / 1000;
                            const speed = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : "0.0";
                            if (this.chatLiveSpeed) this.chatLiveSpeed.textContent = `${speed} tok/s`;
                            if (textSpan) textSpan.innerHTML = this.renderChatMarkdown(streamedRawText);
                            if (this.hudTokens) this.hudTokens.textContent = `${tokenCount}`;
                            if (tokenCount === 1 && this.hudTtft) {
                                const liveTtft = payload.ttft_ms !== undefined ? Math.round(payload.ttft_ms) : Math.round(performance.now() - streamStartTime);
                                this.hudTtft.textContent = `${liveTtft} ms`;
                            }
                            this.scrollChatToBottom();
                        }
                    } else if (eventType === "tool_started") {
                        if (payload.clear_preamble) {
                            streamedRawText = "";
                            if (textSpan) textSpan.innerHTML = "";
                        }
                    } else if (eventType === "tool_call") {
                        if (toolBox) {
                            const callId = payload.call_id || `call_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
                            const card = document.createElement("div");
                            card.className = "chat-tool-card running";
                            card.id = `toolCard_${callId}`;
                            card.dataset.tool = payload.tool || "";
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
                            let existingCard = null;
                            if (payload.call_id) {
                                existingCard = toolBox.querySelector(`#toolCard_${payload.call_id}`);
                            }
                            if (!existingCard && payload.tool) {
                                existingCard = toolBox.querySelector(`.chat-tool-card.running[data-tool="${payload.tool}"]`);
                            }
                            if (!existingCard) {
                                existingCard = toolBox.querySelector(".chat-tool-card.running");
                            }

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
                        if (thoughtIntervalId) {
                            clearInterval(thoughtIntervalId);
                            thoughtIntervalId = null;
                            const finalDurSec = ((performance.now() - (thoughtStartTime || streamStartTime)) / 1000).toFixed(1);
                            const timerEl = document.getElementById(`thoughtTimer_${streamId}`);
                            if (timerEl) timerEl.textContent = `${finalDurSec}s`;
                            const bIcon = assistantRow.querySelector(".thought-brain-icon");
                            if (bIcon) bIcon.classList.remove("thinking");

                            const thoughtCard = document.getElementById(`thoughtCard_${streamId}`);
                            const thoughtBody = document.getElementById(`thoughtText_${streamId}`);

                            if (streamedThoughtText && streamedThoughtText.trim().length > 0) {
                                const words = streamedThoughtText.trim().split(/\s+/).length;
                                const tTitle = assistantRow.querySelector(".thought-title-text");
                                if (tTitle) tTitle.textContent = `Thought for ${finalDurSec}s (${words} words)`;
                                if (thoughtBody) {
                                    thoughtBody.classList.remove("thinking-active");
                                    thoughtBody.innerHTML = this.renderChatMarkdown(streamedThoughtText);
                                }
                                this.showAvatarThoughtLive("completed", { duration_ms: parseFloat(finalDurSec) * 1000, text: streamedThoughtText });
                            } else {
                                if (thoughtCard) thoughtCard.remove();
                                this.showAvatarThoughtLive("hide");
                            }
                        }
                        isStreamCompleted = true;
                        if (payload.response !== undefined) {
                            streamedRawText = payload.response;
                            if (textSpan) textSpan.innerHTML = this.renderChatMarkdown(payload.response);
                        } else if (textSpan && streamedRawText) {
                            textSpan.innerHTML = this.renderChatMarkdown(streamedRawText);
                        }
                        if (cursorSpan) cursorSpan.remove();
                        if (toolBox) {
                            toolBox.querySelectorAll(".chat-tool-card.running").forEach(rc => {
                                rc.classList.remove("running");
                                rc.classList.add("completed");
                                const sp = rc.querySelector(".tool-spinner");
                                if (sp) {
                                    sp.className = "tool-icon";
                                    sp.textContent = "✓";
                                }
                            });
                        }
                        const finalTtft = payload.ttft_ms !== undefined ? payload.ttft_ms : (performance.now() - streamStartTime);
                        const finalDuration = payload.duration_ms !== undefined ? payload.duration_ms : (performance.now() - streamStartTime);
                        const finalTokens = payload.tokens || tokenCount;
                        if (typeof this.updateMetrics === "function") {
                            this.updateMetrics({
                                ttft: finalTtft,
                                e2e: finalDuration,
                                tokens: finalTokens,
                            });
                        }
                        if (telemDiv) {
                            telemDiv.style.display = "flex";
                            const durationSec = finalDuration ? (finalDuration / 1000) : 0;
                            telemDiv.innerHTML = `
                                <span class="telem-item">⚡ ${finalTokens} tokens</span>
                                <span class="telem-item">🚀 ${(payload.tok_per_sec || 0).toFixed(1)} tok/s</span>
                                <span class="telem-item">⏱️ ${durationSec.toFixed(2)}s</span>
                                <span class="telem-item">🎯 TTFT: ${Math.round(finalTtft)} ms</span>
                            `;
                        }
                        if (this.chatLiveSpeed) {
                            this.chatLiveSpeed.style.display = "none";
                        }
                        this.loadChatSessions();
                    } else if (eventType === "error") {
                        if (thoughtIntervalId) {
                            clearInterval(thoughtIntervalId);
                            thoughtIntervalId = null;
                        }
                        if (textSpan) {
                            textSpan.innerHTML += `<div class="chat-stream-error">[Error: ${this.escapeHtml(payload.error || "Unknown stream error")}]</div>`;
                        }
                    }
                }
            }
        } catch (err) {
            if (err.name === "AbortError") {
                if (textSpan) textSpan.innerHTML += ` <span class="chat-stream-stopped">[Generation stopped]</span>`;
            } else if (!isStreamCompleted) {
                if (textSpan) textSpan.innerHTML += ` <span class="chat-stream-error">[Error: ${this.escapeHtml(err.message)}]</span>`;
            }
        } finally {
            if (thoughtIntervalId) {
                clearInterval(thoughtIntervalId);
                thoughtIntervalId = null;
            }
            if (cursorSpan) cursorSpan.remove();
            this.chatStreaming = false;
            this.chatAbortController = null;
            if (this.btnChatSubmit) this.btnChatSubmit.disabled = false;
            if (this.btnChatStopStream) this.btnChatStopStream.style.display = "none";
            if (this.chatLiveSpeed) this.chatLiveSpeed.textContent = "0.0 tok/s";

            // Attach ChatGPT-like Speak and Copy action bar under the generated assistant response
            if (streamedRawText && streamedRawText.trim()) {
                const finalContent = streamedRawText.trim();
                const contentEl = assistantRow.querySelector(".chat-msg-content");
                if (contentEl && !contentEl.querySelector(".chat-msg-actions")) {
                    const actionsDiv = document.createElement("div");
                    actionsDiv.className = "chat-msg-actions";
                    actionsDiv.innerHTML = `
                        <button class="chat-action-btn btn-msg-speak" type="button" title="Read aloud (Speak response)">
                            <span class="action-icon">🔊</span>
                            <span class="action-label">Speak</span>
                        </button>
                        <button class="chat-action-btn btn-msg-copy" type="button" title="Copy response text">
                            <span class="action-icon">📋</span>
                            <span class="action-label">Copy</span>
                        </button>
                    `;
                    const telemEl = contentEl.querySelector(".chat-msg-telemetry");
                    if (telemEl) {
                        contentEl.insertBefore(actionsDiv, telemEl);
                    } else {
                        contentEl.appendChild(actionsDiv);
                    }

                    const speakBtn = actionsDiv.querySelector(".btn-msg-speak");
                    if (speakBtn) {
                        speakBtn.addEventListener("click", () => {
                            this.toggleSpeakMessage(speakBtn, finalContent, currentPersona.toLowerCase());
                        });
                    }
                    const copyBtn = actionsDiv.querySelector(".btn-msg-copy");
                    if (copyBtn) {
                        copyBtn.addEventListener("click", () => {
                            this.copyMessageText(copyBtn, finalContent);
                        });
                    }
                }
            }

            this.scrollChatToBottom();
            if (this.chatTextarea) this.chatTextarea.focus();
        }
    }

    stopChatStream() {
        if (this.chatAbortController) {
            this.chatAbortController.abort();
        }
        this.stopSpeakingMessage();
    }

    /**
     * Synthesize and read aloud a specific chat message response using low-latency streaming TTS.
     * Mirrors the active avatar pipeline: splits text into natural clauses, modulates prosody
     * (pitch, speed, noise scale) per clause based on conversational sentiment, begins playback
     * immediately on phrase 0 (<100ms first audio), and gaplessly queues remaining phrases.
     * Synchronizes 3D avatar facial blendshapes and visor illumination with detected emotions.
     *
     * @param {HTMLElement} btn - The speak action button element clicked.
     * @param {string} rawText - The unescaped text content to synthesize and read aloud.
     * @param {string} [personality] - Optional AI persona name for voice modeling.
     * @returns {Promise<void>}
     */
    async toggleSpeakMessage(btn, rawText, personality) {
        if (!btn || !rawText) return;

        // If this exact button is currently speaking or synthesizing, clicking it stops playback
        if (btn.classList.contains("is-speaking") || (btn.classList.contains("is-loading") && this.speakAbortController)) {
            this.stopSpeakingMessage();
            return;
        }

        // Stop any previously playing speech
        this.stopSpeakingMessage();

        const labelSpan = btn.querySelector(".action-label");
        const iconSpan = btn.querySelector(".action-icon");
        const origLabel = labelSpan ? labelSpan.textContent : "Speak";
        const origIcon = iconSpan ? iconSpan.textContent : "🔊";

        btn.classList.add("is-loading");
        if (labelSpan) labelSpan.textContent = "Synthesizing...";
        if (iconSpan) iconSpan.textContent = "⏳";

        const currentPersona = (personality || (this.personaSelect ? this.personaSelect.value : "neo")).toLowerCase();

        this.speakAbortController = new AbortController();
        this.phraseQueue = [];
        this.isStreamingChatSpeech = true;
        this.isPlayingPhrase = false;
        this.currentSpeakingBtn = btn;
        this._currentSpeakOrig = { label: origLabel, icon: origIcon };

        const playNextPhrase = async () => {
            if (this.phraseQueue.length === 0) {
                if (this.isStreamingChatSpeech) {
                    // Queue temporarily empty while server synthesizes next clause; wait for stream
                    this.isPlayingPhrase = false;
                    return;
                }
                // All clauses finished playing
                this.stopSpeakingMessage();
                return;
            }

            this.isPlayingPhrase = true;
            const item = this.phraseQueue.shift();

            // Transition button state on first audio chunk playback
            if (btn.classList.contains("is-loading")) {
                btn.classList.remove("is-loading");
                btn.classList.add("is-speaking");
                if (labelSpan) labelSpan.textContent = "Stop";
                if (iconSpan) iconSpan.textContent = "⏹️";
            }

            const audio = new Audio(item.audioUrl);
            this.currentAudio = audio;

            // Synchronize 3D avatar emotion, lipsync and mouth bars
            this.startMouthSpeaking();
            this._forEachEngine((eng) => {
                eng.speakStart();
                if (item.emotion) {
                    eng.setEmotion(item.emotion);
                }
            });
            this.updatePipelineStatus({ TTS: "RUNNING", AVATAR: "RUNNING" });

            const onPhraseEnded = () => {
                try {
                    URL.revokeObjectURL(item.audioUrl);
                } catch (_) {}
                if (this.currentAudio === audio) {
                    this.currentAudio = null;
                }
                playNextPhrase();
            };

            audio.onended = onPhraseEnded;
            audio.onerror = onPhraseEnded;

            try {
                await audio.play();
            } catch (playErr) {
                console.warn("[Speak] Audio play interrupted or rejected:", playErr);
                onPhraseEnded();
            }
        };

        try {
            const resp = await fetch("/api/chat/speak/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                signal: this.speakAbortController.signal,
                body: JSON.stringify({
                    text: rawText,
                    personality: currentPersona
                })
            });

            if (!resp.ok) {
                throw new Error(`TTS streaming returned status ${resp.status}`);
            }

            const reader = resp.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n\n");
                buffer = lines.pop() || "";

                for (const chunk of lines) {
                    const trimmed = chunk.trim();
                    if (!trimmed.startsWith("data:")) continue;
                    const jsonStr = trimmed.slice(5).trim();
                    if (!jsonStr) continue;

                    let msg;
                    try {
                        msg = JSON.parse(jsonStr);
                    } catch (_) {
                        continue;
                    }

                    if (msg.event === "phrase" && msg.audio) {
                        // Decode base64 audio to binary Blob URL
                        const binaryStr = atob(msg.audio);
                        const len = binaryStr.length;
                        const bytes = new Uint8Array(len);
                        for (let i = 0; i < len; i++) {
                            bytes[i] = binaryStr.charCodeAt(i);
                        }
                        const blob = new Blob([bytes], { type: "audio/wav" });
                        const audioUrl = URL.createObjectURL(blob);

                        this.phraseQueue.push({
                            audioUrl,
                            emotion: msg.emotion || "neutral",
                            phrase: msg.phrase,
                            index: msg.index
                        });

                        // Start immediate playback on arrival of first clause
                        if (!this.isPlayingPhrase) {
                            playNextPhrase();
                        }
                    } else if (msg.event === "done") {
                        this.isStreamingChatSpeech = false;
                        if (!this.isPlayingPhrase && this.phraseQueue.length === 0) {
                            this.stopSpeakingMessage();
                        }
                    }
                }
            }

            this.isStreamingChatSpeech = false;
            if (!this.isPlayingPhrase && this.phraseQueue.length === 0) {
                this.stopSpeakingMessage();
            }

        } catch (err) {
            if (err.name === "AbortError") {
                // Playback intentionally stopped by user
                return;
            }
            console.warn("[Speak] Streaming TTS failed, attempting Web Speech fallback:", err);
            // Graceful fallback to browser Web Speech API
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
                const cleanSpeechText = rawText.replace(/```[\s\S]*?```/g, "").replace(/`.*?`/g, "").trim() || rawText;
                const utter = new SpeechSynthesisUtterance(cleanSpeechText);
                utter.rate = 1.0;

                btn.classList.remove("is-loading");
                btn.classList.add("is-speaking");
                if (labelSpan) labelSpan.textContent = "Stop";
                if (iconSpan) iconSpan.textContent = "⏹️";

                this.startMouthSpeaking();
                this._forEachEngine((eng) => eng.speakStart());

                const onDone = () => {
                    this.stopSpeakingMessage();
                };

                utter.onend = onDone;
                utter.onerror = onDone;
                this.currentSpeakingBtn = btn;
                window.speechSynthesis.speak(utter);
            } else {
                this.stopSpeakingMessage();
            }
        }
    }

    /**
     * Stop currently active message speech playback, abort active streams, and reset button state.
     * Cleans up phrase audio queues, revokes Blob URLs, and returns avatar expressions to neutral.
     */
    stopSpeakingMessage() {
        if (this.speakAbortController) {
            try {
                this.speakAbortController.abort();
            } catch (_) {}
            this.speakAbortController = null;
        }

        this.isStreamingChatSpeech = false;
        this.isPlayingPhrase = false;

        if (this.phraseQueue && this.phraseQueue.length > 0) {
            this.phraseQueue.forEach((item) => {
                try {
                    if (item.audioUrl) URL.revokeObjectURL(item.audioUrl);
                } catch (_) {}
            });
            this.phraseQueue = [];
        }

        if (this.currentAudio) {
            try {
                this.currentAudio.pause();
                this.currentAudio.currentTime = 0;
            } catch (_) {}
            this.currentAudio = null;
        }

        if (this.currentAudioUrl) {
            try {
                URL.revokeObjectURL(this.currentAudioUrl);
            } catch (_) {}
            this.currentAudioUrl = null;
        }

        if (window.speechSynthesis) {
            try {
                window.speechSynthesis.cancel();
            } catch (_) {}
        }

        this.stopMouthSpeaking();
        this._forEachEngine((eng) => {
            eng.speakStop();
            eng.setEmotion("neutral");
        });
        this.updatePipelineStatus({ TTS: "COMPLETED", AVATAR: "IDLE" });

        if (this.currentSpeakingBtn) {
            this.currentSpeakingBtn.classList.remove("is-speaking", "is-loading");
            const lbl = this.currentSpeakingBtn.querySelector(".action-label");
            const ico = this.currentSpeakingBtn.querySelector(".action-icon");
            const origLabel = this._currentSpeakOrig?.label || "Speak";
            const origIcon = this._currentSpeakOrig?.icon || "🔊";
            if (lbl) lbl.textContent = origLabel;
            if (ico) ico.textContent = origIcon;
            this.currentSpeakingBtn = null;
            this._currentSpeakOrig = null;
        }
    }

    /**
     * Copy text to user clipboard with visual feedback.
     *
     * @param {HTMLElement} btn - The copy button element.
     * @param {string} text - The raw text to copy.
     */
    async copyMessageText(btn, text) {
        if (!btn || !text) return;
        try {
            await navigator.clipboard.writeText(text);
            btn.classList.add("is-copied");
            const lbl = btn.querySelector(".action-label");
            const ico = btn.querySelector(".action-icon");
            const origLabel = lbl ? lbl.textContent : "Copy";
            const origIcon = ico ? ico.textContent : "📋";
            if (lbl) lbl.textContent = "Copied!";
            if (ico) ico.textContent = "✓";
            setTimeout(() => {
                btn.classList.remove("is-copied");
                if (lbl) lbl.textContent = origLabel;
                if (ico) ico.textContent = origIcon;
            }, 1800);
        } catch (err) {
            console.warn("Failed to copy message:", err);
        }
    }

    async startChatVoiceRecord() {
        if (this.chatRecording) return;
        this.chatRecording = true;
        this.chatPreRecordText = this.chatTextarea ? this.chatTextarea.value : "";
        this.chatAudioChunks = [];
        this.chatFinalSpeechTranscript = "";

        // UI state transitions
        if (this.chatBoxWrapper) this.chatBoxWrapper.classList.add("is-recording");
        if (this.btnChatMicRecord) this.btnChatMicRecord.classList.add("recording");
        if (this.chatRecordDock) this.chatRecordDock.style.display = "flex";
        if (this.chatRecordTimer) this.chatRecordTimer.textContent = "0:00";
        if (this.chatRecordStatus) {
            this.chatRecordStatus.textContent = "Listening...";
            this.chatRecordStatus.classList.remove("interim-text");
        }

        // Start elapsed timer
        this.chatRecordStartTime = Date.now();
        if (this.chatRecordTimerInterval) clearInterval(this.chatRecordTimerInterval);
        this.chatRecordTimerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.chatRecordStartTime) / 1000);
            const m = Math.floor(elapsed / 60);
            const s = elapsed % 60;
            if (this.chatRecordTimer) {
                this.chatRecordTimer.textContent = `${m}:${s < 10 ? "0" : ""}${s}`;
            }
        }, 500);

        try {
            // Request browser mic stream
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });
            this.chatMediaStream = stream;

            // Setup Web Audio API analyser & script processor for high fidelity raw PCM
            const AudioCtxClass = window.AudioContext || window.webkitAudioContext;
            this.chatAudioContext = new AudioCtxClass();
            const source = this.chatAudioContext.createMediaStreamSource(stream);

            const analyser = this.chatAudioContext.createAnalyser();
            analyser.fftSize = 64;
            this.chatAnalyser = analyser;
            source.connect(analyser);

            const processor = this.chatAudioContext.createScriptProcessor(4096, 1, 1);
            this.chatScriptNode = processor;
            processor.onaudioprocess = (e) => {
                if (!this.chatRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                this.chatAudioChunks.push(new Float32Array(channelData));
            };
            source.connect(processor);
            processor.connect(this.chatAudioContext.destination);

            // Animate dynamic waveform bars reacting to real volume
            const waveBars = document.querySelectorAll("#chatRecordWaveform .chat-wave-bar");
            const updateWave = () => {
                if (!this.chatRecording || !this.chatAnalyser) return;
                const data = new Uint8Array(this.chatAnalyser.frequencyBinCount);
                this.chatAnalyser.getByteFrequencyData(data);
                for (let i = 0; i < waveBars.length; i++) {
                    const val = data[i * 2] || data[0] || 0;
                    const h = Math.max(4, Math.min(22, (val / 255) * 26));
                    waveBars[i].style.height = `${h}px`;
                    waveBars[i].style.animation = "none";
                }
                this.chatWaveAnimFrame = requestAnimationFrame(updateWave);
            };
            this.chatWaveAnimFrame = requestAnimationFrame(updateWave);

            // Optional Web Speech API for real-time speech preview
            const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (SpeechRec) {
                try {
                    const recognition = new SpeechRec();
                    recognition.continuous = true;
                    recognition.interimResults = true;
                    recognition.lang = "en-US";
                    recognition.onresult = (event) => {
                        let interim = "";
                        let final = "";
                        for (let i = event.resultIndex; i < event.results.length; ++i) {
                            if (event.results[i].isFinal) {
                                final += event.results[i][0].transcript;
                            } else {
                                interim += event.results[i][0].transcript;
                            }
                        }
                        if (final) this.chatFinalSpeechTranscript += " " + final;
                        const displayText = (this.chatFinalSpeechTranscript + " " + interim).trim();
                        if (displayText && this.chatRecordStatus) {
                            this.chatRecordStatus.textContent = `"${displayText}"`;
                            this.chatRecordStatus.classList.add("interim-text");
                        }
                    };
                    recognition.onerror = () => {};
                    recognition.start();
                    this.chatSpeechRecognition = recognition;
                } catch (e) {
                    // Ignore speech recognition error; Whisper will process audio
                }
            }
        } catch (err) {
            console.error("Microphone access error:", err);
            this.cancelChatVoiceRecord();
            alert("Could not access microphone. Please grant microphone permission in your browser.");
        }
    }

    cancelChatVoiceRecord() {
        this.cleanupChatVoiceRecord();
        if (this.chatTextarea) {
            this.chatTextarea.value = this.chatPreRecordText;
            this.chatTextarea.focus();
        }
    }

    async finishChatVoiceRecord() {
        if (!this.chatRecording) return;
        if (this.chatRecordStatus) {
            this.chatRecordStatus.textContent = "Transcribing audio...";
            this.chatRecordStatus.classList.remove("interim-text");
        }
        if (this.btnChatRecordDone) this.btnChatRecordDone.disabled = true;

        const chunks = this.chatAudioChunks.slice();
        const sampleRate = this.chatAudioContext ? this.chatAudioContext.sampleRate : 16000;
        const liveTranscript = this.chatFinalSpeechTranscript.trim();

        this.cleanupChatVoiceRecord();

        let transcribedText = "";
        if (chunks.length > 0) {
            try {
                const wavBlob = this.encodePcmToWav(chunks, sampleRate);
                const res = await fetch("/api/transcribe", {
                    method: "POST",
                    headers: { "Content-Type": "audio/wav" },
                    body: wavBlob,
                });
                if (res.ok) {
                    const data = await res.json();
                    transcribedText = (data.text || "").trim();
                }
            } catch (e) {
                console.warn("Backend Whisper STT failed, using live transcript fallback:", e);
            }
        }

        // Fallback to Web Speech API transcript if Whisper was empty or silent
        if (!transcribedText && liveTranscript) {
            transcribedText = liveTranscript;
        }

        if (this.chatTextarea) {
            if (transcribedText) {
                const prev = this.chatPreRecordText ? this.chatPreRecordText.trim() + " " : "";
                this.chatTextarea.value = prev + transcribedText;
            } else {
                this.chatTextarea.value = this.chatPreRecordText;
            }
            this.chatTextarea.style.height = "auto";
            this.chatTextarea.style.height = `${Math.min(220, this.chatTextarea.scrollHeight)}px`;
            this.chatTextarea.focus();
            const len = this.chatTextarea.value.length;
            this.chatTextarea.setSelectionRange(len, len);
        }
    }

    cleanupChatVoiceRecord() {
        this.chatRecording = false;
        if (this.chatRecordTimerInterval) {
            clearInterval(this.chatRecordTimerInterval);
            this.chatRecordTimerInterval = null;
        }
        if (this.chatWaveAnimFrame) {
            cancelAnimationFrame(this.chatWaveAnimFrame);
            this.chatWaveAnimFrame = null;
        }
        if (this.chatSpeechRecognition) {
            try {
                this.chatSpeechRecognition.stop();
            } catch (e) {}
            this.chatSpeechRecognition = null;
        }
        if (this.chatScriptNode) {
            try {
                this.chatScriptNode.disconnect();
            } catch (e) {}
            this.chatScriptNode = null;
        }
        if (this.chatAnalyser) {
            try {
                this.chatAnalyser.disconnect();
            } catch (e) {}
            this.chatAnalyser = null;
        }
        if (this.chatAudioContext) {
            try {
                this.chatAudioContext.close();
            } catch (e) {}
            this.chatAudioContext = null;
        }
        if (this.chatMediaStream) {
            this.chatMediaStream.getTracks().forEach((track) => track.stop());
            this.chatMediaStream = null;
        }
        if (this.chatBoxWrapper) this.chatBoxWrapper.classList.remove("is-recording");
        if (this.btnChatMicRecord) this.btnChatMicRecord.classList.remove("recording");
        if (this.chatRecordDock) this.chatRecordDock.style.display = "none";
        if (this.btnChatRecordDone) this.btnChatRecordDone.disabled = false;
    }

    encodePcmToWav(chunks, inputSampleRate) {
        let totalLength = 0;
        for (let i = 0; i < chunks.length; i++) totalLength += chunks[i].length;
        const merged = new Float32Array(totalLength);
        let offset = 0;
        for (let i = 0; i < chunks.length; i++) {
            merged.set(chunks[i], offset);
            offset += chunks[i].length;
        }

        const targetRate = 16000;
        let outputSamples = merged;
        if (inputSampleRate !== targetRate && totalLength > 0) {
            const ratio = inputSampleRate / targetRate;
            const newLength = Math.round(totalLength / ratio);
            outputSamples = new Float32Array(newLength);
            for (let i = 0; i < newLength; i++) {
                outputSamples[i] = merged[Math.round(i * ratio)] || 0;
            }
        }

        const buffer = new ArrayBuffer(44 + outputSamples.length * 2);
        const view = new DataView(buffer);

        const writeString = (view, offset, str) => {
            for (let i = 0; i < str.length; i++) {
                view.setUint8(offset + i, str.charCodeAt(i));
            }
        };

        writeString(view, 0, "RIFF");
        view.setUint32(4, 36 + outputSamples.length * 2, true);
        writeString(view, 8, "WAVE");
        writeString(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true); // PCM
        view.setUint16(22, 1, true); // Mono
        view.setUint32(24, targetRate, true);
        view.setUint32(28, targetRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(view, 36, "data");
        view.setUint32(40, outputSamples.length * 2, true);

        let p = 44;
        for (let i = 0; i < outputSamples.length; i++) {
            const s = Math.max(-1, Math.min(1, outputSamples[i]));
            view.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7fff, true);
            p += 2;
        }
        return new Blob([view], { type: "audio/wav" });
    }

    scrollChatToBottom() {
        if (this.chatMessagesContainer) {
            this.chatMessagesContainer.scrollTop = this.chatMessagesContainer.scrollHeight;
        }
    }

    cleanHumanText(text) {
        if (!text) return "";
        let cleaned = String(text);
        // Strip tool_call XML blocks and internal __TOOL_CALL__ strings
        cleaned = cleaned.replace(/__TOOL_CALL__[\s\S]*?(?:$|(?=\n\n)|(?<=\]\}))/gi, "");
        cleaned = cleaned.replace(/__TOOL_CALL__[^\]]*\]?/gi, "");
        cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "");
        cleaned = cleaned.replace(/<\/?(?:tool_call|function|parameter)[^>]*>/gi, "");
        // Strip bracketed tool calls e.g. [system.time], [perception.what_do_you_see], [music.play(query='...')]
        cleaned = cleaned.replace(/\[\s*[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+(?:\([^\]]*\)|:[^\]]*|)\s*\]/gi, "");
        // Strip internal category tags e.g. [conversation], [tools], [call], [meta], [greeting]
        cleaned = cleaned.replace(/\[\s*(?:conversation|tools?|call|meta|internal|action|greeting)\b[^\]]*\]/gi, "");
        // Strip planning or handling prefixes e.g. GREETING_HANDLING: ...
        cleaned = cleaned.replace(/^[A-Z_]+_HANDLING:\s*[^\n]*\n?/gim, "");
        cleaned = cleaned.replace(/\b[A-Z_]+_HANDLING:\s*[^\n]*/gi, "");
        // Strip explicit motion markers like [motion: wave], [motion: raise_hand left], [motion: lean_in]
        cleaned = cleaned.replace(/\[\s*(?:motion|action|emotion|character\.)\s*:[^\]]*\]/gi, "");
        // Strip general bracketed tags e.g. [motion: wave] or [character.execute_action ...]
        cleaned = cleaned.replace(/\[(?:motion|action|emotion)?:?\s*(?:wave|nod|point|kick|walk|turn|sit|stand|lean_in|raise_hand|stop)[^\]]*\]/gi, "");
        // Strip stage directions in asterisks e.g. *waves*, *smiles*, *nods*
        cleaned = cleaned.replace(/\*(?:waves?|raises?\s+(?:her\s+|his\s+|right\s+|left\s+)?hand|points?|kicks?|nods?|smiles?|stops?|leans?|looks?\s+around|shrugs?|pauses?|blinks?|winks?)[^*]*\*/gi, "");
        // Strip stray unclosed motion tags or bracketed prefixes at end of stream like [motion: or [system.
        cleaned = cleaned.replace(/\[\s*(?:motion|action|emotion|[a-zA-Z0-9_\-]+\.)?:?[^\]\n]*$/i, "");
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
        if (btnChat) btnChat.addEventListener("click", () => this.setView("chat"));

        const urlParams = new URLSearchParams(window.location.search);
        const requestedView = urlParams.get("view");
        if (requestedView === "chat") {
            setTimeout(() => this.setView("chat"), 30);
        }

        btnFs.addEventListener("click", () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen().catch(() => {});
            } else {
                document.exitFullscreen().catch(() => {});
            }
        });

        if (btnClear && this.terminalFeed) {
            btnClear.addEventListener("click", () => {
                this.terminalFeed.innerHTML = "";
            });
        }

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
                    this.applyPersonaTheme(data.active);
                }
            }
        } catch (e) {}

        this.personaSelect.addEventListener("change", async (e) => {
            const chosen = e.target.value;
            try {
                localStorage.setItem("robot_active_personality", chosen);
                this.applyPersonaTheme(chosen);
                const disp = this.formatPersonaName(chosen);
                if (this.bubbleTag) {
                    this.bubbleTag.textContent = `${disp}: WARMING UP...`;
                    this.bubbleTag.style.color = "var(--accent-amber, #f59e0b)";
                }
                if (this.bubbleContent) {
                    this.bubbleContent.textContent = "Priming neural cache & sensory context in background...";
                }
                // Dynamically update all assistant sender labels in the current conversation view
                document.querySelectorAll(".msg-assistant .chat-msg-sender").forEach((el) => {
                    el.textContent = disp;
                });
                const switchRes = await fetch(`/personality/${chosen}`, { method: "POST" });
                if (switchRes.ok) {
                    this.logEvent("PERSONA", `Personality switched to ${disp} (priming in background)`, "system");
                    if (this.personaName) this.personaName.textContent = disp;
                    this.syncChatTopBar();
                }
            } catch (err) {
                this.logEvent("ERROR", `Failed to switch personality: ${err.message}`, "error");
            }
        });
    }

    applyPersonaTheme(personaKey) {
        if (!personaKey) return;
        const theme = personaKey.toLowerCase();
        document.body.setAttribute("data-theme", theme);
        document.documentElement.setAttribute("data-theme", theme);
        const engine = this.characterEngine || window.characterEngine;
        if (engine && typeof engine.setPersonaTheme === "function") {
            engine.setPersonaTheme(theme);
        }
    }

    /**
     * Initialize the 3-way LLM provider toggle (Local Qwen, Cloud Gemini, OpenAI-Compatible API),
     * model scanning dropdown, and cyberdeck settings modal.
     * @returns {Promise<void>}
     */
    async initLlmProviderToggle() {
        const btn = document.getElementById("btnLlmToggle");
        const icon = document.getElementById("llmToggleIcon");
        const label = document.getElementById("llmToggleLabel");
        const sub = document.getElementById("llmToggleSub");
        const badge = document.getElementById("llmModeBadge");

        this.btnLlmToggle = btn;
        this.currentLlmProvider = "local";
        const storedModel = typeof localStorage !== "undefined" ? localStorage.getItem("active_openai_model") : null;
        this.activeOpenaiModel = storedModel || "deepseek/deepseek-v4.1-flash";

        const panelStrip = document.getElementById("activeModelStrip");
        const panelBadge = document.getElementById("activeModelBadge");
        const panelName = document.getElementById("activeModelName");
        const panelBtn = document.getElementById("btnSwapModelPanel");
        const panelBtnText = document.getElementById("swapModelBtnText");
        const panelBtnIcon = document.getElementById("swapBtnIcon");

        // Local GGUF Model switcher controls (Top Nav & Chat mode)
        const topNavLocal = document.getElementById("topNavLocalControls");
        const chatLocalControls = document.getElementById("chatLocalControls");

        // OpenAI Model scanning controls & modal (Top Nav, HUD, and Chat mode)
        const topNavOpenai = document.getElementById("topNavOpenaiControls");
        const btnTopNavModelPicker = document.getElementById("btnTopNavModelPicker");
        const btnTopNavScanOpenai = document.getElementById("btnTopNavScanOpenai");
        const btnTopNavOpenaiConfig = document.getElementById("btnTopNavOpenaiConfig");

        const openaiControls = document.getElementById("openaiModelControls");
        const btnHudModelPicker = document.getElementById("btnHudModelPicker");
        const btnConfig = document.getElementById("btnOpenaiConfig");

        const chatOpenaiControls = document.getElementById("chatOpenaiControls");
        const btnChatModelPicker = document.getElementById("btnChatModelPicker");
        const btnChatConfig = document.getElementById("btnChatOpenaiConfig");
        const chatActiveModelPill = document.getElementById("chatActiveModelPill");

        // Searchable Model Picker Modal Elements
        const modelSearchModal = document.getElementById("openaiModelSearchModal");
        const modelSearchInput = document.getElementById("modelSearchInput");
        const btnClearModelSearch = document.getElementById("btnClearModelSearch");
        const modelSearchResultsList = document.getElementById("modelSearchResultsList");
        const modelSearchCountLabel = document.getElementById("modelSearchCountLabel");
        const countAllModelsEl = document.getElementById("countAllModels");
        const vendorTabs = document.querySelectorAll(".model-vendor-tabs .vendor-tab");
        const btnModalRescanModels = document.getElementById("btnModalRescanModels");
        const btnModalOpenConfig = document.getElementById("btnModalOpenConfig");
        const btnCloseModelSearchModal = document.getElementById("btnCloseModelSearchModal");
        const btnCancelModelSearch = document.getElementById("btnCancelModelSearch");

        // OpenAI Endpoint Configuration Modal Elements (Dual-Column Cyberdeck Grid)
        const configModal = document.getElementById("openaiConfigModal");
        const btnCloseModal = document.getElementById("btnCloseOpenaiModal");
        const btnCancelModal = document.getElementById("btnCancelOpenaiModal");
        const btnSaveConfig = document.getElementById("btnSaveOpenaiConfig");
        const btnTestScanModal = document.getElementById("btnTestAndScanModal");

        // Column 1: OpenRouter
        const colOpenrouter = document.getElementById("colOpenrouter");
        const btnActivateOr = document.getElementById("btnActivateOpenrouter");
        const inputOrBaseUrl = document.getElementById("cfgOrBaseUrl");
        const inputOrApiKey = document.getElementById("cfgOrApiKey");
        const inputOrModel = document.getElementById("cfgOrModel");
        const btnScanOr = document.getElementById("btnScanOrModels");
        const btnToggleOrApiKey = document.getElementById("btnToggleOrApiKey");
        const orStatusBadge = document.getElementById("orStatusBadge");

        // Column 2: AI Credits
        const colAicredits = document.getElementById("colAicredits");
        const btnActivateAi = document.getElementById("btnActivateAicredits");
        const inputAiBaseUrl = document.getElementById("cfgAiBaseUrl");
        const inputAiApiKey = document.getElementById("cfgAiApiKey");
        const inputAiModel = document.getElementById("cfgAiModel");
        const btnScanAi = document.getElementById("btnScanAiModels");
        const btnToggleAiApiKey = document.getElementById("btnToggleAiApiKey");
        const aiStatusBadge = document.getElementById("aiStatusBadge");

        // Advanced / Manual Section
        const btnToggleCustomEndpoint = document.getElementById("btnToggleCustomEndpoint");
        const customEndpointBody = document.getElementById("customEndpointBody");
        const caretCustomEndpoint = document.getElementById("caretCustomEndpoint");
        const inputManualBaseUrl = document.getElementById("cfgManualBaseUrl");
        const inputManualApiKey = document.getElementById("cfgManualApiKey");
        const inputManualModel = document.getElementById("cfgManualModel");
        const btnActivateManual = document.getElementById("btnActivateManual");

        // Legacy / Shared
        const inputBaseUrl = document.getElementById("cfgOpenaiBaseUrl") || document.getElementById("inputOpenaiBaseUrl");
        const inputApiKey = document.getElementById("cfgOpenaiApiKey") || document.getElementById("inputOpenaiApiKey");
        const inputModel = document.getElementById("cfgOpenaiModel") || document.getElementById("inputOpenaiModel");
        const inputContext = document.getElementById("cfgOpenaiContext") || document.getElementById("inputOpenaiContext");
        const checkCaching = document.getElementById("cfgOpenaiPromptCaching") || document.getElementById("checkOpenaiPromptCaching");
        const modalFeedback = document.getElementById("openaiConfigFeedback");
        const selectPreset = document.getElementById("cfgOpenaiPreset");

        this.activeCustomProvider = "openrouter";

        /**
         * Select and highlight the active provider column.
         * @param {string} prov - 'openrouter', 'aicredits', or 'custom'.
         * @returns {void}
         */
        const setProviderActive = (prov) => {
            this.activeCustomProvider = prov;
            if (colOpenrouter) colOpenrouter.classList.toggle("active-provider", prov === "openrouter");
            if (colAicredits) colAicredits.classList.toggle("active-provider", prov === "aicredits");

            if (btnActivateOr) {
                const lbl = btnActivateOr.querySelector(".activate-label");
                if (lbl) lbl.textContent = prov === "openrouter" ? "ACTIVE" : "USE THIS";
            }
            if (btnActivateAi) {
                const lbl = btnActivateAi.querySelector(".activate-label");
                if (lbl) lbl.textContent = prov === "aicredits" ? "ACTIVE" : "USE THIS";
            }

            if (modalFeedback) {
                if (prov === "openrouter") {
                    modalFeedback.textContent = "🌐 OpenRouter selected as active provider. Free models available.";
                    modalFeedback.className = "cyberdeck-modal-feedback success";
                } else if (prov === "aicredits") {
                    modalFeedback.textContent = "⚡ AI Credits selected as active provider. High-throughput inference active.";
                    modalFeedback.className = "cyberdeck-modal-feedback success";
                } else {
                    modalFeedback.textContent = "⚙️ Custom / Manual endpoint selected.";
                    modalFeedback.className = "cyberdeck-modal-feedback";
                }
            }
        };

        if (btnActivateOr) btnActivateOr.addEventListener("click", () => setProviderActive("openrouter"));
        if (btnActivateAi) btnActivateAi.addEventListener("click", () => setProviderActive("aicredits"));
        if (btnActivateManual) btnActivateManual.addEventListener("click", () => setProviderActive("custom"));

        if (btnToggleOrApiKey && inputOrApiKey) {
            btnToggleOrApiKey.addEventListener("click", () => {
                inputOrApiKey.type = inputOrApiKey.type === "password" ? "text" : "password";
            });
        }
        if (btnToggleAiApiKey && inputAiApiKey) {
            btnToggleAiApiKey.addEventListener("click", () => {
                inputAiApiKey.type = inputAiApiKey.type === "password" ? "text" : "password";
            });
        }
        if (btnToggleCustomEndpoint && customEndpointBody) {
            btnToggleCustomEndpoint.addEventListener("click", () => {
                const isHidden = customEndpointBody.style.display === "none";
                customEndpointBody.style.display = isHidden ? "block" : "none";
                if (caretCustomEndpoint) caretCustomEndpoint.textContent = isHidden ? "▴" : "▾";
            });
        }

        if (btnScanOr) {
            btnScanOr.addEventListener("click", async () => {
                if (modalFeedback) {
                    modalFeedback.textContent = "Scanning OpenRouter models...";
                    modalFeedback.className = "cyberdeck-modal-feedback";
                }
                const models = await this.scanOpenaiModels(true, "openrouter");
                if (modalFeedback) {
                    modalFeedback.textContent = `✓ Discovered ${models.length} models on OpenRouter!`;
                    modalFeedback.className = "cyberdeck-modal-feedback success";
                }
            });
        }

        if (btnScanAi) {
            btnScanAi.addEventListener("click", async () => {
                if (modalFeedback) {
                    modalFeedback.textContent = "Scanning AI Credits models...";
                    modalFeedback.className = "cyberdeck-modal-feedback";
                }
                const models = await this.scanOpenaiModels(true, "aicredits");
                if (modalFeedback) {
                    modalFeedback.textContent = `✓ Discovered ${models.length} models on AI Credits!`;
                    modalFeedback.className = "cyberdeck-modal-feedback success";
                }
            });
        }

        this.scannedOpenaiModels = [];
        this.currentModelVendorFilter = "all";
        this.isModelSearchModalOpen = false;

        /**
         * Update all model display labels across Top Nav, HUD, and Chat mode.
         * @param {string} [modelName] - Model identifier to display.
         * @returns {void}
         */
        this.updateOpenaiModelDisplays = (modelName) => {
            const m = modelName || this.activeOpenaiModel || (typeof localStorage !== "undefined" ? localStorage.getItem("active_openai_model") : null) || "deepseek/deepseek-v4.1-flash";
            this.activeOpenaiModel = m;
            try {
                if (typeof localStorage !== "undefined") {
                    localStorage.setItem("active_openai_model", m);
                }
            } catch (e) {}
            const topNavLabel = document.getElementById("topNavModelPickerLabel");
            if (topNavLabel) topNavLabel.textContent = m;
            const hudLabel = document.getElementById("hudModelPickerLabel");
            if (hudLabel) hudLabel.textContent = m;
            const chatLabel = document.getElementById("chatModelPickerLabel");
            if (chatLabel) chatLabel.textContent = m;

            const chatModelLabel = document.getElementById("chatModelLabel") || this.chatModelLabel;
            if (chatModelLabel && this.currentLlmProvider === "openai_compatible") {
                chatModelLabel.textContent = "CUSTOM API";
                if (chatActiveModelPill) {
                    chatActiveModelPill.title = `Current Provider: Custom OpenAI-Compatible API (${m}). Click to cycle provider.`;
                }
            }
            if (sub && this.currentLlmProvider === "openai_compatible") {
                sub.textContent = `${m} • CACHED`;
            }
            if (panelName && this.currentLlmProvider === "openai_compatible") {
                panelName.textContent = "Prompt Caching Active";
                panelName.title = `Active Model: ${m} (Prompt Caching Active)`;
            }
        };

        /**
         * Update UI indicators across telemetry panel and top header for active LLM provider.
         * @param {string} prov - 'local', 'gemini', or 'openai_compatible'.
         * @param {string|null} [mode] - Optional mode string from status endpoint.
         * @param {boolean} [isTransitioning=false] - Whether a switch is currently executing.
         * @param {string|null} [targetProv=null] - Desired target provider during transition.
         * @returns {void}
         */
        this.updateLlmToggleUi = (prov, mode, isTransitioning = false, targetProv = null) => {
            if (isTransitioning) {
                if (btn) btn.classList.add("switching");
                if (panelBtn) panelBtn.disabled = true;

                if (targetProv === "gemini") {
                    if (label) label.textContent = "CONNECTING GEMINI...";
                    if (sub) sub.textContent = "FREEING 3.4GB VRAM";
                    if (panelName) panelName.textContent = "Connecting Cloud Gemini (Freeing VRAM)...";
                } else if (targetProv === "openai_compatible") {
                    if (label) label.textContent = "CONNECTING OPENAI...";
                    if (sub) sub.textContent = "FREEING 3.4GB VRAM";
                    if (panelName) panelName.textContent = "Connecting OpenAI-Compatible API (Prompt Caching)...";
                } else {
                    if (label) label.textContent = "STARTING LOCAL...";
                    if (sub) sub.textContent = "ALLOCATING VRAM";
                    if (panelName) panelName.textContent = "Starting Local Qwen (Allocating VRAM)...";
                }

                if (badge) badge.textContent = "WAIT";
                if (panelBtnText) panelBtnText.textContent = "SWAPPING...";
                const chatModelLabel = document.getElementById("chatModelLabel") || this.chatModelLabel;
                if (chatModelLabel) chatModelLabel.textContent = "SWAPPING...";
                return;
            }

            if (btn) btn.classList.remove("switching");
            if (panelBtn) panelBtn.disabled = false;

            const safeProv = prov || "local";
            this.currentLlmProvider = safeProv;

            const activeM = this.activeOpenaiModel || "gpt-4o";

            if (safeProv === "openai_compatible") {
                if (btn) {
                    btn.classList.remove("gemini-mode", "local-mode");
                    btn.classList.add("openai-mode");
                }
                if (panelStrip) {
                    panelStrip.classList.remove("gemini-active");
                    panelStrip.classList.add("openai-active");
                }
                if (topNavOpenai) topNavOpenai.style.display = "inline-flex";
                if (openaiControls) openaiControls.style.display = "flex";
                if (chatOpenaiControls) chatOpenaiControls.style.display = "inline-flex";
                if (topNavLocal) topNavLocal.style.display = "none";
                if (chatLocalControls) chatLocalControls.style.display = "none";

                if (icon) icon.textContent = "🌐";
                if (label) label.textContent = "OPENAI COMPAT";
                if (sub) sub.textContent = `${activeM} • CACHED`;
                if (badge) badge.textContent = "OPENAI";
                if (panelBadge) panelBadge.textContent = "CUSTOM API";
                if (panelName) {
                    panelName.textContent = "Prompt Caching Active";
                    panelName.title = `Active Model: ${activeM} (Prompt Caching Active)`;
                }
                if (panelBtnText) panelBtnText.textContent = "SWAP TO LOCAL";
                if (panelBtnIcon) panelBtnIcon.textContent = "⚡";
                if (btn) btn.title = `Active: OpenAI-Compatible API (${activeM}, Prompt Caching). Click to cycle to Local Qwen (RTX 3050).`;

                this.updateOpenaiModelDisplays(activeM);
            } else if (safeProv === "gemini") {
                if (btn) {
                    btn.classList.remove("local-mode", "openai-mode");
                    btn.classList.add("gemini-mode");
                }
                if (panelStrip) {
                    panelStrip.classList.remove("openai-active");
                    panelStrip.classList.add("gemini-active");
                }
                if (topNavOpenai) topNavOpenai.style.display = "none";
                if (openaiControls) openaiControls.style.display = "none";
                if (chatOpenaiControls) chatOpenaiControls.style.display = "none";
                if (topNavLocal) topNavLocal.style.display = "none";
                if (chatLocalControls) chatLocalControls.style.display = "none";

                if (icon) icon.textContent = "✨";
                if (label) label.textContent = "GEMINI (CLOUD)";
                if (sub) sub.textContent = "1M CONTEXT • HIGH INTEL";
                if (badge) badge.textContent = "GEMINI";
                if (panelBadge) panelBadge.textContent = "CLOUD API";
                if (panelName) panelName.textContent = "Gemini 3.5 Flash Lite • 1M Context";
                if (panelBtnText) panelBtnText.textContent = "SWAP TO OPENAI";
                if (panelBtnIcon) panelBtnIcon.textContent = "🌐";
                if (btn) btn.title = "Active: Cloud Gemini 3.5 Flash Lite (1M Context, High Intelligence). Click to swap to OpenAI API.";
            } else {
                if (btn) {
                    btn.classList.remove("gemini-mode", "openai-mode");
                    btn.classList.add("local-mode");
                }
                if (panelStrip) {
                    panelStrip.classList.remove("gemini-active", "openai-active");
                }
                if (topNavOpenai) topNavOpenai.style.display = "none";
                if (openaiControls) openaiControls.style.display = "none";
                if (chatOpenaiControls) chatOpenaiControls.style.display = "none";
                if (topNavLocal) topNavLocal.style.display = "inline-flex";
                if (chatLocalControls) chatLocalControls.style.display = "inline-flex";

                const ctxK = Math.round((this.localContextWindow || 16384) / 1024);
                if (icon) icon.textContent = "⚡";
                if (label) label.textContent = "LOCAL (RTX 3050)";
                if (sub) sub.textContent = `LOW LATENCY • ${ctxK}K`;
                if (badge) badge.textContent = "LOCAL";
                if (panelBadge) panelBadge.textContent = "LOCAL GPU";
                if (panelName) panelName.textContent = `Qwen 3.5 4B • RTX 3050 • ${ctxK}K`;
                if (panelBtnText) panelBtnText.textContent = "SWAP TO GEMINI";
                if (panelBtnIcon) panelBtnIcon.textContent = "✨";
                if (btn) btn.title = `Active: Local Qwen 3.5 4B on RTX 3050 (Low Latency, ${ctxK}K Context). Click to swap to Cloud Gemini.`;
                if (typeof this.updateLocalModelDisplays === "function" && this.activeLocalModelId) {
                    this.updateLocalModelDisplays(this.activeLocalModelId);
                }
            }

            if (typeof this.syncChatTopBar === "function") this.syncChatTopBar();
        };

        /**
         * Scan available models from the configured OpenAI-compatible endpoint.
          /**
         * Scan available models from the OpenAI-compatible endpoint.
         * @param {boolean} [autoSelectCurrent=true] - Whether to synchronize active model selection.
         * @param {string|null} [preset=null] - Optional preset target ('openrouter', 'aicredits', etc.).
         * @returns {Promise<Array<Object>>}
         */
        this.scanOpenaiModels = async (autoSelectCurrent = true, preset = null) => {
            const scanIcons = [
                document.querySelector("#btnScanOpenaiModels .scan-icon"),
                document.querySelector("#btnChatScanOpenai .scan-icon"),
                document.querySelector("#btnTopNavScanOpenai .scan-icon"),
                document.querySelector("#modalRescanIcon")
            ].filter(Boolean);
            scanIcons.forEach((ic) => ic.classList.add("spinning"));

            const scanButtons = [
                document.getElementById("btnScanOpenaiModels"),
                document.getElementById("btnChatScanOpenai"),
                document.getElementById("btnTopNavScanOpenai"),
                document.getElementById("btnModalRescanModels")
            ].filter(Boolean);
            scanButtons.forEach((b) => b.disabled = true);

            try {
                let url = "/api/llm/openai/models";
                if (preset) {
                    url += `?preset=${encodeURIComponent(preset)}`;
                }
                const res = await fetch(url);
                const data = await res.json();
                if (data.status === "ok" && Array.isArray(data.models)) {
                    this.scannedOpenaiModels = data.models;
                    this.activeOpenaiEndpoint = data.base_url || "";
                    const targetModel = data.active_model || data.current_model || data.model || this.activeOpenaiModel;
                    if (targetModel) {
                        this.activeOpenaiModel = targetModel;
                    }

                    const endpointHint = document.getElementById("modelEndpointHint");
                    if (endpointHint && data.base_url) {
                        const provTag = data.provider_preset ? ` [${data.provider_preset.toUpperCase()}]` : "";
                        endpointHint.textContent = `Endpoint: ${data.base_url}${provTag}`;
                    }

                    this.updateOpenaiModelDisplays(this.activeOpenaiModel);
                    if (this.isModelSearchModalOpen) {
                        this.renderModelSearchResults();
                    }
                    this.logEvent("LLM", `Discovered ${data.count} models at ${data.provider_preset || "OpenAI"} endpoint (Active: ${this.activeOpenaiModel}).`, "success");
                    return data.models;
                } else {
                    this.logEvent("WARNING", `Model scan warning: ${data.message || data.error || "No models returned"}`, "warning");
                }
            } catch (err) {
                this.logEvent("ERROR", `Failed scanning OpenAI models: ${err.message}`, "error");
            } finally {
                scanIcons.forEach((ic) => ic.classList.remove("spinning"));
                scanButtons.forEach((b) => b.disabled = false);
            }
            return [];
        };

        /**
         * Handle user selection of a different OpenAI model.
         * @param {string} selectedModel - Selected model ID.
         * @returns {Promise<void>}
         */
        const handleModelSelectChange = async (selectedModel) => {
            if (!selectedModel) return;
            this.logEvent("LLM", `Selecting OpenAI model "${selectedModel}"...`, "system");
            try {
                const res = await fetch("/api/llm/openai/model", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model: selectedModel })
                });
                const data = await res.json();
                if (data.status === "ok") {
                    const newModel = data.active_model || data.model || selectedModel;
                    this.activeOpenaiModel = newModel;
                    try {
                        if (typeof localStorage !== "undefined") {
                            localStorage.setItem("active_openai_model", newModel);
                        }
                    } catch (e) {}
                    this.updateOpenaiModelDisplays(newModel);
                    this.updateLlmToggleUi("openai_compatible");
                    this.logEvent("LLM", `Active OpenAI model set to "${newModel}" with prompt caching.`, "success");
                } else {
                    this.logEvent("ERROR", `Failed selecting model: ${data.message || data.error}`, "error");
                }
            } catch (err) {
                this.logEvent("ERROR", `Network error updating model: ${err.message}`, "error");
            }
        };

        /**
         * Determine vendor classification tag for a given model ID.
         * @param {string} id - Model identifier.
         * @returns {string} Vendor name.
         */
        const getModelVendorTag = (id) => {
            const low = (id || "").toLowerCase();
            if (low.startsWith("openai/") || low.includes("gpt") || low.includes("o1-") || low.includes("o3-")) return "OPENAI";
            if (low.startsWith("anthropic/") || low.includes("claude")) return "CLAUDE";
            if (low.startsWith("google/") || low.includes("gemini") || low.includes("gemma")) return "GOOGLE";
            if (low.startsWith("meta-llama/") || low.includes("llama")) return "LLAMA";
            if (low.startsWith("qwen/") || low.includes("qwen")) return "QWEN";
            if (low.startsWith("deepseek/") || low.includes("deepseek")) return "DEEPSEEK";
            if (low.startsWith("mistralai/") || low.includes("mistral") || low.includes("codestral")) return "MISTRAL";
            if (low.includes("/")) return low.split("/")[0].toUpperCase();
            return "API";
        };

        /**
         * Render filtered models in the searchable picker modal list.
         * @returns {void}
         */
        this.renderModelSearchResults = () => {
            if (!modelSearchResultsList) return;
            modelSearchResultsList.innerHTML = "";

            const query = (modelSearchInput ? modelSearchInput.value : "").trim().toLowerCase();
            const vendor = this.currentModelVendorFilter || "all";

            const filtered = this.scannedOpenaiModels.filter((m) => {
                const id = (m.id || "").toLowerCase();
                const name = (m.name || "").toLowerCase();

                // Vendor tab filter
                if (vendor !== "all") {
                    if (vendor === "free") {
                        if (!m.is_free && !id.includes(":free") && !id.includes("-free")) return false;
                    } else if (vendor === "openrouter") {
                        const isOr = (this.activeOpenaiEndpoint || "").includes("openrouter.ai") || (inputOrBaseUrl && inputOrBaseUrl.value.includes("openrouter.ai"));
                        if (!isOr && !id.includes("openrouter")) return false;
                    } else if (vendor === "aicredits") {
                        const isAi = (this.activeOpenaiEndpoint || "").includes("aicredits") || (inputAiBaseUrl && inputAiBaseUrl.value.includes("aicredits"));
                        if (!isAi && !id.includes("aicredits") && !id.includes("aion")) return false;
                    } else if (vendor === "openai" && !(id.includes("openai") || id.includes("gpt") || id.includes("o1") || id.includes("o3") || id.includes("chatgpt"))) return false;
                    else if (vendor === "anthropic" && !(id.includes("anthropic") || id.includes("claude"))) return false;
                    else if (vendor === "google" && !(id.includes("google") || id.includes("gemini") || id.includes("gemma"))) return false;
                    else if (vendor === "meta" && !(id.includes("meta") || id.includes("llama"))) return false;
                    else if (vendor === "qwen" && !id.includes("qwen")) return false;
                    else if (vendor === "deepseek" && !id.includes("deepseek")) return false;
                    else if (vendor === "mistral" && !(id.includes("mistral") || id.includes("codestral") || id.includes("mixtral"))) return false;
                }

                // Text search filter
                if (query) {
                    return id.includes(query) || name.includes(query);
                }
                return true;
            });

            if (countAllModelsEl) countAllModelsEl.textContent = this.scannedOpenaiModels.length;
            if (modelSearchCountLabel) {
                modelSearchCountLabel.textContent = `Showing ${filtered.length} of ${this.scannedOpenaiModels.length} models`;
            }

            if (filtered.length === 0) {
                const emptyRow = document.createElement("div");
                emptyRow.className = "model-list-empty";
                emptyRow.style.cssText = "padding: 24px; text-align: center; color: var(--text-dim); font-family: var(--font-mono); font-size: 11px;";
                emptyRow.textContent = query
                    ? `No models matching "${query}". Try clearing the search or switching vendor tabs.`
                    : "No models found. Click 'Rescan API' to fetch models from your endpoint.";
                modelSearchResultsList.appendChild(emptyRow);
                return;
            }

            filtered.forEach((m) => {
                const row = document.createElement("div");
                const isActive = m.id === this.activeOpenaiModel;
                row.className = `model-list-row ${isActive ? "active-model-row" : ""}`;
                row.setAttribute("role", "option");
                row.setAttribute("aria-selected", isActive ? "true" : "false");
                row.dataset.modelId = m.id;

                const vTag = getModelVendorTag(m.id);
                const isFree = Boolean(m.is_free || (m.id && (m.id.includes(":free") || m.id.includes("-free"))));
                row.innerHTML = `
                    <div class="model-row-info">
                        <span class="model-row-vendor-tag">${vTag}</span>
                        ${isFree ? '<span class="model-row-badge free-badge" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); margin-right: 6px; font-weight: 700;">FREE</span>' : ""}
                        <span class="model-row-name" title="${m.id}">${m.id}</span>
                    </div>
                    ${isActive ? '<span class="model-row-badge">✓ ACTIVE</span>' : ""}
                `;

                row.addEventListener("click", () => {
                    handleModelSelectChange(m.id);
                    this.closeModelSearchModal();
                });

                modelSearchResultsList.appendChild(row);
            });
        };

        /**
         * Open the cyberdeck searchable model picker modal.
         * @returns {void}
         */
        this.openModelSearchModal = () => {
            if (!modelSearchModal) return;
            this.isModelSearchModalOpen = true;
            modelSearchModal.style.display = "flex";

            if (modelSearchInput) {
                modelSearchInput.value = "";
                setTimeout(() => modelSearchInput.focus(), 50);
            }
            if (btnClearModelSearch) btnClearModelSearch.style.display = "none";
            this.currentModelVendorFilter = "all";
            vendorTabs.forEach((tab) => {
                tab.classList.toggle("active", tab.dataset.vendor === "all");
            });

            if (this.scannedOpenaiModels.length === 0) {
                this.scanOpenaiModels();
            } else {
                this.renderModelSearchResults();
            }
        };

        /**
         * Close the searchable model picker modal.
         * @returns {void}
         */
        this.closeModelSearchModal = () => {
            if (!modelSearchModal) return;
            this.isModelSearchModalOpen = false;
            modelSearchModal.style.display = "none";
        };

        // Wire model picker trigger buttons across Top Nav, HUD, and Chat mode
        if (btnTopNavModelPicker) btnTopNavModelPicker.addEventListener("click", () => this.openModelSearchModal());
        if (btnHudModelPicker) btnHudModelPicker.addEventListener("click", () => this.openModelSearchModal());
        if (btnChatModelPicker) btnChatModelPicker.addEventListener("click", () => this.openModelSearchModal());
        if (btnModalRescanModels) btnModalRescanModels.addEventListener("click", () => this.scanOpenaiModels());

        // Wire search input typing & clear
        if (modelSearchInput) {
            modelSearchInput.addEventListener("input", (e) => {
                if (btnClearModelSearch) {
                    btnClearModelSearch.style.display = e.target.value ? "inline-block" : "none";
                }
                this.renderModelSearchResults();
            });
            modelSearchInput.addEventListener("keydown", (e) => {
                if (e.key === "Escape") {
                    this.closeModelSearchModal();
                } else if (e.key === "Enter") {
                    const firstRow = modelSearchResultsList ? modelSearchResultsList.querySelector(".model-list-row") : null;
                    if (firstRow && firstRow.dataset.modelId) {
                        handleModelSelectChange(firstRow.dataset.modelId);
                        this.closeModelSearchModal();
                    }
                }
            });
        }

        if (btnClearModelSearch) {
            btnClearModelSearch.addEventListener("click", () => {
                if (modelSearchInput) {
                    modelSearchInput.value = "";
                    modelSearchInput.focus();
                }
                btnClearModelSearch.style.display = "none";
                this.renderModelSearchResults();
            });
        }

        // Wire vendor filter tabs
        vendorTabs.forEach((tab) => {
            tab.addEventListener("click", async () => {
                vendorTabs.forEach((t) => t.classList.remove("active"));
                tab.classList.add("active");
                this.currentModelVendorFilter = tab.dataset.vendor || "all";
                if (this.currentModelVendorFilter === "openrouter" && !(this.activeOpenaiEndpoint || "").includes("openrouter.ai")) {
                    await this.scanOpenaiModels(true, "openrouter");
                } else if (this.currentModelVendorFilter === "aicredits" && !(this.activeOpenaiEndpoint || "").includes("aicredits")) {
                    await this.scanOpenaiModels(true, "aicredits");
                } else {
                    this.renderModelSearchResults();
                }
            });
        });

        // Wire modal close actions
        if (btnCloseModelSearchModal) btnCloseModelSearchModal.addEventListener("click", () => this.closeModelSearchModal());
        if (btnCancelModelSearch) btnCancelModelSearch.addEventListener("click", () => this.closeModelSearchModal());
        if (modelSearchModal) {
            modelSearchModal.addEventListener("click", (e) => {
                if (e.target === modelSearchModal) this.closeModelSearchModal();
            });
        }

        /**
         * Open the cyberdeck OpenAI configuration modal and prefill current values for both providers.
         * @returns {Promise<void>}
         */
        const openConfigModal = async () => {
            try {
                const res = await fetch("/api/llm/status");
                if (res.ok) {
                    const data = await res.json();
                    if (data.openrouter) {
                        this.openrouterApiKey = data.openrouter.api_key || "";
                        if (inputOrBaseUrl) inputOrBaseUrl.value = data.openrouter.base_url || "https://openrouter.ai/api/v1";
                        if (inputOrApiKey) inputOrApiKey.value = data.openrouter.api_key || "";
                        if (inputOrModel) inputOrModel.value = data.openrouter.model || "meta-llama/llama-3.3-70b-instruct:free";
                        if (orStatusBadge) {
                            orStatusBadge.textContent = data.openrouter.has_api_key ? "✓ Configured" : "No Key Set";
                            orStatusBadge.className = `provider-status-badge ${data.openrouter.has_api_key ? "" : "missing"}`;
                        }
                    }
                    if (data.aicredits) {
                        this.aicreditsApiKey = data.aicredits.api_key || "";
                        if (inputAiBaseUrl) inputAiBaseUrl.value = data.aicredits.base_url || "https://aicredits.in/v1";
                        if (inputAiApiKey) inputAiApiKey.value = data.aicredits.api_key || "";
                        if (inputAiModel) inputAiModel.value = data.aicredits.model || "aion-labs/aion-3.0";
                        if (aiStatusBadge) {
                            aiStatusBadge.textContent = data.aicredits.has_api_key ? "✓ Configured" : "No Key Set";
                            aiStatusBadge.className = `provider-status-badge ${data.aicredits.has_api_key ? "" : "missing"}`;
                        }
                    }

                    const cfg = data.openai_compatible || {};
                    const curUrl = (cfg.base_url || "").toLowerCase();
                    if (curUrl.includes("openrouter.ai")) {
                        setProviderActive("openrouter");
                    } else if (curUrl.includes("aicredits")) {
                        setProviderActive("aicredits");
                    } else {
                        setProviderActive("custom");
                        if (inputManualBaseUrl) inputManualBaseUrl.value = cfg.base_url || "";
                        if (inputManualApiKey) inputManualApiKey.value = cfg.api_key || "";
                        if (inputManualModel) inputManualModel.value = cfg.model || "";
                    }

                    if (inputContext) inputContext.value = cfg.context_window || 128000;
                    if (checkCaching) checkCaching.checked = cfg.enable_prompt_caching !== false;
                }
            } catch (e) {}

            if (modalFeedback) {
                modalFeedback.textContent = "";
                modalFeedback.className = "cyberdeck-modal-feedback";
            }
            if (configModal) {
                configModal.style.display = "flex";
                configModal.classList.add("open");
            }
        };

        if (btnTopNavOpenaiConfig) btnTopNavOpenaiConfig.addEventListener("click", openConfigModal);
        if (btnConfig && configModal) btnConfig.addEventListener("click", openConfigModal);
        if (btnChatConfig && configModal) btnChatConfig.addEventListener("click", openConfigModal);
        if (btnModalOpenConfig) {
            btnModalOpenConfig.addEventListener("click", () => {
                this.closeModelSearchModal();
                openConfigModal();
            });
        }

        const closeModal = () => {
            if (configModal) {
                configModal.style.display = "none";
                configModal.classList.remove("open");
            }
        };

        if (btnCloseModal) btnCloseModal.addEventListener("click", closeModal);
        if (btnCancelModal) btnCancelModal.addEventListener("click", closeModal);
        if (configModal) {
            configModal.addEventListener("click", (e) => {
                if (e.target === configModal) closeModal();
            });
        }

        if (btnSaveConfig) {
            btnSaveConfig.addEventListener("click", async () => {
                const orBase = inputOrBaseUrl ? inputOrBaseUrl.value.trim() : "https://openrouter.ai/api/v1";
                const orKey = inputOrApiKey ? inputOrApiKey.value.trim() : "";
                const orModel = inputOrModel ? inputOrModel.value.trim() : "meta-llama/llama-3.3-70b-instruct:free";

                const aiBase = inputAiBaseUrl ? inputAiBaseUrl.value.trim() : "https://aicredits.in/v1";
                const aiKey = inputAiApiKey ? inputAiApiKey.value.trim() : "";
                const aiModel = inputAiModel ? inputAiModel.value.trim() : "aion-labs/aion-3.0";

                const manualBase = inputManualBaseUrl ? inputManualBaseUrl.value.trim() : "";
                const manualKey = inputManualApiKey ? inputManualApiKey.value.trim() : "";
                const manualModel = inputManualModel ? inputManualModel.value.trim() : "";

                const activeProv = this.activeCustomProvider || "openrouter";
                let activeBase = orBase;
                let activeKey = orKey;
                let activeModel = orModel;

                if (activeProv === "aicredits") {
                    activeBase = aiBase;
                    activeKey = aiKey;
                    activeModel = aiModel;
                } else if (activeProv === "custom") {
                    activeBase = manualBase || orBase;
                    activeKey = manualKey || orKey;
                    activeModel = manualModel || orModel;
                }

                const payload = {
                    active_provider: activeProv,
                    preset: activeProv,
                    openrouter: {
                        base_url: orBase,
                        api_key: orKey,
                        model: orModel,
                    },
                    aicredits: {
                        base_url: aiBase,
                        api_key: aiKey,
                        model: aiModel,
                    },
                    base_url: activeBase,
                    api_key: activeKey,
                    model: activeModel,
                    context_window: parseInt(inputContext ? inputContext.value : "128000", 10) || 128000,
                    enable_prompt_caching: checkCaching ? checkCaching.checked : true,
                };

                if (modalFeedback) {
                    modalFeedback.textContent = "Saving OpenRouter & AI Credits configurations to .env...";
                    modalFeedback.className = "cyberdeck-modal-feedback";
                }

                try {
                    const res = await fetch("/api/llm/openai/config", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();
                    if (data.status === "ok") {
                        if (modalFeedback) {
                            modalFeedback.textContent = "✓ Configurations saved to .env! Active provider connected.";
                            modalFeedback.className = "cyberdeck-modal-feedback success";
                        }
                        if (payload.model) this.activeOpenaiModel = payload.model;
                        this.updateOpenaiModelDisplays(payload.model);
                        setTimeout(() => {
                            closeModal();
                            this.scanOpenaiModels(true, activeProv);
                        }, 750);
                    } else {
                        if (modalFeedback) {
                            modalFeedback.textContent = `Error: ${data.message || data.error}`;
                            modalFeedback.className = "cyberdeck-modal-feedback error";
                        }
                    }
                } catch (err) {
                    if (modalFeedback) {
                        modalFeedback.textContent = `Network error: ${err.message}`;
                        modalFeedback.className = "cyberdeck-modal-feedback error";
                    }
                }
            });
        }

        if (btnTestScanModal) {
            btnTestScanModal.addEventListener("click", async () => {
                if (modalFeedback) {
                    modalFeedback.textContent = "Querying endpoint /models...";
                    modalFeedback.className = "cyberdeck-modal-feedback";
                }
                const models = await this.scanOpenaiModels();
                if (modalFeedback) {
                    if (models && models.length > 0) {
                        modalFeedback.textContent = `✓ Found ${models.length} models! Config verified.`;
                        modalFeedback.className = "cyberdeck-modal-feedback success";
                    } else {
                        modalFeedback.textContent = "Notice: Could not list models. Check endpoint URL or API key.";
                        modalFeedback.className = "cyberdeck-modal-feedback error";
                    }
                }
            });
        }

        // Fetch initial status on load
        try {
            const res = await fetch("/api/llm/status");
            if (res.ok) {
                const data = await res.json();
                if (data.context_window && data.provider === "local") {
                    this.localContextWindow = data.context_window;
                }
                if (data.openrouter && data.openrouter.api_key) {
                    this.openrouterApiKey = data.openrouter.api_key;
                }
                if (data.aicredits && data.aicredits.api_key) {
                    this.aicreditsApiKey = data.aicredits.api_key;
                }
                const serverModel = data.active_openai_model || (data.openai_compatible && data.openai_compatible.model);
                if (serverModel) {
                    this.activeOpenaiModel = serverModel;
                    try {
                        if (typeof localStorage !== "undefined") {
                            localStorage.setItem("active_openai_model", serverModel);
                        }
                    } catch (e) {}
                }
                this.updateOpenaiModelDisplays(this.activeOpenaiModel);
                this.updateLlmToggleUi(data.provider, data.mode);
                // Background scan of available models for immediate search readiness
                this.scanOpenaiModels(false);
            }
        } catch (e) {}

        /**
         * Handle user swap action, cycling through Local -> Gemini -> OpenAI -> Local.
         * @returns {Promise<void>}
         */
        const handleSwapAction = async () => {
            const curr = this.currentLlmProvider || "local";
            let target = "local";
            if (curr === "local") {
                target = "gemini";
            } else if (curr === "gemini") {
                target = "openai_compatible";
            } else {
                target = "local";
            }

            this.updateLlmToggleUi(curr, null, true, target);
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
                    let msg = "";
                    if (data.provider === "gemini") {
                        msg = "Swapped to Cloud Gemini 3.6 Flash (1M Context)! Local llama-server stopped & 3.4GB VRAM released.";
                    } else if (data.provider === "openai_compatible") {
                        msg = `Swapped to OpenAI-Compatible API (${this.activeOpenaiModel})! Prompt caching active & 3.4GB VRAM released.`;
                        this.scanOpenaiModels();
                    } else {
                        msg = "Swapped to Local Qwen 3.5 4B on RTX 3050! llama-server spun up & slot cache primed.";
                    }
                    this.logEvent("LLM", msg, "success");
                } else {
                    this.updateLlmToggleUi(curr);
                    this.logEvent("ERROR", `Failed to switch LLM provider: ${data.message || data.error}`, "error");
                }
            } catch (err) {
                this.updateLlmToggleUi(curr);
                this.logEvent("ERROR", `Network error switching LLM provider: ${err.message}`, "error");
            }
        };

        if (btn) btn.addEventListener("click", handleSwapAction);
        if (panelBtn) panelBtn.addEventListener("click", handleSwapAction);
        if (chatActiveModelPill) {
            chatActiveModelPill.addEventListener("click", (e) => {
                if (e.target.closest("#chatOpenaiControls")) return;
                handleSwapAction();
            });
        }
    }

    setView(mode) {
        const safeMode = mode === "chat" ? "chat" : "dual";
        document.body.className = `mode-${safeMode}`;
        document.querySelectorAll(".view-btn").forEach(btn => btn.classList.remove("active"));
        if (safeMode === "dual") document.getElementById("btnViewDual")?.classList.add("active");
        if (safeMode === "chat") {
            document.getElementById("btnViewChat")?.classList.add("active");
            this.onEnterChatMode();
        } else {
            this.onExitChatMode();
        }
        // Recalibrate 3D avatar viewport and matrix canvas immediately
        setTimeout(() => {
            if (this.matrixRain && typeof this.matrixRain.resize === "function") {
                this.matrixRain.resize();
            }
            this._forEachEngine((eng) => {
                if (eng.onResize) eng.onResize();
            });
            window.dispatchEvent(new Event("resize"));
        }, 60);
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
            this.faceStage.classList.remove("state-idle", "state-listening", "state-thinking", "state-speaking", "state-standby");
            this.faceStage.classList.add(`state-${normalizedMode}`);
        }

        // Update Status Pill
        if (this.statusPill) this.statusPill.className = `status-pill status-${normalizedMode}`;
        if (this.statusText) this.statusText.textContent = normalizedMode.toUpperCase();

        // Update Face Caption & 3D Character Posture / Expression
        if (this.charStateBadge) this.charStateBadge.textContent = `STATE: ${normalizedMode.toUpperCase()}`;
        if (this.dbgCharState) this.dbgCharState.textContent = normalizedMode.toUpperCase();

        switch (normalizedMode) {
            case "standby":
                if (this.faceCaption) this.faceCaption.textContent = "STANDBY • WAITING FOR WAKE WORD";
                this.stopMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.speakStop();
                    eng.setEmotion("neutral");
                    eng.lookAt(0, 1.25, 2.5);
                });
                break;
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
        if (!this.terminalFeed) return;
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
            this.currentLlmProvider = data.provider;
            this.updateLlmToggleUi(data.provider, data.mode);
            this.syncChatTopBar();
            this.showDynamicIsland("AI Provider Active", `Switched to ${(data.provider || 'local').toUpperCase()}`, "⚡", "cyan");
        }

        if (data.event === "stt_model_changed" || data.event === "stt.model_changed") {
            if (data.model) {
                this.activeSttModelId = data.model;
                if (typeof this.updateSttDisplays === "function") {
                    this.updateSttDisplays(data.model, data.size_mb);
                }
                this.showDynamicIsland("Whisper Model", `Active: ${data.model}`, "🎙️", "emerald", 2500);
            }
        }

        if (data.event === "tts_engine_changed" || data.event === "tts.engine_changed") {
            if (data.engine) {
                this.activeTtsEngine = data.engine;
                this.activeTtsVoice = data.voice || this.activeTtsVoice;
                if (typeof this.updateTtsDisplays === "function") {
                    this.updateTtsDisplays(data.engine, data.voice, data.sample_rate);
                }
                this.showDynamicIsland("TTS Engine", `Active: ${data.engine.toUpperCase()} (${data.voice || ''})`, "🔊", "pink", 2500);
            }
        }

        if (data.event === "llm_model_changed" || data.event === "llm.local_model_changed") {
            if (data.model) {
                this.activeLocalModelId = data.model;
                if (typeof this.updateLocalModelDisplays === "function") {
                    this.updateLocalModelDisplays(data.model);
                }
                this.showDynamicIsland("Local Model", `Active: ${data.model}`, "⚡", "cyan", 2500);
            }
        }

        if (data.event === "wake_word_detected" || data.wake_word || data.event === "wakeword.detected") {
            const w = data.wake_word || "Voice Wake";
            this.showDynamicIsland("Wake Word Active", `Listening for "${w}"...`, "⚡", "cyan", 3500);
            this.setRobotMode("listening");
            this.logEvent("WAKE", `Wake word detected: ${w}`, "system");
        }

        if (data.event === "standby" || data.event === "wakeword.standby" || data.standby || (data.event === "mode_changed" && data.mode === "standby") || (data.event === "mode" && data.mode === "standby")) {
            let persTitle = data.persona_name;
            if (!persTitle && data.persona) {
                const p = String(data.persona).toLowerCase().trim();
                persTitle = (p === "neo" || p === "cypher") ? "Cypher" : (p === "glados" ? "GLaDOS" : p.charAt(0).toUpperCase() + p.slice(1));
            }
            if (!persTitle) {
                persTitle = this.getActivePersonaTitle();
            }
            const sub = data.text ? `"${data.text}" • Say "Hey ${persTitle}" to resume` : `Listening paused • Say "Hey ${persTitle}" to resume`;
            this.showDynamicIsland("Standby Mode Active", sub, "💤", "amber", 4500);
            this.setRobotMode("standby");
            this.logEvent("STANDBY", `Standby triggered (${data.text || 'user command'})`, "system");
        }

        if (data.event === "scheduler.job_triggered" || data.event === "timer.triggered") {
            const lbl = (data.job && data.job.label) ? data.job.label : "Timer finished!";
            this.showDynamicIsland("Timer Alert", lbl, "⏰", "amber", 4500);
        }

        if (data.event === "voice_mode_changed" && this.updateVoiceUI) {
            this.voiceEnabled = Boolean(data.voice_enabled);
            this.updateVoiceUI(this.voiceEnabled);
        }

        if (data.event === "thinking_toggled") {
            this.thinkingEnabled = Boolean(data.enabled);
            localStorage.setItem("robot_thinking_enabled", String(this.thinkingEnabled));
            this.updateThinkingUI(this.thinkingEnabled);
        }

        if (data.event === "thought_started") {
            this.showAvatarThoughtLive("started");
        }

        if (data.event === "thought" && data.token) {
            this.showAvatarThoughtLive("token", data.token);
        }

        if (data.event === "thought_completed") {
            this.showAvatarThoughtLive("completed", data);
        }

        if (data.event === "personality_switched" && data.personality) {
            const chosen = data.personality;
            if (this.personaSelect && this.personaSelect.value !== chosen) {
                this.personaSelect.value = chosen;
            }
            if (this.personaName) {
                this.personaName.textContent = (chosen === "neo" ? "CYPHER" : chosen.toUpperCase());
            }
            this.applyPersonaTheme(chosen);
            const disp = chosen === "neo" ? "CYPHER" : chosen.toUpperCase();
            this.logEvent("PERSONA", `Personality switched to ${disp}`, "system");
            this.showDynamicIsland("Persona Active", `Switched to ${disp}`, "👤", "emerald");
        }

        if (data.event === "personality_warming" && data.personality) {
            const chosen = data.personality;
            const disp = chosen === "neo" ? "CYPHER" : chosen.toUpperCase();
            if (this.bubbleTag) {
                this.bubbleTag.textContent = `${disp}: WARMING UP...`;
                this.bubbleTag.style.color = "var(--accent-amber, #f59e0b)";
            }
            if (this.bubbleContent) {
                this.bubbleContent.textContent = "Priming neural cache & sensory context in background...";
            }
            const cyberBadge = document.getElementById("cyberDeckBadge");
            if (cyberBadge) {
                cyberBadge.textContent = `${disp}: WARMING...`;
                cyberBadge.className = "badge badge-warning";
            }
        }

        if (data.event === "personality_ready" && data.personality) {
            const chosen = data.personality;
            const disp = chosen === "neo" ? "CYPHER" : chosen.toUpperCase();
            const dur = data.duration_ms ? ` (${Math.round(data.duration_ms)}ms)` : "";
            if (this.bubbleTag) {
                this.bubbleTag.textContent = `${disp}: READY`;
                this.bubbleTag.style.color = "var(--accent-emerald, #10b981)";
            }
            if (this.bubbleContent) {
                this.bubbleContent.textContent = "Ready for voice or text input.";
            }
            const cyberBadge = document.getElementById("cyberDeckBadge");
            if (cyberBadge) {
                cyberBadge.textContent = "STUDIO: READY";
                cyberBadge.className = "badge badge-accent";
            }
            this.logEvent("READY", `${disp} ready for input${dur}`, "system");
        }

        if (data.event === "pipeline_updated" || data.pipeline) {
            this.updatePipelineStatus(data.pipeline);
        }

        if (data.mode) {
            this.setRobotMode(data.mode);
        }

        if (data.partial_transcript) {
            if (this.bubbleTag) {
                this.bubbleTag.textContent = "LISTENING (STREAMING)...";
                this.bubbleTag.style.color = "var(--primary-cyan)";
            }
            if (this.bubbleContent) {
                this.bubbleContent.textContent = data.partial_transcript + " ...";
            }
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
            this._forEachEngine((eng) => {
                eng.speakStop();
                eng.setEmotion("neutral");
            });
            const targetMode = data.mode || (this.currentMode === "standby" ? "standby" : "idle");
            this.setRobotMode(targetMode);
            this.updatePipelineStatus({ TTS: "COMPLETED", AVATAR: targetMode.toUpperCase() });
        }

        if (data.robot_phrase) {
            const cleanPhrase = this.cleanHumanText(data.robot_phrase) || data.robot_phrase;
            if (data.emotion) {
                this._forEachEngine((eng) => eng.setEmotion(data.emotion));
            }
            if (this.bubbleTag) {
                const emoLabel = data.emotion && data.emotion !== "neutral" ? ` • ${data.emotion.toUpperCase()}` : "";
                this.bubbleTag.textContent = `ROBOT RESPONSE${emoLabel}`;
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
                this._forEachEngine((eng) => {
                    eng.speakStop();
                    eng.setEmotion("neutral");
                });
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
                badge.textContent = `COORD: ${st}`;
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
            const charBadge = document.getElementById("charToolBadge");
            const toolNameStr = (data.tool || "RUNNING").toUpperCase();
            if (toolBadge) {
                toolBadge.textContent = `TOOL: ${toolNameStr}`;
                toolBadge.style.display = "inline-block";
                toolBadge.style.background = "rgba(0, 242, 254, 0.2)";
                toolBadge.style.borderColor = "var(--primary-cyan)";
            }
            if (charBadge) {
                charBadge.textContent = `🛠️ TOOL: ${toolNameStr}`;
                charBadge.style.display = "inline-block";
                charBadge.style.borderColor = "var(--primary-cyan)";
                charBadge.style.color = "#00f2fe";
            }
            const elToolName = document.getElementById("activeToolName");
            if (elToolName) elToolName.textContent = toolNameStr;
            const elToolStatus = document.getElementById("lastToolStatus");
            if (elToolStatus) {
                elToolStatus.textContent = "RUNNING";
                elToolStatus.style.color = "var(--primary-cyan)";
            }
            this.logEvent("TOOL", `Executing: ${data.tool}`, "system");
            this.updatePipelineStatus({ TOOLS: "RUNNING" });
        }

        if (data.event === "tool_completed" || data.event === "tool_error") {
            const toolBadge = document.getElementById("activeToolBadge");
            const charBadge = document.getElementById("charToolBadge");
            const isOk = data.event === "tool_completed";
            const toolNameStr = (data.tool || "TOOL").toUpperCase();
            if (toolBadge) {
                if (isOk) {
                    toolBadge.textContent = `TOOL: DONE (${toolNameStr})`;
                    toolBadge.style.background = "rgba(0, 245, 160, 0.2)";
                    toolBadge.style.borderColor = "var(--accent-emerald)";
                } else {
                    toolBadge.textContent = `TOOL: ERR (${toolNameStr})`;
                    toolBadge.style.background = "rgba(255, 87, 87, 0.2)";
                    toolBadge.style.borderColor = "#ff5757";
                }
                setTimeout(() => {
                    if (toolBadge.textContent.includes("DONE") || toolBadge.textContent.includes("ERR")) {
                        toolBadge.style.display = "none";
                    }
                }, 4000);
            }
            if (charBadge) {
                if (isOk) {
                    charBadge.textContent = `🛠️ TOOL: ${toolNameStr} (DONE)`;
                    charBadge.style.borderColor = "var(--accent-emerald)";
                    charBadge.style.color = "var(--accent-emerald)";
                } else {
                    charBadge.textContent = `🛠️ TOOL: ${toolNameStr} (ERR)`;
                    charBadge.style.borderColor = "#ff5757";
                    charBadge.style.color = "#ff5757";
                }
                setTimeout(() => {
                    if (charBadge.textContent.includes("DONE") || charBadge.textContent.includes("ERR")) {
                        charBadge.style.display = "none";
                    }
                }, 4000);
            }
            const elToolStatus = document.getElementById("lastToolStatus");
            if (elToolStatus) {
                elToolStatus.textContent = isOk ? "SUCCESS" : "ERROR";
                elToolStatus.style.color = isOk ? "var(--matrix-bright)" : "#ff4d6d";
            }
            const elToolLat = document.getElementById("lastToolLatency");
            if (elToolLat && data.duration_ms) {
                elToolLat.textContent = `${Math.round(data.duration_ms)} ms`;
            }
            this._toolInvocations = (this._toolInvocations || 0) + 1;
            const elTotalRuns = document.getElementById("totalToolInvocations");
            if (elTotalRuns) elTotalRuns.textContent = this._toolInvocations;
            this.updatePipelineStatus({ TOOLS: data.event === "tool_completed" ? "COMPLETED" : "ERROR" });

            if (isOk) {
                const latStr = data.duration_ms ? ` (${Math.round(data.duration_ms)}ms)` : "";
                this.showDynamicIsland("Tool Succeeded", `${toolNameStr}${latStr}`, "🛠️", "emerald", 2800);
            } else {
                this.showDynamicIsland("Tool Failed", `Failed: ${toolNameStr}`, "⚠️", "rose", 3500);
            }
        }

        if (data.metrics) {
            this.updateMetrics(data.metrics);
        }

        if (data.event === "observability.event") {
            this.handleObservabilityEvent(data);
        }
    }

    /**
     * Update telemetry metrics in the CyberDeck HUD indicators and progress bars.
     * @param {Object} metrics - Latency and throughput metrics object.
     * @param {number} [metrics.vad] - Voice Activity Detection latency in milliseconds.
     * @param {number} [metrics.stt] - Speech-to-Text latency in milliseconds.
     * @param {number} [metrics.context] - Context assembly latency in milliseconds.
     * @param {number} [metrics.ttft] - Time to first token in milliseconds.
     * @param {number} [metrics.synth] - Speech synthesis latency in milliseconds.
     * @param {number} [metrics.e2e] - End-to-end turn latency in milliseconds.
     * @param {number} [metrics.tokens] - Total tokens generated.
     * @param {number} [metrics.tokens_in] - Input prompt tokens.
     * @param {number} [metrics.tokens_out] - Output completion tokens.
     * @returns {void}
     */
    updateMetrics(metrics) {
        if (!metrics) return;
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

    /**
     * Backward-compatible alias for updateMetrics to ensure any legacy callers or
     * cached client sessions do not throw "this.updateTelemetry is not a function".
     * @param {Object} metrics - Telemetry metrics object.
     * @returns {void}
     */
    updateTelemetry(metrics) {
        return this.updateMetrics(metrics);
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
                    if (data.voice_enabled !== undefined && this.updateVoiceUI && this.voiceEnabled !== data.voice_enabled) {
                        this.updateVoiceUI(data.voice_enabled);
                    }
                    if (data.personality) {
                        const pUpper = this.formatPersonaName(data.personality);
                        if (this.personaName) this.personaName.textContent = pUpper;
                        if (this.personaSelect && document.activeElement !== this.personaSelect && this.personaSelect.value !== data.personality) {
                            this.personaSelect.value = data.personality;
                            this.syncChatTopBar();
                            document.querySelectorAll(".msg-assistant .chat-msg-sender").forEach((el) => {
                                el.textContent = pUpper;
                            });
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

                        const sLight = document.getElementById("sensorLightVal");
                        if (sLight) sLight.textContent = (ws.lighting || "NORMAL").toUpperCase();
                        const sMotion = document.getElementById("sensorMotionVal");
                        if (sMotion) sMotion.textContent = (ws.motion || "QUIET").toUpperCase();
                        const sPeople = document.getElementById("sensorPeopleVal");
                        if (sPeople) sPeople.textContent = `${ws.people_count || 0} IN VIEW`;
                        const sObjects = document.getElementById("sensorObjectsVal");
                        if (sObjects) sObjects.textContent = `${ws.objects_count || 0} TRACKED`;

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

        // Cyber-Studio Workstation Tabs & Sensor Card Toggle Setup
        const tabBtns = document.querySelectorAll(".deck-tab-btn");
        tabBtns.forEach((btn) => {
            btn.addEventListener("click", () => {
                tabBtns.forEach((b) => b.classList.remove("active"));
                btn.classList.add("active");
                const targetId = btn.getAttribute("data-target");
                document.querySelectorAll(".deck-pane").forEach((pane) => {
                    pane.style.display = pane.id === targetId ? "flex" : "none";
                });
                if (targetId === "paneMemoryDeck") {
                    if (typeof this.refreshMemoryDeck === "function") {
                        this.refreshMemoryDeck();
                    }
                    if (window.innerWidth <= 768) {
                        setTimeout(() => {
                            const searchEl = document.getElementById("deckMemSearchInput") || document.getElementById("paneMemoryDeck");
                            if (searchEl) {
                                searchEl.scrollIntoView({ behavior: "smooth", block: "center" });
                            }
                        }, 100);
                    }
                }
            });
        });

        const btnToggleRaw = document.getElementById("btnToggleRawState");
        if (btnToggleRaw) {
            btnToggleRaw.addEventListener("click", () => {
                const grid = document.getElementById("sensorCardsGrid");
                const rawPre = document.getElementById("worldStateCode");
                if (grid && rawPre) {
                    const isRaw = rawPre.style.display !== "none";
                    rawPre.style.display = isRaw ? "none" : "block";
                    grid.style.display = isRaw ? "grid" : "none";
                    btnToggleRaw.textContent = isRaw ? "RAW JSON" : "SENSORS";
                }
            });
        }

        // Camera Stream Auto-Recovery: /camera/stream provides continuous 60 FPS MJPEG.
        // If the stream drops or hardware is offline, display high-tech radar standby.
        if (this.cameraImg) {
            this.cameraImg.onerror = () => {
                this.cameraImg.style.display = "none";
                const radar = document.getElementById("radarStandby");
                if (radar) radar.style.display = "flex";
                setTimeout(() => {
                    this.cameraImg.src = `/camera/stream?t=${Date.now()}`;
                }, 3000);
            };
            this.cameraImg.onload = () => {
                this.cameraImg.style.display = "block";
                const radar = document.getElementById("radarStandby");
                if (radar) radar.style.display = "none";
            };
        }

        // Hardware Observability Telemetry Polling (CPU, System RAM, RTX 3050, AMD 680M)
        const fetchHw = async () => {
            try {
                const res = await fetch("/observability/hardware");
                if (res.ok) {
                    const hw = await res.json();
                    if (hw.cpu_percent !== undefined) {
                        const el = document.getElementById("hwCpuVal");
                        if (el) el.textContent = `${hw.cpu_percent.toFixed(1)} %`;
                        const bar = document.getElementById("hwCpuBar");
                        if (bar) bar.style.width = `${Math.min(100, Math.max(5, hw.cpu_percent))}%`;
                    }
                    if (hw.ram_used_gb !== undefined) {
                        const total = hw.ram_total_gb ? ` / ${hw.ram_total_gb.toFixed(1)}` : "";
                        const el = document.getElementById("hwRamVal");
                        if (el) el.textContent = `${hw.ram_used_gb.toFixed(1)}${total} GB`;
                        const bar = document.getElementById("hwRamBar");
                        if (bar && hw.ram_total_gb) {
                            bar.style.width = `${Math.min(100, (hw.ram_used_gb / hw.ram_total_gb) * 100)}%`;
                        }
                    }
                    if (hw.nvidia) {
                        if (hw.nvidia.vram_used_mb !== undefined) {
                            const el = document.getElementById("hwVramVal");
                            if (el) el.textContent = `${Math.round(hw.nvidia.vram_used_mb)} MB`;
                            const bar = document.getElementById("hwVramBar");
                            if (bar && hw.nvidia.vram_total_mb) {
                                bar.style.width = `${Math.min(100, (hw.nvidia.vram_used_mb / hw.nvidia.vram_total_mb) * 100)}%`;
                            }
                        }
                        if (hw.nvidia.temperature_c !== undefined) {
                            const el = document.getElementById("hwTempVal");
                            if (el) el.textContent = `${Math.round(hw.nvidia.temperature_c)} °C`;
                            const bar = document.getElementById("hwTempBar");
                            if (bar) bar.style.width = `${Math.min(100, (hw.nvidia.temperature_c / 90) * 100)}%`;
                        }
                    }
                    if (hw.amd) {
                        const el = document.getElementById("hwAmdVal");
                        const val = hw.amd.utilization_percent !== undefined ? hw.amd.utilization_percent : (hw.amd_utilization || 5);
                        if (el) el.textContent = `${Math.round(val)} %`;
                        const bar = document.getElementById("hwAmdBar");
                        if (bar) bar.style.width = `${Math.min(100, Math.max(8, val))}%`;
                    }
                }
            } catch (e) {}
        };
        fetchHw();
        setInterval(fetchHw, 2000);

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
        const traceBadge = document.getElementById("activeTraceBadge");
        if (trace && trace.trace_id && traceBadge) {
            traceBadge.textContent = `TRACE: ${trace.trace_id.substring(0,8)}`;
        }
        const box = document.getElementById("traceTreeBox");
        if (!box) return;
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

    /**
     * Initialize Asynchronous Neural Image Studio UI controls, model catalog, and job listeners.
     * @returns {void}
     */
    initImageGen() {
        const btnGen = document.getElementById("btnGenImage");
        const btnCancel = document.getElementById("btnCancelImage");
        const input = document.getElementById("imgPromptInput");
        const btnToggle = document.getElementById("btnToggleImageGen");
        const drawer = document.getElementById("cardImageGen");
        const modelSelect = document.getElementById("imgModelSelect");
        const stepsInput = document.getElementById("imgStepsInput");

        if (btnToggle && drawer) {
            btnToggle.addEventListener("click", (e) => {
                e.stopPropagation();
                drawer.classList.toggle("collapsed");
                btnToggle.textContent = drawer.classList.contains("collapsed") ? "+" : "−";
            });
        }

        // Auto-tune steps and resolution when model selection changes
        if (modelSelect && stepsInput) {
            modelSelect.addEventListener("change", () => {
                const sel = modelSelect.value;
                if (sel === "sd-turbo" || sel === "sdxl-turbo" || sel === "flux-schnell" || sel === "fast-procedural") {
                    stepsInput.value = "4";
                } else if (sel === "dreamshaper-v8" || sel === "sd-1.5") {
                    stepsInput.value = "20";
                } else if (sel === "flux-dev") {
                    stepsInput.value = "24";
                }
            });

            // Fetch live model catalog and cached states from backend
            fetch("/api/images/models")
                .then(r => r.json())
                .then(data => {
                    if (data && data.models) {
                        data.models.forEach(m => {
                            const opt = modelSelect.querySelector(`option[value="${m.id}"]`);
                            if (opt && m.is_cached) {
                                if (!opt.textContent.includes("✓")) {
                                    opt.textContent += " ✓ [Cached]";
                                }
                            }
                        });
                    }
                })
                .catch(() => {});
        }

        /**
         * Submit an asynchronous text-to-image synthesis job to the unified diffusion pipeline.
         * @returns {Promise<void>}
         */
        const submitImage = async () => {
            if (!input) return;
            const prompt = input.value.trim();
            if (!prompt) return;

            const resSelect = document.getElementById("imgResSelect");
            const [width, height] = (resSelect ? resSelect.value : "512x512").split("x").map(Number);
            const steps = stepsInput ? (parseInt(stepsInput.value, 10) || 4) : 4;
            const targetModel = modelSelect ? modelSelect.value : "sd-turbo";

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
                    body: JSON.stringify({
                        prompt: prompt,
                        model: targetModel,
                        width: width,
                        height: height,
                        steps: steps
                    })
                });
                const data = await res.json();
                if (res.ok && !data.error) {
                    input.value = "";
                    this.logEvent("IMAGE", `Job queued on ${targetModel}: "${prompt}" (${width}x${height}, ${steps} st)`, "system");
                    try {
                        const pollRes = await fetch("/images/jobs");
                        if (pollRes.ok) {
                            const jobsData = await pollRes.json();
                            if (jobsData.jobs) this.updateImageJobsUI(jobsData.jobs);
                        }
                    } catch (e) {}
                } else {
                    const err = data.error || (data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
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

        // Quick style presets
        document.querySelectorAll(".img-style-chip").forEach((chip) => {
            chip.addEventListener("click", (e) => {
                e.preventDefault();
                document.querySelectorAll(".img-style-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                const stylePrompt = chip.getAttribute("data-style");
                if (stylePrompt && input) {
                    input.value = stylePrompt;
                    input.focus();
                }
            });
        });

        const previewImg = document.getElementById("imgPreviewImg");
        if (previewImg) {
            previewImg.style.cursor = "zoom-in";
            previewImg.addEventListener("click", () => {
                if (previewImg.src) {
                    window.open(previewImg.src, "_blank");
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
                    const mModel = document.getElementById("metaModel");
                    if (mModel) mModel.textContent = completedJob.model || '--';
                    const mBackend = document.getElementById("metaBackend");
                    if (mBackend) mBackend.textContent = completedJob.backend || '--';
                    const mDevice = document.getElementById("metaDevice");
                    if (mDevice) mDevice.textContent = completedJob.device || '--';
                    const mTime = document.getElementById("metaTime");
                    if (mTime) mTime.textContent = `${(completedJob.duration_s || 0).toFixed(2)}s`;
                    const mRes = document.getElementById("metaRes");
                    if (mRes) mRes.textContent = `${completedJob.width}x${completedJob.height}`;
                    const mSteps = document.getElementById("metaSteps");
                    if (mSteps) mSteps.textContent = completedJob.steps || '--';
                }
                if (previewImg && !previewImg._boundClick) {
                    previewImg._boundClick = true;
                    previewImg.addEventListener("click", () => {
                        if (previewImg.src && !previewImg.src.endsWith("#")) {
                            window.open(previewImg.src, "_blank");
                        }
                    });
                }
            } else {
                if (placeholder) placeholder.style.display = "block";
                if (previewImg) previewImg.style.display = "none";
                if (metaBlock) metaBlock.style.display = "none";
            }
        }
    }

    initAudioDeck() {
        const spectrumCanvas = document.getElementById("audioSpectrumCanvas");
        const presetBadge = document.getElementById("audioPresetBadge");
        const playbackBadge = document.getElementById("audioPlaybackBadge");
        const eqChips = document.querySelectorAll("#eqPresetChips .eq-chip");
        const trackTitle = document.getElementById("trackTitleDisplay");
        const volSlider = document.getElementById("musicVolSlider");
        const volVal = document.getElementById("musicVolVal");
        const btnRestart = document.getElementById("btnRestartMusic");
        const btnPlay = document.getElementById("btnTogglePlayback");
        const chainBtns = document.querySelectorAll(".chain-btn");

        // 1. Equalizer preset switcher
        eqChips.forEach(chip => {
            chip.addEventListener("click", async () => {
                const preset = chip.dataset.preset;
                eqChips.forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                if (presetBadge) presetBadge.textContent = `EQ: ${preset.toUpperCase()}`;

                try {
                    const res = await fetch("/api/equalizer", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ preset: preset })
                    });
                    const data = await res.json();
                    if (data && data.status === "ok") {
                        if (presetBadge) presetBadge.textContent = `EQ: ${data.preset.toUpperCase()}`;
                    }
                } catch (err) {
                    console.error("Failed to switch equalizer preset:", err);
                }
            });
        });

        // 2. Fetch initial music & equalizer state
        const refreshMusicState = async () => {
            try {
                const res = await fetch("/api/music/status");
                const data = await res.json();
                if (data && data.status) {
                    const st = data.status;
                    if (trackTitle && st.current_track && st.current_track !== "None") {
                        trackTitle.textContent = st.current_track;
                    }
                    if (playbackBadge) {
                        playbackBadge.textContent = st.is_playing ? "STATUS: PLAYING" : (st.is_paused ? "STATUS: PAUSED" : "STATUS: READY");
                        playbackBadge.classList.toggle("badge-accent", !!st.is_playing);
                    }
                    if (btnPlay) {
                        btnPlay.textContent = st.is_playing ? "⏸ Pause" : "⏯ Play";
                    }
                    if (st.volume !== undefined && volSlider && !volSlider.matches(':active')) {
                        const pct = Math.round(st.volume * 100);
                        volSlider.value = pct;
                        if (volVal) volVal.textContent = `${pct}%`;
                    }
                }
                if (data && data.equalizer && data.equalizer.active_preset) {
                    const act = data.equalizer.active_preset;
                    if (presetBadge) presetBadge.textContent = `EQ: ${act.toUpperCase()}`;
                    eqChips.forEach(c => {
                        c.classList.toggle("active", c.dataset.preset === act);
                    });
                }
            } catch (err) {
                // Ignore transient errors
            }
        };
        refreshMusicState();
        setInterval(refreshMusicState, 5000);

        // 3. Volume slider (Direct REST endpoint)
        if (volSlider) {
            volSlider.addEventListener("input", () => {
                if (volVal) volVal.textContent = `${volSlider.value}%`;
            });
            volSlider.addEventListener("change", async () => {
                const volPct = parseInt(volSlider.value, 10);
                if (volVal) volVal.textContent = `${volPct}%`;
                try {
                    await fetch("/api/music/volume", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ volume: volPct / 100.0 })
                    });
                } catch (err) {
                    console.error("Volume set error:", err);
                }
            });
        }

        // 4. Play / Pause & Restart buttons (Direct REST endpoints with immediate visual feedback)
        if (btnRestart) {
            btnRestart.addEventListener("click", async () => {
                btnRestart.classList.add("executing");
                setTimeout(() => btnRestart.classList.remove("executing"), 400);
                try {
                    const res = await fetch("/api/music/restart", { method: "POST" });
                    const data = await res.json();
                    if (data && data.status === "ok") {
                        refreshMusicState();
                    }
                } catch (err) {
                    console.error("Restart music error:", err);
                }
            });
        }

        if (btnPlay) {
            btnPlay.addEventListener("click", async () => {
                btnPlay.classList.add("executing");
                setTimeout(() => btnPlay.classList.remove("executing"), 400);
                try {
                    const res = await fetch("/api/music/toggle", { method: "POST" });
                    const data = await res.json();
                    if (data) {
                        btnPlay.textContent = data.is_playing ? "⏸ Pause" : "⏯ Play";
                        refreshMusicState();
                    }
                } catch (err) {
                    console.error("Toggle music playback error:", err);
                }
            });
        }

        // 5. Chained macro buttons (Works in both Main HUD and Chat Workspace modes)
        const triggerMacro = (prompt, btnElem) => {
            if (!prompt) return;
            if (btnElem) {
                btnElem.classList.add("executing");
                setTimeout(() => btnElem.classList.remove("executing"), 600);
            }

            const dualInput = document.getElementById("chatInput");
            const dualSendBtn = document.getElementById("chatSendBtn");
            const wsInput = document.getElementById("chatWsTextarea");
            const wsSendBtn = document.getElementById("chatWsSendBtn");

            if (document.body.classList.contains("mode-chat") && wsInput && wsSendBtn) {
                wsInput.value = prompt;
                wsSendBtn.click();
            } else if (dualInput && dualSendBtn) {
                dualInput.value = prompt;
                dualSendBtn.click();
            }
        };

        const macroButtons = document.querySelectorAll(".chain-btn, .macro-btn");
        macroButtons.forEach(btn => {
            btn.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                triggerMacro(btn.dataset.prompt, btn);
            });
        });

        // 6. Cybernetic Audio Spectrum Visualizer Animation
        if (spectrumCanvas) {
            const ctx = spectrumCanvas.getContext("2d");
            const numBars = 32;
            const barHeights = new Float32Array(numBars);
            const barTargets = new Float32Array(numBars);

            const animateSpectrum = () => {
                const w = spectrumCanvas.width = spectrumCanvas.clientWidth || 280;
                const h = spectrumCanvas.height = 38;
                ctx.clearRect(0, 0, w, h);

                const isSpeaking = this.currentMode === "speaking" || (playbackBadge && playbackBadge.textContent.includes("PLAYING"));
                const barWidth = Math.max(3, (w - (numBars - 1) * 3) / numBars);

                for (let i = 0; i < numBars; i++) {
                    if (isSpeaking) {
                        if (Math.random() < 0.3) {
                            barTargets[i] = Math.random() * (h * 0.85) + 5;
                        }
                    } else {
                        // Ambient cybernetic breathing pulse
                        barTargets[i] = Math.sin(Date.now() * 0.003 + i * 0.3) * 4 + 6;
                    }
                    barHeights[i] += (barTargets[i] - barHeights[i]) * 0.25;

                    const barH = Math.max(3, Math.min(h, barHeights[i]));
                    const x = i * (barWidth + 3);
                    const y = h - barH;

                    const grad = ctx.createLinearGradient(0, h, 0, y);
                    grad.addColorStop(0, "rgba(0, 242, 254, 0.65)");
                    grad.addColorStop(1, "rgba(0, 255, 102, 0.95)");

                    ctx.fillStyle = grad;
                    ctx.fillRect(x, y, barWidth, barH);

                    ctx.fillStyle = "#55ff99";
                    ctx.fillRect(x, y, barWidth, 1.5);
                }

                requestAnimationFrame(animateSpectrum);
            };
            requestAnimationFrame(animateSpectrum);
        }
    }

    /**
     * Initialize Memory Deck Explorer (Mem0 persistent facts & conversation search)
     */
    initMemoryDeck() {
        const memCount = document.getElementById("deckMemCount");
        const chatTurns = document.getElementById("deckChatTurns");
        const searchInput = document.getElementById("deckMemSearchInput");
        const searchBtn = document.getElementById("btnDeckMemSearch");
        const resultsContainer = document.getElementById("deckMemResults");
        const filterChips = document.querySelectorAll("#memFilterChips .mem-chip");

        // 1. Fetch live memory & conversation turn stats
        const loadMemoryStats = async () => {
            try {
                const res = await fetch("/api/memory/stats");
                const data = await res.json();
                if (data && data.status === "ok") {
                    if (memCount) memCount.textContent = data.facts_count ?? 0;
                    if (chatTurns) chatTurns.textContent = data.messages_count ?? 0;
                }
            } catch (e) {
                console.error("Failed to load memory stats:", e);
            }
        };
        this.refreshMemoryDeck = loadMemoryStats;
        loadMemoryStats();

        // 2. Semantic search execution
        const performSearch = async (query = "") => {
            const q = (query !== undefined && query !== null && query !== "" ? query : (searchInput ? searchInput.value : "")).trim();
            if (!resultsContainer) return;

            resultsContainer.innerHTML = `<div class="mem-empty-hint"><span class="tool-pulse-dot"></span> Searching persistent memory graph...</div>`;

            try {
                const searchUrl = q ? `/memory/search?query=${encodeURIComponent(q)}&limit=15` : `/memory/search?query=user&limit=15`;
                const [memRes, histRes] = await Promise.allSettled([
                    fetch(searchUrl).then(r => r.json()),
                    q ? fetch(`/history/search?query=${encodeURIComponent(q)}&limit=6`).then(r => r.json()) : Promise.resolve({ results: [] })
                ]);

                const memData = memRes.status === "fulfilled" ? (memRes.value.results || []) : [];
                const histData = histRes.status === "fulfilled" ? (histRes.value.results || []) : [];

                if (memData.length === 0 && histData.length === 0) {
                    resultsContainer.innerHTML = `
                        <div class="mem-empty-hint">
                            No stored memories matching "<b>${this.escapeHtml(q || 'all')}</b>" found. Speak or chat with the robot to establish persistent facts.
                        </div>`;
                    return;
                }

                let html = "";
                // Render Mem0 facts
                memData.forEach((item, idx) => {
                    const text = typeof item === "string" ? item : (item.memory || item.fact || item.text || JSON.stringify(item));
                    html += `
                        <div class="mem-result-item">
                            <div class="mem-result-tag">🧠 MEM0 FACT #${idx + 1}</div>
                            <div class="mem-result-text">${this.escapeHtml(text)}</div>
                        </div>`;
                });

                // Render conversation history
                histData.forEach((item) => {
                    const content = item.content || item.text || "";
                    const role = (item.role || "turn").toUpperCase();
                    html += `
                        <div class="mem-result-item" style="border-left: 2px solid var(--primary-cyan);">
                            <div class="mem-result-tag">💬 CONVERSATION (${role})</div>
                            <div class="mem-result-text">${this.escapeHtml(content)}</div>
                        </div>`;
                });

                resultsContainer.innerHTML = html;
                if (window.innerWidth <= 768) {
                    setTimeout(() => {
                        resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
                    }, 80);
                }
            } catch (err) {
                console.error("Memory search failed:", err);
                resultsContainer.innerHTML = `<div class="mem-empty-hint" style="color: #ff4444;">Search failed: ${this.escapeHtml(err.message)}</div>`;
            }
        };

        if (searchBtn) {
            searchBtn.addEventListener("click", () => performSearch());
        }

        if (searchInput) {
            searchInput.addEventListener("focus", () => {
                if (window.innerWidth <= 768) {
                    setTimeout(() => {
                        searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
                    }, 120);
                }
            });
            searchInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    performSearch();
                }
            });
        }

        filterChips.forEach(chip => {
            chip.addEventListener("click", () => {
                filterChips.forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                const q = chip.dataset.query || "";
                if (searchInput) searchInput.value = q;
                performSearch(q);
                if (window.innerWidth <= 768 && searchInput) {
                    setTimeout(() => {
                        searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
                    }, 100);
                }
            });
        });
    }

    /**
     * Initialize Acoustic Waveform Oscilloscope & KV-Cache Gauges in card-metrics (Eliminates empty space)
     */
    initPipelineTelemetryDeck() {
        const canvas = document.getElementById("acousticWaveformCanvas");
        const noiseFloorElem = document.getElementById("acousticNoiseFloor");
        const peakChip = document.getElementById("vadPeakChip");
        const vadChip = document.getElementById("vadStatusChip");
        const tokSpeedElem = document.getElementById("activeTokSpeed");
        const budgetBar = document.getElementById("contextBudgetBar");
        const budgetRatio = document.getElementById("contextTokenRatio");

        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        let phase = 0;

        const renderWaveform = () => {
            const w = canvas.width = canvas.clientWidth || 220;
            const h = canvas.height = 36;
            ctx.clearRect(0, 0, w, h);

            const isSpeaking = this.currentMode === "speaking" || this.currentMode === "listening";
            const amplitude = isSpeaking ? (h * 0.38) : (h * 0.12);
            const freq = isSpeaking ? 0.08 : 0.03;
            phase += isSpeaking ? 0.14 : 0.04;

            ctx.beginPath();
            ctx.moveTo(0, h / 2);

            for (let x = 0; x < w; x++) {
                const env = Math.sin((x / w) * Math.PI); // Window envelope
                const y = (h / 2) + Math.sin(x * freq + phase) * amplitude * env + (Math.random() - 0.5) * (isSpeaking ? 3 : 1);
                ctx.lineTo(x, y);
            }

            ctx.strokeStyle = isSpeaking ? "rgba(0, 255, 102, 0.95)" : "rgba(0, 242, 254, 0.55)";
            ctx.lineWidth = 1.5;
            ctx.shadowBlur = isSpeaking ? 8 : 4;
            ctx.shadowColor = isSpeaking ? "#00ff66" : "#00f2fe";
            ctx.stroke();

            // Periodically refresh telemetry chips
            if (Math.random() < 0.04) {
                if (isSpeaking) {
                    if (noiseFloorElem) noiseFloorElem.textContent = "FLOOR: -34 dB";
                    if (peakChip) peakChip.textContent = "PEAK: -6 dB";
                    if (vadChip) {
                        vadChip.textContent = "VAD: ACTIVE SPEECH";
                        vadChip.style.color = "var(--matrix-bright)";
                    }
                } else {
                    if (noiseFloorElem) noiseFloorElem.textContent = "FLOOR: -46 dB";
                    if (peakChip) peakChip.textContent = "PEAK: -14 dB";
                    if (vadChip) {
                        vadChip.textContent = "VAD: ARMED (150ms)";
                        vadChip.style.color = "var(--text-dim)";
                    }
                }
            }

            requestAnimationFrame(renderWaveform);
        };
        requestAnimationFrame(renderWaveform);
    }

    /**
     * Initialize Local LLM Model switcher modal, top nav trigger, and chat pill.
     * Allows hot-swapping local GGUF models on the RTX 3050 CUDA GPU.
     * @returns {Promise<void>}
     */
    async initLocalLlmModelControls() {
        const topNavPicker = document.getElementById("btnTopNavLocalPicker");
        const topNavLabel = document.getElementById("topNavLocalPickerLabel");
        const chatPicker = document.getElementById("btnChatLocalPicker");
        const chatLabel = document.getElementById("chatLocalPickerLabel");
        const modal = document.getElementById("localModelModal");
        const btnClose = document.getElementById("btnCloseLocalModelModal");
        const btnCancel = document.getElementById("btnCancelLocalModelModal");
        const cardsGrid = document.getElementById("localModelCardsGrid");
        const modalHint = document.getElementById("localModelModalHint");

        this.activeLocalModelId = "qwen3.5-4b";
        this.localModelsList = [];

        /**
         * Open Local Model Modal and render current cards.
         * @returns {Promise<void>}
         */
        this.openLocalModelModal = async () => {
            if (!modal) return;
            modal.style.display = "flex";
            modal.classList.add("open");
            await this.refreshLocalModels();
        };

        /**
         * Close Local Model Modal.
         * @returns {void}
         */
        this.closeLocalModelModal = () => {
            if (!modal) return;
            modal.style.display = "none";
            modal.classList.remove("open");
        };

        if (topNavPicker) topNavPicker.addEventListener("click", () => this.openLocalModelModal());
        if (chatPicker) chatPicker.addEventListener("click", () => this.openLocalModelModal());
        if (btnClose) btnClose.addEventListener("click", () => this.closeLocalModelModal());
        if (btnCancel) btnCancel.addEventListener("click", () => this.closeLocalModelModal());
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) this.closeLocalModelModal();
            });
        }

        /**
         * Fetch registered local models and render cards.
         * @returns {Promise<void>}
         */
        this.refreshLocalModels = async () => {
            if (!cardsGrid) return;
            try {
                const res = await fetch("/api/llm/models");
                if (!res.ok) return;
                const data = await res.json();
                this.localModelsList = data.local_models || [];
                if (data.active_local_model) {
                    this.activeLocalModelId = data.active_local_model;
                    this.updateLocalModelDisplays(this.activeLocalModelId);
                }
                this.renderLocalModelCards();
            } catch (err) {
                console.error("Failed to fetch local LLM models:", err);
            }
        };

        /**
         * Render model cards inside the local model selector modal.
         * @returns {void}
         */
        this.renderLocalModelCards = () => {
            if (!cardsGrid) return;
            cardsGrid.innerHTML = "";
            this.localModelsList.forEach(m => {
                const isActive = m.id === this.activeLocalModelId;
                const card = document.createElement("div");
                card.className = `model-profile-card ${isActive ? "active" : ""}`;
                
                const badgeText = m.is_vision ? "VISION VLM" : (m.is_uncensored ? "UNCENSORED" : "INSTRUCT");
                const vramEst = m.parameters === "4B" ? "~3.4 GB" : (m.parameters === "2B" ? "~2.3 GB" : (m.parameters === "0.5B" ? "~0.8 GB" : "~1.2 GB"));

                card.innerHTML = `
                    <div class="model-card-header">
                        <span class="model-card-title">${m.name || m.id}</span>
                        <span class="model-card-badge">${badgeText}</span>
                    </div>
                    <div class="model-card-desc">${m.description || "GGUF quantized local model running on RTX 3050 CUDA."}</div>
                    <div class="model-card-meta">
                        <span class="meta-pill">Params: ${m.parameters || "N/A"}</span>
                        <span class="meta-pill">VRAM: ${vramEst}</span>
                        <span class="meta-pill">Context: 16,384</span>
                        <span class="meta-pill">FlashAttn: Yes</span>
                    </div>
                    <div class="model-card-footer">
                        <button class="card-select-btn ${isActive ? "active-btn" : ""}" type="button">
                            ${isActive ? "✓ ACTIVE LOCAL MODEL" : "LOAD THIS MODEL"}
                        </button>
                    </div>
                `;

                if (!isActive) {
                    card.addEventListener("click", () => this.switchLocalModel(m.id));
                }
                cardsGrid.appendChild(card);
            });
        };

        /**
         * Switch active local LLM model via REST API.
         * @param {string} modelId - Model ID to load.
         * @returns {Promise<void>}
         */
        this.switchLocalModel = async (modelId) => {
            if (modalHint) modalHint.textContent = `Hot-swapping to ${modelId}... RTX 3050 VRAM loading...`;
            this.showDynamicIsland("Local Model Switch", `Loading ${modelId} on RTX 3050...`, "⚡", "amber", 4000);
            this.logEvent("LLM", `Hot-swapping local model to ${modelId}...`, "system");

            try {
                const res = await fetch("/api/models/local/switch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model: modelId })
                });
                const data = await res.json();
                if (data.status === "ok") {
                    this.activeLocalModelId = modelId;
                    this.updateLocalModelDisplays(modelId);
                    if (modalHint) modalHint.textContent = `✓ Model ${modelId} loaded successfully!`;
                    this.showDynamicIsland("Local Model Ready", `${modelId} loaded on RTX 3050`, "✓", "cyan", 3000);
                    this.logEvent("LLM", `Local model successfully switched to ${modelId} (${data.parameters || ""})`, "success");
                    this.renderLocalModelCards();
                    setTimeout(() => this.closeLocalModelModal(), 600);
                } else {
                    if (modalHint) modalHint.textContent = `Error: ${data.message || data.error}`;
                    this.logEvent("ERROR", `Failed to switch local model: ${data.message || data.error}`, "error");
                }
            } catch (err) {
                if (modalHint) modalHint.textContent = `Network error: ${err.message}`;
                this.logEvent("ERROR", `Network error switching local model: ${err.message}`, "error");
            }
        };

        /**
         * Update labels and badges across Top Nav, HUD, and Chat view for local models.
         * @param {string} modelId - Active local model identifier.
         * @returns {void}
         */
        this.updateLocalModelDisplays = (modelId) => {
            const shortName = (modelId || "").replace(".gguf", "").replace("-Q4_K_M", "").replace("qwen", "Qwen ").replace("llama", "Llama ");
            if (topNavLabel) topNavLabel.textContent = shortName;
            if (chatLabel) chatLabel.textContent = shortName;
            const chatModelLabel = document.getElementById("chatModelLabel");
            if (chatModelLabel && this.currentLlmProvider === "local") {
                chatModelLabel.textContent = `LOCAL ${shortName.toUpperCase()}`;
            }
        };

        // Fetch initial local model status
        await this.refreshLocalModels();
    }

    /**
     * Initialize Whisper STT model manager, top nav trigger, audio deck chips, and modal catalog.
     * Manages whisper-server daemon on AMD Radeon 680M iGPU via Vulkan backend.
     * @returns {Promise<void>}
     */
    async initSttModelControls() {
        const topNavPicker = document.getElementById("btnTopNavSttPicker");
        const topNavLabel = document.getElementById("topNavSttPickerLabel");
        const chatPicker = document.getElementById("btnChatWhisperPicker");
        const chatLabel = document.getElementById("chatWhisperPickerLabel");
        const deckOpenBtn = document.getElementById("btnOpenSttModalDeck");
        const deckActivePill = document.getElementById("deckSttActivePill");
        const deckEngineState = document.getElementById("deckSttEngineState");
        const deckPresetChips = document.querySelectorAll("#sttPresetChips .stt-chip");
        const modal = document.getElementById("whisperModelModal");
        const btnClose = document.getElementById("btnCloseWhisperModelModal");
        const btnCancel = document.getElementById("btnCancelWhisperModelModal");
        const cardsGrid = document.getElementById("whisperModelCardsGrid");
        const modalHint = document.getElementById("whisperModalHint");
        const modalRunningStatus = document.getElementById("whisperModalRunningStatus");

        this.activeSttModelId = "base.en";
        this.sttModelsList = [];

        /**
         * Open Whisper STT Modal and fetch live models.
         * @returns {Promise<void>}
         */
        this.openSttModal = async () => {
            if (!modal) return;
            modal.style.display = "flex";
            modal.classList.add("open");
            await this.refreshSttStatus();
        };

        /**
         * Close Whisper STT Modal.
         * @returns {void}
         */
        this.closeSttModal = () => {
            if (!modal) return;
            modal.style.display = "none";
            modal.classList.remove("open");
        };

        if (topNavPicker) topNavPicker.addEventListener("click", () => this.openSttModal());
        if (chatPicker) chatPicker.addEventListener("click", () => this.openSttModal());
        if (deckOpenBtn) deckOpenBtn.addEventListener("click", () => this.openSttModal());
        if (btnClose) btnClose.addEventListener("click", () => this.closeSttModal());
        if (btnCancel) btnCancel.addEventListener("click", () => this.closeSttModal());
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) this.closeSttModal();
            });
        }

        // Wire Audio Deck Preset Chips
        deckPresetChips.forEach(chip => {
            chip.addEventListener("click", async () => {
                const targetModel = chip.dataset.model;
                if (!targetModel || targetModel === this.activeSttModelId) return;
                await this.switchSttModel(targetModel);
            });
        });

        /**
         * Query STT status and model catalog from backend.
         * @returns {Promise<void>}
         */
        this.refreshSttStatus = async () => {
            try {
                const res = await fetch("/api/stt/models");
                if (!res.ok) return;
                const data = await res.json();
                this.sttModelsList = data.models || [];
                if (data.active_model) {
                    this.activeSttModelId = data.active_model;
                    this.updateSttDisplays(data.active_model);
                }
                if (modalRunningStatus) {
                    modalRunningStatus.textContent = data.is_running ? "● ONLINE" : "○ STOPPED";
                    modalRunningStatus.style.color = data.is_running ? "#10b981" : "#ef4444";
                }
                if (deckEngineState) {
                    deckEngineState.textContent = data.is_running ? "ONLINE (Port 8081)" : "OFFLINE";
                    deckEngineState.className = `stt-state-tag ${data.is_running ? "online" : "offline"}`;
                }
                this.renderSttModelCards();
            } catch (err) {
                console.error("Failed to query STT status:", err);
            }
        };

        /**
         * Render model cards in the Whisper STT modal.
         * @returns {void}
         */
        this.renderSttModelCards = () => {
            if (!cardsGrid) return;
            cardsGrid.innerHTML = "";
            this.sttModelsList.forEach(m => {
                const isActive = m.id === this.activeSttModelId;
                const card = document.createElement("div");
                card.className = `model-profile-card stt-profile-card ${isActive ? "active" : ""}`;

                const speedBadge = m.latency_rating === "ultra_fast" ? "⚡ ULTRA FAST" : (m.latency_rating === "fast" ? "⚡ FAST" : "🔮 HIGH ACCURACY");

                card.innerHTML = `
                    <div class="model-card-header">
                        <span class="model-card-title">${m.name || m.id}</span>
                        <span class="model-card-badge">${speedBadge}</span>
                    </div>
                    <div class="model-card-desc">${m.description || "Whisper speech-to-text model accelerated on AMD Radeon 680M Vulkan."}</div>
                    <div class="model-card-meta">
                        <span class="meta-pill">Size: ${m.size_mb} MB</span>
                        <span class="meta-pill">Params: ${m.parameters}</span>
                        <span class="meta-pill">Lang: ${m.multilingual ? "Multilingual" : "English Only"}</span>
                        <span class="meta-pill">Device: Vulkan iGPU</span>
                    </div>
                    <div class="model-card-footer">
                        <button class="card-select-btn ${isActive ? "active-btn" : ""}" type="button">
                            ${isActive ? "✓ ACTIVE STT MODEL" : "ACTIVATE THIS MODEL"}
                        </button>
                    </div>
                `;

                if (!isActive) {
                    card.addEventListener("click", () => this.switchSttModel(m.id));
                }
                cardsGrid.appendChild(card);
            });
        };

        /**
         * Switch active Whisper STT model via REST API.
         * @param {string} modelId - Model ID to activate.
         * @returns {Promise<void>}
         */
        this.switchSttModel = async (modelId) => {
            if (modalHint) modalHint.textContent = `Hot-swapping whisper-server to ${modelId} on Vulkan...`;
            deckPresetChips.forEach(c => {
                if (c.dataset.model === modelId) c.classList.add("loading");
            });

            this.showDynamicIsland("Whisper STT Switch", `Loading ${modelId} (AMD Radeon 680M)...`, "🎙️", "emerald", 3500);
            this.logEvent("STT", `Hot-swapping Whisper STT to ${modelId} on Vulkan...`, "system");

            try {
                const res = await fetch("/api/stt/switch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ model: modelId })
                });
                const data = await res.json();
                if (data.status === "ok") {
                    this.activeSttModelId = modelId;
                    this.updateSttDisplays(modelId, data.size_mb);
                    if (modalHint) modalHint.textContent = `✓ Whisper ${modelId} active on port 8081 (Vulkan dev 0)!`;
                    this.showDynamicIsland("Whisper Ready", `${modelId} active (${data.size_mb || ""}MB)`, "✓", "emerald", 3000);
                    this.logEvent("STT", `Whisper model successfully hot-swapped to ${modelId} (${data.size_mb || ""}MB, Vulkan)`, "success");
                    this.renderSttModelCards();
                    setTimeout(() => this.closeSttModal(), 600);
                } else {
                    if (modalHint) modalHint.textContent = `Error: ${data.message || data.error}`;
                    this.logEvent("ERROR", `Failed to switch Whisper model: ${data.message || data.error}`, "error");
                }
            } catch (err) {
                if (modalHint) modalHint.textContent = `Network error: ${err.message}`;
                this.logEvent("ERROR", `Network error switching Whisper model: ${err.message}`, "error");
            } finally {
                deckPresetChips.forEach(c => c.classList.remove("loading"));
            }
        };

        /**
         * Update labels and active chips across Top Nav, Audio Deck, and Chat mode.
         * @param {string} modelId - Active model ID.
         * @param {number} [sizeMb] - Model file size in MB.
         * @returns {void}
         */
        this.updateSttDisplays = (modelId, sizeMb = null) => {
            if (topNavLabel) topNavLabel.textContent = modelId;
            if (chatLabel) chatLabel.textContent = `Whisper: ${modelId}`;
            if (deckActivePill) {
                const sz = sizeMb ? ` (${sizeMb} MB)` : "";
                deckActivePill.textContent = `${modelId}${sz}`;
            }
            deckPresetChips.forEach(c => {
                c.classList.toggle("active", c.dataset.model === modelId);
            });
        };

        // Initial fetch
        await this.refreshSttStatus();
    }

    /**
     * Initialize Text-to-Speech (TTS) multi-engine controls (Kokoro Neural & Piper ONNX).
     * Connects Top Nav picker, Dev Panel Audio Deck card, Chat view top bar,
     * and the 60+ voice Cyberdeck catalog modal with live audio previews and hot-swapping.
     * @returns {Promise<void>}
     */
    async initTtsModelControls() {
        const topNavPicker = document.getElementById("btnTopNavTtsPicker");
        const topNavLabel = document.getElementById("topNavTtsPickerLabel");
        const topNavBadge = document.getElementById("topNavTtsBadge");
        const chatPicker = document.getElementById("btnChatTtsPicker");
        const chatLabel = document.getElementById("chatTtsPickerLabel");
        const deckOpenBtn = document.getElementById("btnOpenTtsModalDeck");
        const deckActivePill = document.getElementById("deckTtsActivePill");
        const deckEngineState = document.getElementById("deckTtsEngineState");
        const btnDeckKokoro = document.getElementById("btnDeckSwitchKokoro");
        const btnDeckPiper = document.getElementById("btnDeckSwitchPiper");
        const deckPresetChips = document.querySelectorAll("#ttsPresetChips .tts-chip");

        const modal = document.getElementById("ttsModelModal");
        const btnClose = document.getElementById("btnCloseTtsModelModal");
        const btnCancel = document.getElementById("btnCancelTtsModelModal");
        const modalKokoroCard = document.getElementById("modalEngineCardKokoro");
        const modalPiperCard = document.getElementById("modalEngineCardPiper");
        const badgeKokoro = document.getElementById("badgeModalKokoro");
        const badgePiper = document.getElementById("badgeModalPiper");
        const filterChips = document.querySelectorAll("#ttsFilterChips .tts-filter-chip");
        const searchInput = document.getElementById("ttsVoiceSearchInput");
        const cardsGrid = document.getElementById("ttsVoiceCardsGrid");
        const modalHint = document.getElementById("ttsModalHint");
        const countAll = document.getElementById("countTtsAll");

        this.activeTtsEngine = "kokoro";
        this.activeTtsVoice = "af_heart";
        this.ttsVoicesList = [];
        this.ttsEnginesList = [];
        this.ttsActiveFilter = "all";
        this.ttsSearchQuery = "";
        this.ttsPreviewAudio = null;
        this.ttsActivePreviewBtn = null;

        /**
         * Open Text-to-Speech Modal and refresh voice catalog.
         * @returns {Promise<void>}
         */
        this.openTtsModal = async () => {
            if (!modal) return;
            modal.style.display = "flex";
            modal.classList.add("open");
            await this.refreshTtsStatus();
        };

        /**
         * Close Text-to-Speech Modal and stop any running preview audio.
         * @returns {void}
         */
        this.closeTtsModal = () => {
            if (!modal) return;
            modal.style.display = "none";
            modal.classList.remove("open");
            if (this.ttsPreviewAudio) {
                this.ttsPreviewAudio.pause();
                this.ttsPreviewAudio = null;
            }
            if (this.ttsActivePreviewBtn) {
                this.ttsActivePreviewBtn.classList.remove("playing");
                this.ttsActivePreviewBtn.textContent = "🔊 Preview";
                this.ttsActivePreviewBtn = null;
            }
        };

        if (topNavPicker) topNavPicker.addEventListener("click", () => this.openTtsModal());
        if (chatPicker) chatPicker.addEventListener("click", () => this.openTtsModal());
        if (deckOpenBtn) deckOpenBtn.addEventListener("click", () => this.openTtsModal());
        if (btnClose) btnClose.addEventListener("click", () => this.closeTtsModal());
        if (btnCancel) btnCancel.addEventListener("click", () => this.closeTtsModal());
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) this.closeTtsModal();
            });
        }

        // Audio deck engine switcher buttons
        if (btnDeckKokoro) {
            btnDeckKokoro.addEventListener("click", async () => {
                if (this.activeTtsEngine === "kokoro") return;
                await this.switchTtsEngineAndVoice("kokoro", "af_heart");
            });
        }
        if (btnDeckPiper) {
            btnDeckPiper.addEventListener("click", async () => {
                if (this.activeTtsEngine === "piper") return;
                await this.switchTtsEngineAndVoice("piper", "en-us-lessac-medium");
            });
        }

        // Modal engine choice cards
        if (modalKokoroCard) {
            modalKokoroCard.addEventListener("click", async () => {
                if (this.activeTtsEngine !== "kokoro") {
                    await this.switchTtsEngineAndVoice("kokoro", "af_heart");
                }
            });
        }
        if (modalPiperCard) {
            modalPiperCard.addEventListener("click", async () => {
                if (this.activeTtsEngine !== "piper") {
                    await this.switchTtsEngineAndVoice("piper", "en-us-lessac-medium");
                }
            });
        }

        // Audio deck preset chips
        deckPresetChips.forEach(chip => {
            chip.addEventListener("click", async () => {
                const eng = chip.dataset.engine || "kokoro";
                const voice = chip.dataset.voice;
                if (!voice) return;
                await this.switchTtsEngineAndVoice(eng, voice);
            });
        });

        // Filter chips in modal
        filterChips.forEach(chip => {
            chip.addEventListener("click", () => {
                filterChips.forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                this.ttsActiveFilter = chip.dataset.filter || "all";
                this.renderTtsVoiceCards();
            });
        });

        // Search input in modal
        if (searchInput) {
            searchInput.addEventListener("input", (e) => {
                this.ttsSearchQuery = (e.target.value || "").toLowerCase().trim();
                this.renderTtsVoiceCards();
            });
        }

        /**
         * Fetch TTS status and voices catalog from backend.
         * @returns {Promise<void>}
         */
        this.refreshTtsStatus = async () => {
            try {
                const [statusRes, voicesRes] = await Promise.all([
                    fetch("/api/tts/status"),
                    fetch("/api/tts/voices"),
                ]);

                if (statusRes.ok) {
                    const statusData = await statusRes.json();
                    this.activeTtsEngine = statusData.active_engine || "kokoro";
                    this.activeTtsVoice = statusData.active_voice || "af_heart";
                    this.ttsEnginesList = statusData.engines || [];
                    this.updateTtsDisplays(this.activeTtsEngine, this.activeTtsVoice, statusData.sample_rate);
                }

                if (voicesRes.ok) {
                    const voicesData = await voicesRes.json();
                    this.ttsVoicesList = voicesData.voices || [];
                    if (countAll) countAll.textContent = this.ttsVoicesList.length.toString();
                }

                this.renderTtsVoiceCards();
            } catch (err) {
                console.error("Failed to refresh TTS status:", err);
            }
        };

        /**
         * Render voice cards into the modal grid filtered by engine, gender, accent, and search term.
         * @returns {void}
         */
        this.renderTtsVoiceCards = () => {
            if (!cardsGrid) return;
            cardsGrid.innerHTML = "";

            // Filter voices based on active modal engine, category filter, and search
            const filtered = this.ttsVoicesList.filter(v => {
                // Must match active engine
                if (v.engine !== this.activeTtsEngine) return false;

                // Category filter
                if (this.ttsActiveFilter === "female" && v.gender !== "female") return false;
                if (this.ttsActiveFilter === "male" && v.gender !== "male") return false;
                if (this.ttsActiveFilter === "american" && v.accent !== "American") return false;
                if (this.ttsActiveFilter === "british" && v.accent !== "British") return false;
                if (this.ttsActiveFilter === "international" && (v.accent === "American" || v.accent === "British")) return false;

                // Search query
                if (this.ttsSearchQuery) {
                    const haystack = `${v.name} ${v.id} ${v.accent} ${v.style} ${v.gender}`.toLowerCase();
                    if (!haystack.includes(this.ttsSearchQuery)) return false;
                }
                return true;
            });

            if (filtered.length === 0) {
                cardsGrid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-dim); padding: 30px; font-family: var(--font-mono); font-size: 11px;">No voices found matching criteria.</div>`;
                return;
            }

            filtered.forEach(v => {
                const isActive = (v.id === this.activeTtsVoice || v.id.endsWith(this.activeTtsVoice) || this.activeTtsVoice.endsWith(v.id));
                const card = document.createElement("div");
                card.className = `tts-voice-card ${isActive ? "active" : ""}`;

                const genderIcon = v.gender === "female" ? "♀" : "♂";
                const srText = v.engine === "kokoro" ? "24kHz" : "16-22kHz";

                card.innerHTML = `
                    <div class="model-card-header">
                        <span class="model-card-title">${v.name} <span style="font-size: 10px; color: ${v.gender === 'female' ? '#f472b6' : '#60a5fa'}; font-weight: 700;">${genderIcon}</span></span>
                        <span class="model-card-badge">${v.accent.toUpperCase()}</span>
                    </div>
                    <div class="model-card-desc">${v.style || "Neural speech synthesis voice."}</div>
                    <div class="model-card-meta">
                        <span class="meta-pill">${v.id}</span>
                        <span class="meta-pill">${srText}</span>
                        <span class="meta-pill">${v.lang ? v.lang.toUpperCase() : "EN"}</span>
                    </div>
                    <div class="card-actions">
                        <button class="tts-preview-btn" type="button" data-id="${v.id}" data-engine="${v.engine}">🔊 Preview</button>
                        <button class="tts-select-btn ${isActive ? 'active-btn' : ''}" type="button" data-id="${v.id}" data-engine="${v.engine}">${isActive ? '✓ Active' : 'Select'}</button>
                    </div>
                `;

                // Wire Preview button
                const btnPreview = card.querySelector(".tts-preview-btn");
                if (btnPreview) {
                    btnPreview.addEventListener("click", async (e) => {
                        e.stopPropagation();
                        await this.previewTtsVoice(v.engine, v.id, btnPreview);
                    });
                }

                // Wire Select button
                const btnSelect = card.querySelector(".tts-select-btn");
                if (btnSelect) {
                    btnSelect.addEventListener("click", async (e) => {
                        e.stopPropagation();
                        if (isActive) return;
                        await this.switchTtsEngineAndVoice(v.engine, v.id);
                    });
                }

                cardsGrid.appendChild(card);
            });
        };

        /**
         * Synthesize preview audio for a specific voice using POST /api/tts/test.
         * @param {string} engine - TTS engine name ('kokoro' or 'piper').
         * @param {string} voiceId - Voice identifier.
         * @param {HTMLButtonElement} btn - The preview button clicked.
         * @returns {Promise<void>}
         */
        this.previewTtsVoice = async (engine, voiceId, btn) => {
            if (this.ttsPreviewAudio && this.ttsActivePreviewBtn === btn) {
                this.ttsPreviewAudio.pause();
                this.ttsPreviewAudio = null;
                btn.classList.remove("playing");
                btn.textContent = "🔊 Preview";
                this.ttsActivePreviewBtn = null;
                return;
            }

            if (this.ttsPreviewAudio) {
                this.ttsPreviewAudio.pause();
                this.ttsPreviewAudio = null;
            }
            if (this.ttsActivePreviewBtn) {
                this.ttsActivePreviewBtn.classList.remove("playing");
                this.ttsActivePreviewBtn.textContent = "🔊 Preview";
            }

            btn.classList.add("playing");
            btn.textContent = "⏳ Generating...";
            this.ttsActivePreviewBtn = btn;

            try {
                const sampleText = engine === "kokoro"
                    ? "Hello! This is Kokoro, an ultra-realistic neural voice running locally on the robot."
                    : "Hello! This is Piper, a high-speed local speech synthesis voice.";

                const res = await fetch("/api/tts/test", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        engine: engine,
                        voice: voiceId,
                        text: sampleText,
                    }),
                });

                if (!res.ok) throw new Error(`HTTP ${res.status}`);

                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const audio = new Audio(url);
                this.ttsPreviewAudio = audio;

                btn.textContent = "⏹️ Stop";

                audio.onended = () => {
                    btn.classList.remove("playing");
                    btn.textContent = "🔊 Preview";
                    if (this.ttsActivePreviewBtn === btn) this.ttsActivePreviewBtn = null;
                    this.ttsPreviewAudio = null;
                    URL.revokeObjectURL(url);
                };

                audio.onerror = () => {
                    btn.classList.remove("playing");
                    btn.textContent = "🔊 Preview";
                    if (this.ttsActivePreviewBtn === btn) this.ttsActivePreviewBtn = null;
                    this.ttsPreviewAudio = null;
                    URL.revokeObjectURL(url);
                };

                await audio.play();
            } catch (err) {
                console.error("Preview failed:", err);
                btn.classList.remove("playing");
                btn.textContent = "❌ Failed";
                setTimeout(() => {
                    btn.textContent = "🔊 Preview";
                }, 1500);
            }
        };

        /**
         * Switch the active TTS engine and voice via POST /api/tts/switch.
         * @param {string} engine - Engine identifier ('kokoro' or 'piper').
         * @param {string} [voice] - Voice identifier to set.
         * @returns {Promise<void>}
         */
        this.switchTtsEngineAndVoice = async (engine, voice = null) => {
            if (modalHint) modalHint.textContent = `Switching TTS to ${engine.toUpperCase()} (${voice || 'default'})...`;

            try {
                const res = await fetch("/api/tts/switch", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ engine: engine, voice: voice }),
                });

                const data = await res.json();
                if (data.status === "ok") {
                    this.activeTtsEngine = data.engine;
                    this.activeTtsVoice = data.voice;
                    this.updateTtsDisplays(data.engine, data.voice, data.sample_rate);
                    if (modalHint) modalHint.textContent = `✓ TTS active: ${data.engine.toUpperCase()} (${data.voice}) at ${data.sample_rate}Hz!`;
                    this.showDynamicIsland("TTS Ready", `${data.engine.toUpperCase()} (${data.voice})`, "✓", "pink", 3000);
                    this.logEvent("TTS", `Switched active TTS engine to ${data.engine.toUpperCase()} (${data.voice})`, "success");
                    this.renderTtsVoiceCards();
                    setTimeout(() => this.closeTtsModal(), 500);
                } else {
                    if (modalHint) modalHint.textContent = `Error: ${data.detail || data.error}`;
                }
            } catch (err) {
                console.error("Failed to switch TTS engine:", err);
                if (modalHint) modalHint.textContent = `Network error: ${err.message}`;
            }
        };

        /**
         * Update labels and active chips across Top Nav, Audio Deck, Chat top bar, and modal cards.
         * @param {string} engine - Active engine ('kokoro' or 'piper').
         * @param {string} voice - Active voice name or ID.
         * @param {number} [sampleRate] - Sample rate in Hz.
         * @returns {void}
         */
        this.updateTtsDisplays = (engine, voice, sampleRate = 24000) => {
            const cleanVoice = voice ? PathBasename(voice).replace(".onnx", "").replace("en-us-", "") : "default";
            const engUpper = engine ? engine.toUpperCase() : "TTS";
            const srText = sampleRate ? `${Math.round(sampleRate / 1000)}kHz` : "24kHz";

            if (topNavLabel) topNavLabel.textContent = `${engUpper}: ${cleanVoice}`;
            if (topNavBadge) topNavBadge.textContent = srText;
            if (chatLabel) chatLabel.textContent = `TTS: ${cleanVoice}`;

            if (deckActivePill) {
                deckActivePill.textContent = `${cleanVoice} (${engUpper})`;
            }

            if (btnDeckKokoro) btnDeckKokoro.classList.toggle("active", engine === "kokoro");
            if (btnDeckPiper) btnDeckPiper.classList.toggle("active", engine === "piper");

            if (modalKokoroCard) modalKokoroCard.classList.toggle("active", engine === "kokoro");
            if (modalPiperCard) modalPiperCard.classList.toggle("active", engine === "piper");
            if (badgeKokoro) {
                badgeKokoro.className = `choice-badge ${engine === 'kokoro' ? 'badge-active' : ''}`;
                badgeKokoro.textContent = engine === "kokoro" ? "ACTIVE" : "SELECT";
            }
            if (badgePiper) {
                badgePiper.className = `choice-badge ${engine === 'piper' ? 'badge-active' : ''}`;
                badgePiper.textContent = engine === "piper" ? "ACTIVE" : "SELECT";
            }

            deckPresetChips.forEach(chip => {
                const chipEng = chip.dataset.engine || "kokoro";
                const chipVoice = chip.dataset.voice || "";
                const matches = (chipEng === engine) && (chipVoice === voice || voice.includes(chipVoice) || chipVoice.includes(voice));
                chip.classList.toggle("active", matches);
            });
        };

        // Helper to get file basename
        function PathBasename(str) {
            if (!str) return "";
            const parts = str.split(/[\\/]/);
            return parts[parts.length - 1];
        }

        // Initial fetch
        await this.refreshTtsStatus();
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

        // Dynamically get active theme colors from CSS custom properties on document.body
        const computed = getComputedStyle(document.body);
        const matrixGreen = (computed.getPropertyValue("--matrix-green") || "#00ff66").trim();
        const matrixBright = (computed.getPropertyValue("--matrix-bright") || "#ffffff").trim();

        // Translucent fade over canvas to produce trails
        this.ctx.fillStyle = "rgba(5, 5, 5, 0.1)";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.font = `${this.fontSize}px 'JetBrains Mono', monospace`;

        for (let i = 0; i < this.drops.length; i++) {
            const char = this.characters.charAt(Math.floor(Math.random() * this.characters.length));
            const x = i * this.fontSize;
            const y = this.drops[i] * this.fontSize;

            // Highlight leading glyph in bright accent / white
            if (Math.random() > 0.88) {
                this.ctx.fillStyle = matrixBright;
            } else {
                this.ctx.fillStyle = matrixGreen;
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
    window.app = window.robotUI;
});
