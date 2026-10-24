/**
 * Robot Core — Frontend Controller
 * Connects to Robot Core WebSocket & REST API:
 * - Real-time state-driven facial expressions & natural blinking
 * - Mouth audio-reactive animation synced with speaking
 * - World State & Camera Snapshot auto-refresh
 * - Live pipeline telemetry metrics (STT, LLM, TTS, E2E)
 * - View switching: Face Only, Dual Mode, Console Only
 */

/** Reconnect delay for the MJPEG camera stream after a failure, and its ceiling. */
const CAMERA_RETRY_BASE_MS = 1500;
const CAMERA_RETRY_MAX_MS = 30000;
/** Give up after this many consecutive failures; the radar standby stays up until the user re-arms. */
const CAMERA_RETRY_MAX_ATTEMPTS = 6;

class RobotUI {
    constructor() {
        this.ws = null;
        this.currentMode = "idle";
        this.blinkInterval = null;
        this.mouthInterval = null;
        this.cameraInterval = null;
        this.worldStateInterval = null;

        // ─── Timer / lifecycle hygiene ───────────────────────────────────────
        // Every interval and tracked timeout created by this instance is recorded
        // here so it can be torn down in one place (see stopAllTimers()).
        this._intervals = new Set();
        this._timeouts = new Set();
        this._stopped = false;
        this._pollCounters = { status: 0, traces: 0 };

        this.initAuth();

        // Tear down all timers/streams when the page is hidden or unloaded so
        // backgrounded tabs do not keep polling or reconnecting.
        this._onPageLeave = () => this.stopAllTimers();
        window.addEventListener("pagehide", this._onPageLeave);
        window.addEventListener("beforeunload", this._onPageLeave);

        // DOM Elements
        this.faceStage = document.getElementById("faceStage");
        this.faceCaption = document.getElementById("faceCaption");
        this.statusPill = document.getElementById("globalStatusPill");
        this.statusText = document.getElementById("globalStatusText");
        this.personaSelect = document.getElementById("personaSelect");
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
        this.initEngineCluster();
        this.initOverlayManager();
        this.initPersonalitySelector();
        this.initPersonaManager();
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
        this.initImageGallery();
        this.initHfDownloadMonitor();
        this.initAudioDeck();
        this.initMemoryDeck();
        this.initPipelineTelemetryDeck();
        this.startPollers();
        this.fetchThinkingState();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Timer lifecycle helpers
    //
    // All recurring/deferred work goes through these helpers so that every id is
    // recorded in this._intervals / this._timeouts and can be cancelled at once
    // by stopAllTimers() on pagehide/beforeunload.
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create a tracked interval.
     * @param {Function} fn - Callback invoked on each tick.
     * @param {number} ms - Period in milliseconds.
     * @returns {number} Interval id.
     */
    _interval(fn, ms) {
        const id = setInterval(() => {
            if (this._stopped) {
                this._clearIv(id);
                return;
            }
            fn();
        }, ms);
        this._intervals.add(id);
        return id;
    }

    /**
     * Clear and untrack an interval id (safe with null/undefined and untracked ids).
     * @param {number|null} id - Interval id returned by _interval() or setInterval().
     * @returns {void}
     */
    _clearIv(id) {
        if (id === null || id === undefined) return;
        clearInterval(id);
        this._intervals.delete(id);
    }

    /**
     * Create a tracked timeout.
     * @param {Function} fn - Callback invoked once.
     * @param {number} ms - Delay in milliseconds.
     * @returns {number} Timeout id.
     */
    _timeout(fn, ms) {
        const id = setTimeout(() => {
            this._timeouts.delete(id);
            if (this._stopped) return;
            fn();
        }, ms);
        this._timeouts.add(id);
        return id;
    }

    /**
     * Clear and untrack a timeout id.
     * @param {number|null} id - Timeout id returned by _timeout().
     * @returns {void}
     */
    _clearTmo(id) {
        if (id === null || id === undefined) return;
        clearTimeout(id);
        this._timeouts.delete(id);
    }

    /**
     * Cancel every timer owned by this instance and stop background work.
     * Bound to pagehide/beforeunload; also stops the WebSocket reconnect loop,
     * requestAnimationFrame loops and the ambient MatrixRain canvas.
     * @returns {void}
     */
    stopAllTimers() {
        if (this._stopped) return;
        this._stopped = true;

        this._intervals.forEach((id) => clearInterval(id));
        this._intervals.clear();
        this._timeouts.forEach((id) => clearTimeout(id));
        this._timeouts.clear();

        this._clearTmo(this._wsConnectTimer);
        this._wsConnectTimer = null;

        if (this._avatarThoughtTimerInterval) {
            this._clearIv(this._avatarThoughtTimerInterval);
            this._avatarThoughtTimerInterval = null;
        }
        if (this.chatRecordTimerInterval) {
            this._clearIv(this.chatRecordTimerInterval);
            this.chatRecordTimerInterval = null;
        }
        if (this.hfDownloadPollInterval) {
            this._clearIv(this.hfDownloadPollInterval);
            this.hfDownloadPollInterval = null;
        }
        if (this.blinkInterval) {
            clearTimeout(this.blinkInterval);
            this.blinkInterval = null;
        }
        if (this.chatWaveAnimFrame) {
            cancelAnimationFrame(this.chatWaveAnimFrame);
            this.chatWaveAnimFrame = null;
        }
        try {
            this.stopSpeakingMessage();
        } catch (e) {}
        try {
            this._stopTtsPreview();
        } catch (e) {}
        try {
            this.cleanupChatVoiceRecord();
        } catch (e) {}
        try {
            if (this.ws) {
                this.ws.onclose = null;
                this.ws.onmessage = null;
                this.ws.close();
            }
        } catch (e) {}
        // Stop the ambient background canvas loop too.
        if (window.matrixRain) window.matrixRain._stopped = true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Auth-aware request helpers
    //
    // A bearer token, when present, is kept in sessionStorage and attached to
    // every request from these two helpers. With no token stored, requests go out
    // completely unmodified (the backend login page / 401 handles the rest).
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Capture a ?token= URL parameter into sessionStorage and strip it from the
     * browser address bar so it is not left in history or visible in the URL.
     * @returns {void}
     */
    initAuth() {
        try {
            const url = new URL(window.location.href);
            const token = url.searchParams.get("token") || url.searchParams.get("access_token");
            if (!token) return;
            sessionStorage.setItem("robot_auth_token", token);
            url.searchParams.delete("token");
            url.searchParams.delete("access_token");
            const query = url.searchParams.toString();
            history.replaceState(null, "", url.pathname + (query ? `?${query}` : "") + url.hash);
        } catch (e) {
            // sessionStorage / history unavailable: requests stay anonymous.
        }
    }

    /**
     * @returns {string} Stored bearer token, or empty string when anonymous.
     */
    getAuthToken() {
        try {
            return sessionStorage.getItem("robot_auth_token") || "";
        } catch (e) {
            return "";
        }
    }

    /**
     * Central fetch wrapper: adds an Authorization: Bearer header when a token is
     * stored. All API calls in this controller go through here.
     * @param {string|Request} input - Request URL or Request object.
     * @param {RequestInit} [init] - Fetch options.
     * @returns {Promise<Response>}
     */
    apiFetch(input, init) {
        const token = this.getAuthToken();
        if (!token) return fetch(input, init);
        const options = init ? { ...init } : {};
        const headers = new Headers(options.headers || {});
        if (!headers.has("Authorization")) headers.set("Authorization", `Bearer ${token}`);
        options.headers = headers;
        return fetch(input, options);
    }

    /**
     * Central URL builder for header-less streaming GETs (WebSocket, MJPEG).
     * Appends ?token= when a bearer token is available.
     * @param {string} url - Target URL.
     * @returns {string} URL with token query parameter, if any.
     */
    streamUrl(url) {
        const token = this.getAuthToken();
        if (!token || !url) return url;
        return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
    }

    /**
     * @returns {boolean} True while the /ws/events WebSocket is connected.
     */
    _isWsOpen() {
        return Boolean(this.ws) && this.ws.readyState === WebSocket.OPEN;
    }

    /**
     * Show or hide a self-contained "telemetry offline" indicator.
     * @param {boolean} offline - Whether reconnect attempts have been exhausted.
     * @param {string} [detail] - Optional detail line.
     * @returns {void}
     */
    _setWsOfflineIndicator(offline, detail) {
        let el = document.getElementById("wsOfflineBadge");
        if (offline) {
            if (!el) {
                el = document.createElement("div");
                el.id = "wsOfflineBadge";
                el.style.cssText =
                    "position:fixed;top:8px;right:8px;z-index:9999;padding:6px 10px;border-radius:8px;" +
                    "background:rgba(239,68,68,0.16);border:1px solid #ef4444;color:#fca5a5;" +
                    "font:11px/1.3 monospace;pointer-events:none;";
                document.body.appendChild(el);
            }
            el.textContent = detail ? `TELEMETRY OFFLINE • ${detail}` : "TELEMETRY OFFLINE (poll fallback active)";
            el.style.display = "block";
        } else if (el) {
            el.style.display = "none";
        }
    }

    /**
     * Keep both chat send buttons in step with their composers: a blank
     * message or an in-flight reply means there is nothing to submit.
     * @returns {void}
     */
    syncChatComposer() {
        const mainBtn = document.getElementById("chatSendBtn");
        const mainInput = document.getElementById("chatInput");
        if (mainBtn && mainInput) {
            mainBtn.disabled = mainInput.value.trim() === "";
        }
        if (this.btnChatSubmit && this.chatTextarea) {
            this.btnChatSubmit.disabled = Boolean(this.chatStreaming) || this.chatTextarea.value.trim() === "";
        }
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
                    const res = await this.apiFetch("/voice-mode", {
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
                this.syncChatComposer();
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
                const resp = await this.apiFetch("/chat", {
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
                this.syncChatComposer();
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

        input.addEventListener("input", () => this.syncChatComposer());
        this.syncChatComposer();
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
                    // The drawer starts under the one-row bar; the opened suite strip
                    // would cover its header.
                    if (isOpen) this.setEngineClusterOpen(false);
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
                this.syncChatComposer();
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

        this.syncChatComposer();

        if (this.btnChatStopStream) {
            this.btnChatStopStream.addEventListener("click", (e) => {
                e.preventDefault();
                this.stopChatStream();
            });
        }

        // The keyboard-shortcut hint cannot fit next to the mic/send buttons.
        if (this.chatTextarea) {
            const desktopPlaceholder = this.chatTextarea.placeholder;
            const mobileMedia = window.matchMedia("(max-width: 768px)");
            const syncPlaceholder = () => {
                this.chatTextarea.placeholder = mobileMedia.matches ? "Message the robot…" : desktopPlaceholder;
            };
            if (mobileMedia.addEventListener) {
                mobileMedia.addEventListener("change", syncPlaceholder);
            }
            syncPlaceholder();
        }
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
     * Checks the header dropdown selector or the active persona property.
     * @returns {string} Formatted title ('Cypher', 'Luna', 'Aria', 'GLaDOS', or 'Robot').
     */
    getActivePersonaTitle() {
        if (this.personaSelect && this.personaSelect.value) {
            const val = this.personaSelect.value.toLowerCase().trim();
            if (val === "neo" || val === "cypher") return "Cypher";
            if (val === "glados") return "GLaDOS";
            return val.charAt(0).toUpperCase() + val.slice(1);
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
            this._clearTmo(this._islandTimeout);
        }

        this._islandTimeout = this._timeout(() => {
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
            const res = await this.apiFetch("/api/thinking");
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
            const res = await this.apiFetch("/api/thinking", {
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
                if (this._avatarThoughtTimerInterval) this._clearIv(this._avatarThoughtTimerInterval);
                this._avatarThoughtTimerInterval = this._interval(() => {
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
                this._clearIv(this._avatarThoughtTimerInterval);
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
                this._timeout(() => {
                    if (badge && !this._avatarThoughtTimerInterval) {
                        badge.style.display = "none";
                    }
                }, 2500);
            }
            // Auto-collapse the holographic thought matrix after 10s of quiet idle
            if (this._avatarThoughtHideTimeout) this._clearTmo(this._avatarThoughtHideTimeout);
            this._avatarThoughtHideTimeout = this._timeout(() => {
                this._avatarThoughtHideTimeout = null;
                if (card && !this._avatarThoughtTimerInterval) {
                    card.classList.add("collapsed");
                }
            }, 10000);
        } else if (action === "hide") {
            if (this._avatarThoughtTimerInterval) {
                this._clearIv(this._avatarThoughtTimerInterval);
                this._avatarThoughtTimerInterval = null;
            }
            if (card) card.style.display = "none";
            if (badge) badge.style.display = "none";
        }
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
            await this.apiFetch("/voice-mode", {
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
            await this.apiFetch("/api/view-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: "chat" }),
            });
        } catch (e) {}

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
            const res = await this.apiFetch("/voice-mode", {
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
            await this.apiFetch("/api/view-mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode: "dual" }),
            });
        } catch (e) {}
    }

    async loadChatSessions() {
        if (!this.chatSessionsList) return;
        try {
            const res = await this.apiFetch("/chat/sessions");
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
            const res = await this.apiFetch("/chat/sessions", {
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
            const res = await this.apiFetch(`/chat/sessions/${sessionId}`);
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
            this.scrollChatToBottom();
        } catch (e) {
            console.error("Failed to select chat session:", e);
        }
    }

    async deleteChatSession(sessionId) {
        if (!confirm("Are you sure you want to delete this conversation thread?")) return;
        try {
            const res = await this.apiFetch(`/chat/sessions/${sessionId}`, { method: "DELETE" });
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

    /**
     * Render the chat welcome splash screen tailored to the active personality and isolated database.
     * @returns {void}
     */
    renderChatWelcome() {
        if (!this.chatMessagesContainer) return;
        const rawPersona = this.personaSelect ? this.personaSelect.value : "neo";
        const personaTitle = (typeof this.formatPersonaName === "function")
            ? this.formatPersonaName(rawPersona)
            : (rawPersona === "neo" ? "CYPHER" : rawPersona.toUpperCase());

        this.chatMessagesContainer.innerHTML = `
            <div class="chat-welcome-hero">
                <div class="hero-logo">🤖</div>
                <h2 class="hero-title">${this.escapeHtml(personaTitle)} AI WORKSPACE</h2>
                <p class="hero-desc">
                    Isolated conversational thread with independent persona database and zero memory leakage.
                    Microphone is unloaded to preserve audio privacy and hardware bandwidth.
                </p>
                <div class="hero-features">
                    <div class="hero-feat"><span class="feat-icon">⚡</span> Real-time token streaming with live tok/s</div>
                    <div class="hero-feat"><span class="feat-icon">🛠️</span> Interactive tool execution telemetry cards</div>
                    <div class="hero-feat"><span class="feat-icon">🧠</span> Isolated vector facts & persistent database</div>
                </div>
            </div>
        `;
    }

    /**
     * Synchronize all UI indicators, welcome hero, and hardware pickers when active persona changes.
     * @param {string} personaId - Persona identifier (e.g. 'jarvis', 'luna', 'neo').
     * @param {string} [personaTitle] - Optional display title (e.g. 'Jarvis', 'Luna', 'CYPHER').
     * @returns {Promise<void>}
     */
    async updateActivePersonaUI(personaId, personaTitle) {
        if (!personaId) return;
        const cleanId = personaId.toLowerCase();
        const title = personaTitle || ((typeof this.formatPersonaName === "function")
            ? this.formatPersonaName(cleanId)
            : (cleanId === "neo" ? "CYPHER" : cleanId.toUpperCase()));

        if (this.personaSelect && this.personaSelect.value !== cleanId) {
            this.personaSelect.value = cleanId;
        }

        // Dynamically update existing welcome hero in DOM if visible
        const existingHeroTitle = document.querySelector(".chat-welcome-hero .hero-title");
        if (existingHeroTitle) {
            existingHeroTitle.textContent = `${title} AI WORKSPACE`;
        }

        // Update assistant sender tags in the active chat view
        document.querySelectorAll(".msg-assistant .chat-msg-sender").forEach((el) => {
            el.textContent = title;
        });

        this.applyPersonaTheme(cleanId);

        // Refresh hardware pickers to display newly applied models (TTS, STT, LLM)
        try {
            await Promise.all([
                this.refreshTtsStatus ? this.refreshTtsStatus() : Promise.resolve(),
                this.refreshSttStatus ? this.refreshSttStatus() : Promise.resolve(),
                this.refreshLlmStatus ? this.refreshLlmStatus() : Promise.resolve(),
                this.loadChatSessions ? this.loadChatSessions() : Promise.resolve(),
            ]);
        } catch (e) {
            console.error("Failed to refresh status after persona switch:", e);
        }

        // If the chat messages container is empty or showing the hero, re-render welcome hero
        if (this.chatMessagesContainer && (!this.chatMessagesContainer.children.length || document.querySelector(".chat-welcome-hero"))) {
            this.renderChatWelcome();
        }
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

        const headerHtml = `
            <div class="chat-msg-header">
                <span class="chat-msg-sender">${senderLabel}</span>
                ${badgeHtml}
                <span class="chat-msg-time">${timeStr}</span>
            </div>`;

        row.innerHTML = `
            <div class="chat-msg-content">
                ${headerHtml}
                ${thoughtHtml}
                ${toolHtml}
                <div class="chat-msg-body">${contentHtml}</div>
                ${actionsHtml}
                ${telemHtml}
            </div>
        `;
        this.chatMessagesContainer.appendChild(row);
        this._pruneChatMessages();

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
        this.syncChatComposer();
        if (this.btnChatStopStream) this.btnChatStopStream.style.display = "inline-flex";
        this.chatTextarea.value = "";
        this.chatTextarea.style.height = "auto";

        // Append User bubble
        const userRow = document.createElement("div");
        userRow.className = "chat-msg-row msg-user";
        userRow.innerHTML = `
            <div class="chat-msg-content">
                <div class="chat-msg-header">
                    <span class="chat-msg-sender">YOU</span>
                    <span class="chat-msg-time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
                </div>
                <div class="chat-msg-body">${this.renderChatMarkdown(text)}</div>
            </div>
        `;
        this.chatMessagesContainer.appendChild(userRow);
        this._pruneChatMessages();
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
        this._pruneChatMessages();
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
            if (thoughtIntervalId) this._clearIv(thoughtIntervalId);
            thoughtIntervalId = this._interval(() => {
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
            const res = await this.apiFetch("/chat/stream", {
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
                            thoughtIntervalId = this._interval(() => {
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
                            this._clearIv(thoughtIntervalId);
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
                            this._clearIv(thoughtIntervalId);
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
                            this._scheduleStreamingRender(textSpan, streamedRawText);
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
                            this._flushStreamingRender(textSpan, "");
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
                            this._clearIv(thoughtIntervalId);
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
                        }
                        // Guaranteed final render (throttled renders are flushed here).
                        this._flushStreamingRender(textSpan, streamedRawText);
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
                            this._clearIv(thoughtIntervalId);
                            thoughtIntervalId = null;
                        }
                        this._appendStreamSuffix(textSpan, `<div class="chat-stream-error">[Error: ${this.escapeHtml(payload.error || "Unknown stream error")}]</div>`);
                    }
                }
            }
        } catch (err) {
            if (err.name === "AbortError") {
                this._appendStreamSuffix(textSpan, ` <span class="chat-stream-stopped">[Generation stopped]</span>`);
            } else if (!isStreamCompleted) {
                this._appendStreamSuffix(textSpan, ` <span class="chat-stream-error">[Error: ${this.escapeHtml(err.message)}]</span>`);
            }
        } finally {
            if (thoughtIntervalId) {
                this._clearIv(thoughtIntervalId);
                thoughtIntervalId = null;
            }
            // Guarantee one final, unthrottled render so the visible markdown always
            // matches the complete streamed text even if the stream ended abruptly.
            this._flushStreamingRender(textSpan, streamedRawText);
            if (cursorSpan) cursorSpan.remove();
            this.chatStreaming = false;
            this.chatAbortController = null;
            this.syncChatComposer();
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
            // Track the in-flight blob URL so a mid-phrase stop revokes it too
            // (it has already been shifted out of this.phraseQueue at this point).
            this.currentAudioUrl = item.audioUrl;

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
                if (this.currentAudioUrl === item.audioUrl) {
                    this.currentAudioUrl = null;
                }
                if (this.currentAudio === audio) {
                    this.currentAudio = null;
                }
                // Drop handlers and release the decoded element for GC.
                audio.onended = null;
                audio.onerror = null;
                audio.src = "";
                audio.load();
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
            const resp = await this.apiFetch("/api/chat/speak/stream", {
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
     * Stop a TTS voice-preview clip and release its blob URL.
     * Previously paused previews leaked their ObjectURL and kept handlers attached.
     * @returns {void}
     */
    _stopTtsPreview() {
        const audio = this.ttsPreviewAudio;
        if (audio) {
            try {
                audio.pause();
                audio.onended = null;
                audio.onerror = null;
                audio.src = "";
                audio.load();
            } catch (_) {}
            this.ttsPreviewAudio = null;
        }
        if (this.ttsPreviewUrl) {
            try {
                URL.revokeObjectURL(this.ttsPreviewUrl);
            } catch (_) {}
            this.ttsPreviewUrl = null;
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
                // Detach handlers so a late ended/error event cannot resume playback.
                this.currentAudio.onended = null;
                this.currentAudio.onerror = null;
                this.currentAudio.src = "";
                this.currentAudio.load();
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
        if (this.chatRecordTimerInterval) this._clearIv(this.chatRecordTimerInterval);
        this.chatRecordTimerInterval = this._interval(() => {
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
                if (this._stopped || !this.chatRecording || !this.chatAnalyser) return;
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
            this.syncChatComposer();
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
                const res = await this.apiFetch("/api/transcribe", {
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
            this.syncChatComposer();
        }
    }

    cleanupChatVoiceRecord() {
        this.chatRecording = false;
        if (this.chatRecordTimerInterval) {
            this._clearIv(this.chatRecordTimerInterval);
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

    /**
     * Release per-row resources before a chat row is detached (blob URLs recorded
     * on the row, plus an optional row-owned cleanup callback). Listeners bound to
     * the row and its children are dropped together with the detached nodes.
     * @param {Element} row - Chat message row element.
     * @returns {void}
     */
    _releaseChatRow(row) {
        if (!row) return;
        if (typeof row._cleanup === "function") {
            try {
                row._cleanup();
            } catch (e) {}
            row._cleanup = null;
        }
        if (Array.isArray(row._blobUrls)) {
            row._blobUrls.forEach((u) => {
                try {
                    URL.revokeObjectURL(u);
                } catch (e) {}
            });
            row._blobUrls = null;
        }
    }

    /**
     * Keep the chat transcript bounded to CHAT_MAX_ROWS message rows by detaching
     * the oldest ones in a single Range.deleteContents() DOM operation.
     * @param {number} [maxRows] - Maximum number of rows to retain.
     * @returns {void}
     */
    _pruneChatMessages(maxRows = RobotUI.CHAT_MAX_ROWS) {
        const container = this.chatMessagesContainer;
        if (!container) return;
        const rows = container.children;
        if (rows.length <= maxRows) return;

        const excess = rows.length - maxRows;
        for (let i = 0; i < excess; i++) {
            this._releaseChatRow(rows[i]);
        }

        const range = document.createRange();
        range.setStartBefore(rows[0]);
        range.setEndAfter(rows[excess - 1]);
        range.deleteContents();
        range.detach && range.detach();
    }

    /**
     * Coalesce streaming markdown re-renders to at most one per ~100ms.
     * Renders immediately when the throttle window has elapsed, otherwise schedules
     * a single trailing render so no token is ever lost.
     * @param {HTMLElement} el - Target element whose innerHTML is re-rendered.
     * @param {string} text - Latest raw markdown text.
     * @returns {void}
     */
    _scheduleStreamingRender(el, text) {
        if (!el) return;
        el._pendingText = text;
        if (el._renderTimer) return;

        const since = performance.now() - (el._lastRenderAt || 0);
        if (since >= RobotUI.STREAM_RENDER_MS) {
            this._renderStreamingInto(el);
            return;
        }
        el._renderTimer = this._timeout(() => {
            el._renderTimer = null;
            this._renderStreamingInto(el);
        }, RobotUI.STREAM_RENDER_MS - since);
    }

    /**
     * Perform one throttled markdown render.
     * @param {HTMLElement} el - Target element.
     * @returns {void}
     */
    _renderStreamingInto(el) {
        el._lastRenderAt = performance.now();
        el.innerHTML = this.renderChatMarkdown(el._pendingText || "") + (el._suffixHtml || "");
    }

    /**
     * Cancel throttling and guarantee a final render for the given text.
     * @param {HTMLElement} el - Target element.
     * @param {string} [text] - Final raw markdown text (defaults to the pending text).
     * @returns {void}
     */
    _flushStreamingRender(el, text) {
        if (!el) return;
        if (el._renderTimer) {
            this._clearTmo(el._renderTimer);
            el._renderTimer = null;
        }
        if (text !== undefined) el._pendingText = text;
        this._renderStreamingInto(el);
    }

    /**
     * Append trailing markup (error / stopped notices) that must survive later
     * renders, then guarantee a final render.
     * @param {HTMLElement} el - Target element.
     * @param {string} html - Already-escaped HTML to append after the rendered text.
     * @returns {void}
     */
    _appendStreamSuffix(el, html) {
        if (!el) return;
        el._suffixHtml = (el._suffixHtml || "") + html;
        this._flushStreamingRender(el);
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
                modelUrl: "/static/models/avatar.glb?v=9.0-skinny-cute-pro",
            });
            this.logEvent("3D-CORE", "Animated 3D character initialized (WebGL2).", "system");

            // Update debug metrics every 500ms (tracked so it is torn down on unload)
            this._interval(() => {
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
            await this.apiFetch("/character/action", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(actionObj),
            });
        } catch (e) {}
    }

    /**
     * Point the camera image at an MJPEG endpoint and restart the reconnect budget.
     * @param {string} path - Stream path, e.g. "/camera/stream".
     * @returns {void}
     */
    armCameraStream(path) {
        this.clearCameraReconnect();
        this._cameraRetryAttempts = 0;
        this._cameraRetryDelay = CAMERA_RETRY_BASE_MS;
        this.requestCameraStream(path);
    }

    /**
     * Request an MJPEG endpoint without touching the reconnect budget.
     * @param {string} path - Stream path, e.g. "/camera/stream".
     * @returns {void}
     */
    requestCameraStream(path) {
        if (!this.cameraImg) return;
        this.cameraImg.style.display = "block";
        this.cameraImg.src = this.streamUrl(`${path}?t=${Date.now()}`);
    }

    /**
     * Cancel a scheduled camera reconnect.
     * @returns {void}
     */
    clearCameraReconnect() {
        if (this._cameraRetryTimer) {
            clearTimeout(this._cameraRetryTimer);
            this._cameraRetryTimer = null;
        }
    }

    /**
     * Back off and retry the camera stream after an error, showing radar standby meanwhile.
     * Stops retrying after CAMERA_RETRY_MAX_ATTEMPTS so a machine without camera
     * hardware does not keep the browser tab loading forever.
     * @returns {void}
     */
    onCameraStreamError() {
        this.cameraImg.style.display = "none";
        const radar = document.getElementById("radarStandby");
        const setRadarLabel = (text) => {
            if (!radar) return;
            const label = radar.querySelector(".radar-label");
            if (label) label.innerHTML = `<span class="radar-dot"></span> ${text}`;
            radar.style.display = "flex";
        };

        if (!this.cameraEnabled) {
            setRadarLabel("CAMERA HARDWARE PAUSED");
            return;
        }

        this._cameraRetryAttempts = (this._cameraRetryAttempts || 0) + 1;
        if (this._cameraRetryAttempts >= CAMERA_RETRY_MAX_ATTEMPTS) {
            setRadarLabel("NO CAMERA HARDWARE — TAP 📷 TO RETRY");
            this.logEvent("PERCEPTION", "Camera stream unavailable — reconnect attempts stopped", "warning");
            return;
        }

        setRadarLabel(`CAMERA OFFLINE — RETRY ${this._cameraRetryAttempts}/${CAMERA_RETRY_MAX_ATTEMPTS}`);
        const delay = this._cameraRetryDelay || CAMERA_RETRY_BASE_MS;
        this._cameraRetryDelay = Math.min(delay * 2, CAMERA_RETRY_MAX_MS);
        this.clearCameraReconnect();
        this._cameraRetryTimer = setTimeout(() => {
            this._cameraRetryTimer = null;
            if (this.cameraEnabled) this.requestCameraStream("/camera/stream");
        }, delay);
    }

    /**
     * Phone header: the persona and engine suite collapses into a second row that
     * the ENGINE control opens, so the bar never degrades into unlabelled glyphs.
     * @returns {void}
     */
    initEngineCluster() {
        const btn = document.getElementById("btnEngineCluster");
        const closeBtn = document.getElementById("btnCloseEngineCluster");
        if (!btn) return;
        btn.addEventListener("click", () => {
            this.setEngineClusterOpen(!document.body.classList.contains("engine-cluster-open"));
        });
        if (closeBtn) closeBtn.addEventListener("click", () => this.setEngineClusterOpen(false));
    }

    /**
     * @param {boolean} open - Show the persona/engine row under the header bar.
     * @returns {void}
     */
    setEngineClusterOpen(open) {
        document.body.classList.toggle("engine-cluster-open", open);
        const btn = document.getElementById("btnEngineCluster");
        if (btn) btn.setAttribute("aria-expanded", String(open));
    }

    /**
     * One owner for every modal-style overlay. Modals open and close through
     * inline `display` at a dozen call sites, so this watches the DOM instead of
     * each call: focus moves in, Tab stays inside, the page behind becomes inert,
     * scrolling is locked, and focus returns to the control that opened it.
     * @returns {void}
     */
    initOverlayManager() {
        this._overlayLayers = [...document.querySelectorAll(".cyber-modal-backdrop, .api-auth-gate")];
        this._activeOverlay = null;
        this._overlayReturnFocus = null;

        const observer = new MutationObserver(() => this.syncOverlayState());
        this._overlayLayers.forEach((layer) => {
            observer.observe(layer, { attributes: true, attributeFilter: ["style", "class", "hidden"] });
        });
        this._overlayObserver = observer;

        document.addEventListener("keydown", (e) => this.onOverlayKeydown(e), true);
    }

    /**
     * @returns {HTMLElement|null} Topmost visible overlay layer.
     */
    openOverlayLayer() {
        const open = (this._overlayLayers || []).filter(
            (layer) => getComputedStyle(layer).display !== "none"
        );
        return open[open.length - 1] || null;
    }

    /**
     * @param {HTMLElement} layer
     * @returns {HTMLElement[]} Focusable elements in tab order.
     */
    overlayTabbables(layer) {
        return [...layer.querySelectorAll(
            'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]),' +
            'select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )].filter((el) => el.offsetParent !== null);
    }

    /**
     * Reconcile focus, inertness and scroll lock with which layer is on top.
     * @returns {void}
     */
    syncOverlayState() {
        const top = this.openOverlayLayer();
        if (top === this._activeOverlay) return;
        const previous = this._activeOverlay;
        this._activeOverlay = top;

        if (top && !previous) {
            this._overlayReturnFocus = document.activeElement;
            // Compensate for the scrollbar the lock removes, so nothing shifts.
            const gap = window.innerWidth - document.documentElement.clientWidth;
            if (gap > 0) document.body.style.paddingRight = `${gap}px`;
            document.documentElement.classList.add("overlay-open");
            document.body.classList.add("overlay-open");
        }

        (this._overlayLayers || []).forEach((layer) => {
            const visible = getComputedStyle(layer).display !== "none";
            if (visible && layer !== top) layer.setAttribute("inert", "");
            else layer.removeAttribute("inert");
        });

        if (top) {
            ["header.top-nav", "main.main-viewport", "#chatSidebar", "#chatSidebarBackdrop"]
                .forEach((sel) => document.querySelector(sel)?.setAttribute("inert", ""));
            this.focusOverlay(top);
            return;
        }

        document.documentElement.classList.remove("overlay-open");
        document.body.classList.remove("overlay-open");
        document.body.style.paddingRight = "";
        ["header.top-nav", "main.main-viewport", "#chatSidebar", "#chatSidebarBackdrop"]
            .forEach((sel) => document.querySelector(sel)?.removeAttribute("inert"));
        const returnTo = this._overlayReturnFocus;
        this._overlayReturnFocus = null;
        if (returnTo && document.contains(returnTo) && typeof returnTo.focus === "function") {
            returnTo.focus({ preventScroll: true });
        }
    }

    /**
     * @param {HTMLElement} layer
     * @returns {void}
     */
    focusOverlay(layer) {
        if (layer.contains(document.activeElement)) return;
        const target = [...layer.querySelectorAll("input:not([type=hidden]), textarea")].find(
            (el) => el.offsetParent !== null
        ) || layer.querySelector(".cyber-modal, .lightbox-modal");
        if (target && !target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
        target?.focus({ preventScroll: true });
    }

    /**
     * Escape closes the topmost overlay through its own close control so each
     * surface keeps its cleanup; Tab never leaves an open layer.
     * @param {KeyboardEvent} e
     * @returns {void}
     */
    onOverlayKeydown(e) {
        const top = this.openOverlayLayer();
        if (e.key === "Tab" && top) {
            const items = this.overlayTabbables(top);
            if (!items.length) {
                e.preventDefault();
                return;
            }
            const [first] = items;
            const last = items[items.length - 1];
            const active = document.activeElement;
            if (!top.contains(active)) {
                e.preventDefault();
                first.focus();
            } else if (e.shiftKey && (active === first || active === top)) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && active === last) {
                e.preventDefault();
                first.focus();
            }
            return;
        }
        if (e.key !== "Escape") return;
        if (top) {
            const closeBtn = top.querySelector(".modal-close-btn, .lightbox-close-btn");
            if (closeBtn) closeBtn.click();
            return;
        }
        if (document.body.classList.contains("engine-cluster-open")) {
            this.setEngineClusterOpen(false);
            return;
        }
        if (this.chatSidebar?.classList.contains("mobile-open") && this.btnToggleChatSidebar) {
            this.btnToggleChatSidebar.click();
        }
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
                    await this.apiFetch("/interrupt", { method: "POST" });
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

        // ─── Camera Box Controls & Hardware Toggle ─────────────────────────
        const cameraBox = document.getElementById("cameraBox");
        const btnToggleCam = document.getElementById("btnToggleCamera");
        const btnReset = document.getElementById("btnStreamReset");
        const btnExpand = document.getElementById("btnStreamExpand");
        const resizeHandle = document.getElementById("resizeHandle");
        this.cameraEnabled = true;

        if (btnToggleCam) {
            /**
             * Handles camera toggle button click to pause/resume hardware video stream.
             * @param {MouseEvent} e - Click event.
             * @returns {Promise<void>}
             */
            btnToggleCam.addEventListener("click", async (e) => {
                e.stopPropagation();
                const targetState = !this.cameraEnabled;
                try {
                    const res = await this.apiFetch("/camera-mode", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ enabled: targetState }),
                    });
                    if (res.ok) {
                        const data = await res.json();
                        this.cameraEnabled = data.camera_enabled;
                        const radar = document.getElementById("radarStandby");
                        if (this.cameraEnabled) {
                            btnToggleCam.classList.remove("muted");
                            btnToggleCam.title = "Camera Active. Click to pause camera and release hardware.";
                            btnToggleCam.textContent = "📷";
                            if (radar) radar.style.display = "none";
                            this.armCameraStream("/camera/stream");
                            this.logEvent("PERCEPTION", "Camera stream resumed", "system");
                        } else {
                            btnToggleCam.classList.add("muted");
                            btnToggleCam.title = "Camera Paused. Click to enable camera.";
                            btnToggleCam.textContent = "🚫";
                            this.clearCameraReconnect();
                            if (this.cameraImg) {
                                this.cameraImg.src = "";
                                this.cameraImg.style.display = "none";
                            }
                            if (radar) {
                                radar.style.display = "flex";
                                const rLabel = radar.querySelector(".radar-label");
                                if (rLabel) rLabel.innerHTML = '<span class="radar-dot"></span> CAMERA HARDWARE PAUSED';
                            }
                            this.logEvent("PERCEPTION", "Camera stream paused (hardware released)", "system");
                        }
                    }
                } catch (err) {
                    console.error("Failed to toggle camera:", err);
                }
            });
        }

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

        // ─── Face Mesh Toggle ────────────────────────────────────────────────
        const btnFaceMesh = document.getElementById("btnToggleFaceMesh");
        /** @type {boolean} Whether the face mesh MJPEG stream is currently active. */
        this.faceMeshActive = false;

        if (btnFaceMesh) {
            // Attempt to check initial face mesh status from the server
            this.apiFetch("/api/face-mesh/status")
                .then((r) => r.json())
                .then((d) => {
                    this.faceMeshActive = d.available && d.running;
                    this._updateFaceMeshBtn(btnFaceMesh);
                })
                .catch(() => {});

            /**
             * Toggles the camera image source between the raw annotated stream
             * (/camera/stream) and the MediaPipe 468-point face mesh stream
             * (/camera/face-mesh), and calls the server toggle endpoint.
             * @param {MouseEvent} e - Click event.
             * @returns {Promise<void>}
             */
            btnFaceMesh.addEventListener("click", async (e) => {
                e.stopPropagation();
                try {
                    const targetEnabled = !this.faceMeshActive;
                    const res = await this.apiFetch("/api/face-mesh/toggle", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ enabled: targetEnabled }),
                    });
                    if (res.ok) {
                        const data = await res.json();
                        this.faceMeshActive = data.running && data.available;
                        this._updateFaceMeshBtn(btnFaceMesh);
                        if (this.cameraImg) {
                            if (this.faceMeshActive) {
                                this.armCameraStream("/camera/face-mesh");
                                this.logEvent("PERCEPTION", "Face Mesh stream active — 468-point overlay", "system");
                            } else {
                                this.armCameraStream("/camera/stream");
                                this.logEvent("PERCEPTION", "Face Mesh disabled — annotated stream", "system");
                            }
                        }
                        // Update stream overlay label
                        const overlayEl = document.querySelector(".stream-overlay");
                        if (overlayEl) {
                            overlayEl.innerHTML = this.faceMeshActive
                                ? '<span class="rec-dot"></span> LIVE MAX FPS | FACE MESH 🟢'
                                : '<span class="rec-dot"></span> LIVE MAX FPS | YUNET HUD';
                        }
                    }
                } catch (err) {
                    console.error("Failed to toggle face mesh:", err);
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

    /**
     * Update the face mesh toggle button appearance to reflect active/inactive state.
     *
     * Active state: green glow with "🟢" indicator and MESH label.
     * Inactive state: neutral ghost style with "🫥" indicator.
     *
     * @param {HTMLButtonElement} btn - The face mesh toggle button element.
     * @returns {void}
     */
    _updateFaceMeshBtn(btn) {
        if (!btn) return;
        if (this.faceMeshActive) {
            btn.textContent = "🟢";
            btn.classList.add("active-mesh");
            btn.title = "Face Mesh ACTIVE — Click to disable (MediaPipe 468-point overlay)";
        } else {
            btn.textContent = "🫥";
            btn.classList.remove("active-mesh");
            btn.title = "Toggle Face Mesh (MediaPipe 468-point overlay)";
        }
    }

    async initPersonalitySelector() {
        if (!this.personaSelect) return;

        // Fetch available personalities and current active
        try {
            const res = await this.apiFetch("/api/personas");
            if (res.ok) {
                const data = await res.json();
                const personas = data.personas || [];
                if (personas.length > 0) {
                    this.personaSelect.innerHTML = "";
                    personas.forEach((p) => {
                        const opt = document.createElement("option");
                        opt.value = p.id;
                        opt.textContent = (p.display_name || p.id).toUpperCase();
                        this.personaSelect.appendChild(opt);
                    });
                }
                const savedPersona = localStorage.getItem("robot_active_personality");
                if (savedPersona && savedPersona !== data.active && personas.some(p => p.id === savedPersona)) {
                    await this.apiFetch(`/api/personas/${savedPersona}/activate`, { method: "POST" });
                    data.active = savedPersona;
                }
                if (data.active) {
                    this.personaSelect.value = data.active;
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
                const switchRes = await this.apiFetch(`/api/personas/${chosen}/activate`, { method: "POST" });
                if (switchRes.ok) {
                    this.logEvent("PERSONA", `Personality switched to ${disp} with isolated database`, "system");
                    await this.updateActivePersonaUI(chosen, disp);
                } else {
                    // Fallback to legacy personality switch
                    await this.apiFetch(`/personality/${chosen}`, { method: "POST" });
                    await this.updateActivePersonaUI(chosen, disp);
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
     * Refresh the top-navigation persona selector dropdown with the latest available personas.
     * @returns {Promise<void>}
     */
    async refreshPersonaSelector() {
        if (!this.personaSelect) return;
        try {
            const res = await this.apiFetch("/api/personas");
            if (!res.ok) return;
            const data = await res.json();
            const personas = data.personas || [];
            const activeKey = data.active || "neo";

            this.personaSelect.innerHTML = "";
            personas.forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.id;
                opt.textContent = (p.display_name || p.id).toUpperCase();
                this.personaSelect.appendChild(opt);
            });
            this.personaSelect.value = activeKey;
        } catch (e) {}
    }

    /**
     * Initialize the AI Persona Vault & Isolated Database Manager modal,
     * handling card rendering, isolated database statistics, per-persona default model presets,
     * new persona creation with automated DB provisioning, and persona deletion.
     * @returns {void}
     */
    initPersonaManager() {
        const btnOpen = document.getElementById("btnOpenPersonaModal");
        const modal = document.getElementById("personaModal");
        const btnClose = document.getElementById("btnClosePersonaModal");
        const btnFooterClose = document.getElementById("btnFooterClosePersonaModal");
        const tabList = document.getElementById("tabPersonaList");
        const tabCreate = document.getElementById("tabPersonaCreate");
        const panelList = document.getElementById("panelPersonaList");
        const panelCreate = document.getElementById("panelPersonaCreate");
        const cardsGrid = document.getElementById("personaCardsGrid");
        const formCreate = document.getElementById("formCreatePersona");
        const ttsEngineSelect = document.getElementById("personaSelectTtsEngine");
        const ttsVoiceSelect = document.getElementById("personaSelectTtsVoice");
        const colorInput = document.getElementById("personaInputColor");
        const colorCode = document.getElementById("colorPreviewCode");
        const statusMsg = document.getElementById("personaFormStatusMsg");
        const btnCancel = document.getElementById("btnCancelPersonaForm");
        const totalCount = document.getElementById("personaTotalCount");
        const activeBadge = document.getElementById("personaModalActiveBadge");

        if (!btnOpen || !modal) return;

        // Populate voice dropdown based on engine
        const updateTtsVoices = (engine) => {
            if (!ttsVoiceSelect) return;
            ttsVoiceSelect.innerHTML = "";
            if (engine === "kokoro") {
                const kokoroVoices = [
                    { id: "af_heart", label: "Heart (af_heart - Warm, Affectionate)" },
                    { id: "af_bella", label: "Bella (af_bella - Energetic, Cheerful)" },
                    { id: "af_nicole", label: "Nicole (af_nicole - Clear, Articulate)" },
                    { id: "af_sky", label: "Sky (af_sky - Melodic, Soft)" },
                    { id: "af_sarah", label: "Sarah (af_sarah - Professional)" },
                    { id: "am_adam", label: "Adam (am_adam - Warm, Confident)" },
                    { id: "am_echo", label: "Echo (am_echo - Resonant, Tech)" },
                    { id: "am_michael", label: "Michael (am_michael - Calm, Smooth)" },
                    { id: "bf_emma", label: "Emma (bf_emma - British Female)" },
                    { id: "bm_george", label: "George (bm_george - British Male)" },
                ];
                kokoroVoices.forEach((v) => {
                    const opt = document.createElement("option");
                    opt.value = v.id;
                    opt.textContent = v.label;
                    ttsVoiceSelect.appendChild(opt);
                });
            } else {
                const piperVoices = [
                    { id: "en-us-amy-medium", label: "Amy Medium (Piper en-US)" },
                    { id: "en_US-lessac-medium", label: "Lessac Medium (Piper en-US)" },
                    { id: "en-us-kathleen-low", label: "Kathleen Low (Piper en-US)" },
                    { id: "en-us-hfc_female-medium", label: "HFC Female (Piper en-US)" },
                    { id: "en-us-libritts-high", label: "LibriTTS High (Piper en-US)" },
                ];
                piperVoices.forEach((v) => {
                    const opt = document.createElement("option");
                    opt.value = v.id;
                    opt.textContent = v.label;
                    ttsVoiceSelect.appendChild(opt);
                });
            }
        };

        if (ttsEngineSelect) {
            ttsEngineSelect.addEventListener("change", (e) => {
                updateTtsVoices(e.target.value);
            });
            updateTtsVoices(ttsEngineSelect.value || "kokoro");
        }

        if (colorInput && colorCode) {
            colorInput.addEventListener("input", (e) => {
                colorCode.textContent = e.target.value;
            });
        }

        // Format byte sizes for isolated databases
        const formatBytes = (bytes) => {
            if (!bytes || bytes === 0) return "0 B";
            const k = 1024;
            const sizes = ["B", "KB", "MB", "GB"];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
        };

        // Render personas
        const renderPersonas = async () => {
            if (!cardsGrid) return;
            try {
                const res = await this.apiFetch("/api/personas");
                if (!res.ok) throw new Error("Failed to load personas");
                const data = await res.json();
                const personas = data.personas || [];
                const activeKey = data.active || "neo";

                if (totalCount) totalCount.textContent = personas.length;
                if (activeBadge) {
                    const activeP = personas.find(p => p.id === activeKey);
                    activeBadge.textContent = `ACTIVE: ${(activeP ? (activeP.display_name || activeP.id) : activeKey).toUpperCase()}`;
                }

                cardsGrid.innerHTML = "";
                personas.forEach((p) => {
                    const card = document.createElement("div");
                    card.className = `persona-card ${p.is_active ? "is-active" : ""}`;
                    const defModels = p.default_models || {};
                    const avatarColor = p.avatar_color || "#00ffcc";
                    const initial = (p.display_name || p.id).charAt(0).toUpperCase();

                    card.innerHTML = `
                        <div class="persona-card-header">
                            <div class="persona-card-title-group">
                                <div class="persona-card-avatar" style="background: ${avatarColor}22; border-color: ${avatarColor}; color: ${avatarColor};">
                                    ${initial}
                                </div>
                                <div>
                                    <div class="persona-card-name">${this.escapeHtml(p.display_name || p.id)}</div>
                                    <div class="persona-card-id">ID: @${this.escapeHtml(p.id)}</div>
                                </div>
                            </div>
                            ${p.is_active ? '<span class="persona-card-badge badge-active">ACTIVE</span>' : ''}
                        </div>
                        <div class="persona-card-desc">${this.escapeHtml(p.description || "No description provided.")}</div>
                        <div class="persona-db-box">
                            <div class="persona-db-path" title="${this.escapeHtml(p.database_path || '')}">
                                📁 ${this.escapeHtml(p.database_path || 'data/memory/personas/' + p.id + '/conversations.db')}
                            </div>
                            <div class="persona-db-stats">
                                <span class="persona-db-stat-item">Size: <span class="persona-db-stat-val">${formatBytes(p.size_bytes)}</span></span>
                                <span class="persona-db-stat-item">Turns: <span class="persona-db-stat-val">${p.turns || 0}</span></span>
                                <span class="persona-db-stat-item">Sessions: <span class="persona-db-stat-val">${p.sessions || 0}</span></span>
                            </div>
                        </div>
                        <div class="persona-models-matrix">
                            <span class="persona-model-chip chip-llm" title="Default LLM">🧠 ${this.escapeHtml(defModels.llm || 'qwen3.5-4b')}</span>
                            <span class="persona-model-chip chip-voice" title="Default TTS Voice">🗣️ ${this.escapeHtml(defModels.tts_voice || 'default')}</span>
                            <span class="persona-model-chip chip-stt" title="Default STT">🎙️ ${this.escapeHtml(defModels.stt || 'base.en')}</span>
                            <span class="persona-model-chip chip-image" title="Default Image">🎨 ${this.escapeHtml(defModels.image || 'dreamshaper-v8')}</span>
                            ${defModels.thinking_mode ? '<span class="persona-model-chip" style="color: #c084fc; border-color: rgba(192,132,252,0.4);" title="Thinking Mode">⚡ Deep Thought</span>' : ''}
                        </div>
                        <div class="persona-card-actions">
                            ${!p.is_active ? `<button class="btn-card-activate" data-persona-id="${this.escapeHtml(p.id)}">⚡ Activate</button>` : ''}
                            <button class="btn-card-delete" data-persona-id="${this.escapeHtml(p.id)}" ${p.is_active || p.is_system ? 'disabled title="Active and built-in system personas cannot be deleted"' : 'title="Delete Persona & Isolated Database"'}>🗑️ Delete</button>
                        </div>
                    `;
                    cardsGrid.appendChild(card);
                });

                // Attach card action handlers
                cardsGrid.querySelectorAll(".btn-card-activate").forEach((btn) => {
                    btn.addEventListener("click", async (e) => {
                        const targetId = e.currentTarget.dataset.personaId;
                        try {
                            btn.textContent = "Activating...";
                            btn.disabled = true;
                            const actRes = await this.apiFetch(`/api/personas/${targetId}/activate`, { method: "POST" });
                            if (!actRes.ok) throw new Error(await actRes.text());
                            this.logEvent("PERSONA", `Activated persona ${targetId} with isolated database.`, "system");
                            await this.updateActivePersonaUI(targetId);
                            await renderPersonas();
                        } catch (err) {
                            alert(`Failed to activate persona: ${err.message}`);
                            btn.textContent = "⚡ Activate";
                            btn.disabled = false;
                        }
                    });
                });

                cardsGrid.querySelectorAll(".btn-card-delete").forEach((btn) => {
                    btn.addEventListener("click", async (e) => {
                        const targetId = e.currentTarget.dataset.personaId;
                        if (!confirm(`Are you sure you want to delete persona '@${targetId}' and its isolated database? This cannot be undone.`)) {
                            return;
                        }
                        try {
                            btn.textContent = "...";
                            btn.disabled = true;
                            const delRes = await this.apiFetch(`/api/personas/${targetId}?delete_database=true`, { method: "DELETE" });
                            if (!delRes.ok) {
                                const errJson = await delRes.json().catch(() => ({}));
                                throw new Error(errJson.detail || "Deletion failed");
                            }
                            this.logEvent("PERSONA", `Persona @${targetId} and isolated database deleted.`, "system");
                            await this.refreshPersonaSelector();
                            await renderPersonas();
                        } catch (err) {
                            alert(`Cannot delete persona: ${err.message}`);
                            btn.textContent = "🗑️ Delete";
                            btn.disabled = false;
                        }
                    });
                });

            } catch (err) {
                cardsGrid.innerHTML = `<div class="persona-load-error">Failed to load personas: ${this.escapeHtml(err.message)}</div>`;
            }
        };

        // Open & Close Modal
        const openModal = () => {
            modal.style.display = "flex";
            switchTab("list");
            renderPersonas();
        };

        const closeModal = () => {
            modal.style.display = "none";
        };

        btnOpen.addEventListener("click", openModal);
        btnClose.addEventListener("click", closeModal);
        if (btnFooterClose) btnFooterClose.addEventListener("click", closeModal);

        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeModal();
        });

        // Tab Switching
        const switchTab = (tab) => {
            if (tab === "list") {
                tabList.classList.add("active");
                tabCreate.classList.remove("active");
                panelList.style.display = "block";
                panelCreate.style.display = "none";
            } else {
                tabList.classList.remove("active");
                tabCreate.classList.add("active");
                panelList.style.display = "none";
                panelCreate.style.display = "block";
                if (statusMsg) {
                    statusMsg.textContent = "";
                    statusMsg.className = "form-status-msg";
                }
            }
        };

        tabList.addEventListener("click", () => switchTab("list"));
        tabCreate.addEventListener("click", () => switchTab("create"));
        if (btnCancel) btnCancel.addEventListener("click", () => switchTab("list"));

        // Form Submit
        if (formCreate) {
            formCreate.addEventListener("submit", async (e) => {
                e.preventDefault();
                const idInput = document.getElementById("personaInputId");
                const nameInput = document.getElementById("personaInputDisplayName");
                const descInput = document.getElementById("personaInputDesc");
                const promptInput = document.getElementById("personaInputPrompt");
                const styleInput = document.getElementById("personaInputStyle");
                const llmSelect = document.getElementById("personaSelectLlm");
                const sttSelect = document.getElementById("personaSelectStt");
                const imgSelect = document.getElementById("personaSelectImage");
                const thinkInput = document.getElementById("personaInputThinking");

                const personaId = (idInput ? idInput.value : "").trim().toLowerCase();
                const displayName = (nameInput ? nameInput.value : "").trim();
                const prompt = (promptInput ? promptInput.value : "").trim();

                if (!personaId || !displayName || !prompt) {
                    if (statusMsg) {
                        statusMsg.textContent = "Please fill in ID, Display Name, and System Prompt.";
                        statusMsg.className = "form-status-msg error";
                    }
                    return;
                }

                const payload = {
                    id: personaId,
                    display_name: displayName,
                    description: descInput ? descInput.value.trim() : "",
                    system_prompt: prompt,
                    speaking_style: styleInput ? styleInput.value.trim() : "",
                    avatar_color: colorInput ? colorInput.value : "#00ffcc",
                    default_models: {
                        llm: llmSelect ? llmSelect.value : "qwen3.5-4b",
                        tts_engine: ttsEngineSelect ? ttsEngineSelect.value : "kokoro",
                        tts_voice: ttsVoiceSelect ? ttsVoiceSelect.value : "af_heart",
                        stt: sttSelect ? sttSelect.value : "base.en",
                        image: imgSelect ? imgSelect.value : "dreamshaper-v8",
                        thinking_mode: thinkInput ? thinkInput.checked : false,
                    },
                };

                const submitBtn = document.getElementById("btnSubmitPersonaForm");
                try {
                    if (submitBtn) {
                        submitBtn.textContent = "Provisioning Database...";
                        submitBtn.disabled = true;
                    }
                    const createRes = await this.apiFetch("/api/personas", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });
                    if (!createRes.ok) {
                        const errJson = await createRes.json().catch(() => ({}));
                        throw new Error(errJson.detail || "Creation failed");
                    }
                    this.logEvent("PERSONA", `Persona @${personaId} created and isolated database provisioned!`, "system");
                    formCreate.reset();
                    if (colorInput && colorCode) colorCode.textContent = colorInput.value;
                    await this.refreshPersonaSelector();
                    switchTab("list");
                    await renderPersonas();
                } catch (err) {
                    if (statusMsg) {
                        statusMsg.textContent = `Error: ${err.message}`;
                        statusMsg.className = "form-status-msg error";
                    }
                } finally {
                    if (submitBtn) {
                        submitBtn.textContent = "✨ Provision Persona & Database";
                        submitBtn.disabled = false;
                    }
                }
            });
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

        // Local GGUF model switcher controls (top header only)
        const topNavLocal = document.getElementById("topNavLocalControls");

        // OpenAI model scanning controls & modal (top header only)
        const topNavOpenai = document.getElementById("topNavOpenaiControls");
        const btnTopNavModelPicker = document.getElementById("btnTopNavModelPicker");

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
         * Publish the active OpenAI-compatible model into the top header and provider toggle.
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

            if (sub && this.currentLlmProvider === "openai_compatible") {
                sub.textContent = `${m} • CACHED`;
            }
        };

        /**
         * Update the top header provider toggle and its model-suite controls for the active LLM provider.
         * @param {string} prov - 'local', 'gemini', or 'openai_compatible'.
         * @param {string|null} [mode] - Optional mode string from status endpoint.
         * @param {boolean} [isTransitioning=false] - Whether a switch is currently executing.
         * @param {string|null} [targetProv=null] - Desired target provider during transition.
         * @returns {void}
         */
        this.updateLlmToggleUi = (prov, mode, isTransitioning = false, targetProv = null) => {
            if (isTransitioning) {
                if (btn) btn.classList.add("switching");
                if (badge) badge.textContent = "WAIT";

                if (targetProv === "gemini") {
                    if (label) label.textContent = "CONNECTING GEMINI...";
                    if (sub) sub.textContent = "FREEING 3.4GB VRAM";
                } else if (targetProv === "openai_compatible") {
                    if (label) label.textContent = "CONNECTING OPENAI...";
                    if (sub) sub.textContent = "FREEING 3.4GB VRAM";
                } else {
                    if (label) label.textContent = "STARTING LOCAL...";
                    if (sub) sub.textContent = "ALLOCATING VRAM";
                }
                return;
            }

            if (btn) btn.classList.remove("switching");

            const safeProv = prov || "local";
            this.currentLlmProvider = safeProv;

            const activeM = this.activeOpenaiModel || "gpt-4o";

            if (safeProv === "openai_compatible") {
                if (btn) {
                    btn.classList.remove("gemini-mode", "local-mode");
                    btn.classList.add("openai-mode");
                }
                if (topNavOpenai) topNavOpenai.style.display = "inline-flex";
                if (topNavLocal) topNavLocal.style.display = "none";

                if (icon) icon.textContent = "🌐";
                if (label) label.textContent = "OPENAI COMPAT";
                if (sub) sub.textContent = `${activeM} • CACHED`;
                if (badge) badge.textContent = "OPENAI";
                if (btn) btn.title = `Active: OpenAI-Compatible API (${activeM}, Prompt Caching). Click to cycle to Local Qwen (RTX 3050).`;

                this.updateOpenaiModelDisplays(activeM);
            } else if (safeProv === "gemini") {
                if (btn) {
                    btn.classList.remove("local-mode", "openai-mode");
                    btn.classList.add("gemini-mode");
                }
                if (topNavOpenai) topNavOpenai.style.display = "none";
                if (topNavLocal) topNavLocal.style.display = "none";

                if (icon) icon.textContent = "✨";
                if (label) label.textContent = "GEMINI (CLOUD)";
                if (sub) sub.textContent = "1M CONTEXT • HIGH INTEL";
                if (badge) badge.textContent = "GEMINI";
                if (btn) btn.title = "Active: Cloud Gemini 3.5 Flash Lite (1M Context, High Intelligence). Click to swap to OpenAI API.";
            } else {
                if (btn) {
                    btn.classList.remove("gemini-mode", "openai-mode");
                    btn.classList.add("local-mode");
                }
                if (topNavOpenai) topNavOpenai.style.display = "none";
                if (topNavLocal) topNavLocal.style.display = "inline-flex";

                const ctxK = Math.round((this.localContextWindow || 16384) / 1024);
                if (icon) icon.textContent = "⚡";
                if (label) label.textContent = "LOCAL (RTX 3050)";
                if (sub) sub.textContent = `LOW LATENCY • ${ctxK}K`;
                if (badge) badge.textContent = "LOCAL";
                if (btn) btn.title = `Active: Local Qwen 3.5 4B on RTX 3050 (Low Latency, ${ctxK}K Context). Click to swap to Cloud Gemini.`;
                if (typeof this.updateLocalModelDisplays === "function" && this.activeLocalModelId) {
                    this.updateLocalModelDisplays(this.activeLocalModelId);
                }
            }
        };

        /**
         * Scan available models from the OpenAI-compatible endpoint.
         * @param {boolean} [autoSelectCurrent=true] - Whether to synchronize active model selection.
         * @param {string|null} [preset=null] - Optional preset target ('openrouter', 'aicredits', etc.).
         * @returns {Promise<Array<Object>>}
         */
        this.scanOpenaiModels = async (autoSelectCurrent = true, preset = null) => {
            const scanIcons = [
                document.getElementById("modalRescanIcon")
            ].filter(Boolean);
            scanIcons.forEach((ic) => ic.classList.add("spinning"));

            const scanButtons = [
                document.getElementById("btnModalRescanModels")
            ].filter(Boolean);
            scanButtons.forEach((b) => b.disabled = true);

            try {
                let url = "/api/llm/openai/models";
                if (preset) {
                    url += `?preset=${encodeURIComponent(preset)}`;
                }
                const res = await this.apiFetch(url);
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
                const res = await this.apiFetch("/api/llm/openai/model", {
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

        // Wire the model picker trigger in the top header
        if (btnTopNavModelPicker) btnTopNavModelPicker.addEventListener("click", () => this.openModelSearchModal());
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
                const res = await this.apiFetch("/api/llm/status");
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

        const saveConfig = async () => {
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
                    const res = await this.apiFetch("/api/llm/openai/config", {
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
                            modalFeedback.textContent = `Error: ${data.message || data.error || `HTTP ${res.status}`}`;
                            modalFeedback.className = "cyberdeck-modal-feedback error";
                        }
                    }
                } catch (err) {
                    if (modalFeedback) {
                        modalFeedback.textContent = `Network error: ${err.message}`;
                        modalFeedback.className = "cyberdeck-modal-feedback error";
                    }
                }
        };

        if (btnSaveConfig) btnSaveConfig.addEventListener("click", saveConfig);

        /* The endpoint dialog is one form, so Enter in any field saves exactly
           once and the password inputs are no longer orphaned. */
        const configForm = document.getElementById("openaiConfigForm");
        if (configForm) {
            configForm.addEventListener("submit", (e) => {
                e.preventDefault();
                saveConfig();
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
            const res = await this.apiFetch("/api/llm/status");
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
                const res = await this.apiFetch("/api/llm/provider", {
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
        // Humanoid natural asynchronous blinking (all timers tracked + stopped on teardown)
        const doBlink = () => {
            if (this._stopped) return;
            const eyes = document.querySelectorAll(".eye");
            eyes.forEach(e => e.classList.add("blink"));
            this._timeout(() => {
                eyes.forEach(e => e.classList.remove("blink"));
            }, 120);

            // Double blink chance
            if (Math.random() < 0.25) {
                this._timeout(() => {
                    eyes.forEach(e => e.classList.add("blink"));
                    this._timeout(() => eyes.forEach(e => e.classList.remove("blink")), 110);
                }, 240);
            }

            const nextBlink = Math.random() * 3500 + 2000;
            this.blinkInterval = this._timeout(doBlink, nextBlink);
        };
        this.blinkInterval = this._timeout(doBlink, 2000);
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
        if (this.mouthInterval) this._clearIv(this.mouthInterval);
        this.mouthInterval = this._interval(() => {
            this.mouthBars.forEach((bar, idx) => {
                const height = Math.floor(Math.random() * 38) + 8;
                bar.style.height = `${height}px`;
            });
        }, 75);
    }

    stopMouthSpeaking() {
        if (this.mouthInterval) {
            this._clearIv(this.mouthInterval);
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

        this._wsBackoffMs = RobotUI.WS_BACKOFF_MIN_MS;
        this._wsFailures = 0;
        this._wsConnectTimer = null;

        const connect = () => {
            this._wsConnectTimer = null;
            let socket;
            try {
                // streamUrl() appends ?token= when a bearer token is stored.
                socket = new WebSocket(this.streamUrl(wsUrl));
            } catch (e) {
                console.warn("WebSocket construction failed:", e);
                this._scheduleWsReconnect(connect);
                return;
            }
            this.ws = socket;

            this.ws.onopen = () => {
                // Successful open resets the backoff ladder.
                this._wsBackoffMs = RobotUI.WS_BACKOFF_MIN_MS;
                this._wsFailures = 0;
                this._setWsOfflineIndicator(false);
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

            this.ws.onerror = () => {
                // onclose always follows; reconnection is handled there.
            };

            this.ws.onclose = () => {
                this._scheduleWsReconnect(connect);
            };
        };

        connect();
    }

    /**
     * Exponential-backoff WebSocket reconnect: 1s → 2s → 4s … capped at 30s, with
     * jitter, reset to 1s on a successful open. After WS_MAX_FAST_RETRIES failures
     * the UI shows an offline indicator and only a slow probe is kept alive, so a
     * dead backend is never hammered.
     * @param {Function} connect - Opens the next WebSocket attempt.
     * @returns {void}
     */
    _scheduleWsReconnect(connect) {
        if (this._stopped) return;
        if (this._wsConnectTimer) this._clearTmo(this._wsConnectTimer);

        this._wsFailures++;
        const exhausted = this._wsFailures >= RobotUI.WS_MAX_FAST_RETRIES;
        if (exhausted) {
            this._setWsOfflineIndicator(true, `retries every ${RobotUI.WS_SLOW_RETRY_MS / 1000}s`);
            this.logEvent("WS", `Telemetry stream offline after ${this._wsFailures} attempts; slow retries only.`, "warning");
        } else {
            this.logEvent("WS", `Telemetry stream closed; reconnect attempt ${this._wsFailures}.`, "system");
        }

        const base = Math.min(RobotUI.WS_BACKOFF_MAX_MS, this._wsBackoffMs || RobotUI.WS_BACKOFF_MIN_MS);
        // ±25% jitter so many clients do not reconnect in lockstep.
        const jitter = base * 0.25 * (Math.random() - 0.5) * 2;
        const delay = exhausted ? RobotUI.WS_SLOW_RETRY_MS : Math.max(500, Math.round(base + jitter));
        this._wsBackoffMs = Math.min(RobotUI.WS_BACKOFF_MAX_MS, base * 2);

        this._wsConnectTimer = this._timeout(() => connect(), delay);
    }

    handleServerEvent(data) {
        if (data.event === "audio_waveform") {
            this.latestWaveform = data;
            return;
        }

        if (data.event === "llm.provider_changed" && this.updateLlmToggleUi) {
            this.currentLlmProvider = data.provider;
            this.updateLlmToggleUi(data.provider, data.mode);
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
            const disp = (typeof this.formatPersonaName === "function")
                ? this.formatPersonaName(chosen)
                : (chosen === "neo" ? "CYPHER" : chosen.toUpperCase());
            this.updateActivePersonaUI(chosen, disp);
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
            if (this._phraseTimeout) this._clearTmo(this._phraseTimeout);
            this._phraseTimeout = this._timeout(() => {
                this._phraseTimeout = null;
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

        const budgetBar = document.getElementById("contextBudgetBar");
        const budgetRatio = document.getElementById("contextTokenRatio");
        const tokSpeedElem = document.getElementById("activeTokSpeed");

        if (metrics.tok_per_sec !== undefined && tokSpeedElem) {
            tokSpeedElem.textContent = `${Math.round(metrics.tok_per_sec)} T/s`;
        }

        if (budgetBar || budgetRatio) {
            let maxCtx = 16384;
            if (this.currentLlmProvider === "gemini") {
                maxCtx = 1000000;
            } else if (this.currentLlmProvider === "openai") {
                maxCtx = 128000;
            } else if (this.activeLocalModelContext) {
                maxCtx = this.activeLocalModelContext;
            }

            const currentTokens = (metrics.tokens_in || 0) + (metrics.tokens_out || 0) || (typeof metrics.tokens === "number" ? metrics.tokens : 0) || 842;
            const pct = Math.min(100, Math.max(0.5, (currentTokens / maxCtx) * 100));

            if (budgetRatio) {
                budgetRatio.textContent = `${currentTokens.toLocaleString()} / ${maxCtx >= 1000000 ? `${(maxCtx / 1000000).toFixed(0)}M` : `${Math.round(maxCtx / 1024)}k`} TOKENS (${pct.toFixed(1)}%)`;
            }
            if (budgetBar) {
                budgetBar.style.width = `${pct}%`;
                if (pct > 85) {
                    budgetBar.style.background = "linear-gradient(90deg, #ff0055, #ff5500)";
                } else if (pct > 60) {
                    budgetBar.style.background = "linear-gradient(90deg, #ffaa00, #ffdd00)";
                } else {
                    budgetBar.style.background = "linear-gradient(90deg, var(--matrix-bright), var(--primary-cyan))";
                }
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
        // Poll status for robust state synchronization. This data also arrives over
        // /ws/events, so while the socket is OPEN the poll is slowed to a periodic
        // reconciliation tick; it only runs at full rate as a disconnected fallback.
        this._interval(async () => {
            this._pollCounters.status++;
            if (this._isWsOpen() && this._pollCounters.status % RobotUI.WS_POLL_SLOWDOWN !== 0) return;
            try {
                const res = await this.apiFetch("/status");
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
                        if (this.personaSelect && document.activeElement !== this.personaSelect && this.personaSelect.value !== data.personality) {
                            this.personaSelect.value = data.personality;
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
                        metrics.tokens_in = tIn;
                        metrics.tokens_out = tOut;
                        if (t.LLM_TOKENS_PER_SEC && t.LLM_TOKENS_PER_SEC.last) {
                            metrics.tok_per_sec = t.LLM_TOKENS_PER_SEC.last;
                        }
                        this.updateMetrics(metrics);
                    }
                }
            } catch (e) {}
        }, 1200);

        // Poll Traces & Pipeline State. trace_updated / pipeline_updated are also
        // pushed over /ws/events, so this is throttled while the socket is live.
        this._interval(async () => {
            this._pollCounters.traces++;
            if (this._isWsOpen() && this._pollCounters.traces % RobotUI.WS_POLL_SLOWDOWN !== 0) return;
            try {
                const res = await this.apiFetch("/observability/traces?limit=1");
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

        // Poll World State & Perception (not delivered over /ws/events, so always polled)
        this._interval(async () => {
            try {
                const res = await this.apiFetch("/world-state");
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

        // Camera stream recovery: /camera/stream is a continuous MJPEG feed, so a dead
        // endpoint surfaces as an image error. Back off and give up instead of retrying
        // forever — an endless retry keeps the browser tab spinning.
        if (this.cameraImg) {
            this.cameraImg.onerror = () => this.onCameraStreamError();
            this.cameraImg.onload = () => {
                if (!this.cameraEnabled) return;
                this.clearCameraReconnect();
                this._cameraRetryAttempts = 0;
                this._cameraRetryDelay = CAMERA_RETRY_BASE_MS;
                this.cameraImg.style.display = "block";
                const radar = document.getElementById("radarStandby");
                if (radar) radar.style.display = "none";
            };
        }

        // Hardware Observability Telemetry Polling (CPU, System RAM, RTX 3050, AMD 680M)
        const fetchHw = async () => {
            try {
                const res = await this.apiFetch("/observability/hardware");
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
        this._interval(fetchHw, 2000);

        this._interval(async () => {
            try {
                const res = await this.apiFetch("/images/jobs");
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
        const resSelect = document.getElementById("imgResSelect");
        const viewportLabel = document.getElementById("canvasViewportLabel");

        /* The canvas HUD states the resolution the next job will use; leaving it
           at 512×512 while the Size control says 1024 is a lie the user can see. */
        const syncViewportLabel = () => {
            if (!resSelect || !viewportLabel) return;
            const [w, h] = resSelect.value.split("x");
            viewportLabel.textContent = `AI VIEWPORT • ${w}×${h} PX`;
        };

        if (resSelect) {
            resSelect.addEventListener("change", syncViewportLabel);
            syncViewportLabel();
        }

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
            this.apiFetch("/api/images/models")
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
                const res = await this.apiFetch("/images/generate", {
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
                        const pollRes = await this.apiFetch("/images/jobs");
                        if (pollRes.ok) {
                            const jobsData = await pollRes.json();
                            if (jobsData.jobs) this.updateImageJobsUI(jobsData.jobs);
                        }
                    } catch (e) {}
                } else {
                    const err = data.error || (data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : `HTTP ${res.status}`);
                    this.logEvent("IMAGE", `Image generation error: ${err}`, "system");
                    if (btnGen) {
                        btnGen.textContent = "Generate";
                        btnGen.classList.remove("generating");
                    }
                    this.syncImageComposer();
                    this.updatePipelineStatus({ IMAGE: "ERROR" });
                }
            } catch (e) {
                console.error("Image gen failed", e);
                this.logEvent("IMAGE", `Failed to send generation request: ${e.message}`, "system");
                if (btnGen) {
                    btnGen.textContent = "Generate";
                    btnGen.classList.remove("generating");
                }
                this.syncImageComposer();
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
            input.addEventListener("input", () => this.syncImageComposer());
            input.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    submitImage();
                }
            });
            this.syncImageComposer();
        }

        if (btnCancel) {
            btnCancel.addEventListener("click", async (e) => {
                e.preventDefault();
                if (this.activeImageJob) {
                    await this.apiFetch(`/images/cancel/${this.activeImageJob}`, { method: "POST" });
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
                    this.syncImageComposer();
                }
            });
        });

        const previewImg = document.getElementById("imgPreviewImg");
        if (previewImg) {
            previewImg.style.cursor = "zoom-in";
            // Mark as bound so updateImageJobsUI() (called on every jobs poll) never adds a second handler.
            previewImg._boundClick = true;
            previewImg.addEventListener("click", () => {
                if (previewImg.src) {
                    window.open(previewImg.src, "_blank");
                }
            });
        }
    }

    /**
     * Keep the studio's Generate button honest: it is unavailable while a job
     * is in flight and whenever the prompt is blank.
     * @returns {void}
     */
    syncImageComposer() {
        const btnGen = document.getElementById("btnGenImage");
        const input = document.getElementById("imgPromptInput");
        if (!btnGen || !input) return;
        btnGen.disabled = Boolean(this.activeImageJob) || input.value.trim() === "";
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
                btnGen.textContent = "Generate";
                btnGen.classList.remove("generating");
            }
            this.syncImageComposer();
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
                if (this.loadGalleryImages) {
                    this.loadGalleryImages(false);
                }
            } else {
                if (placeholder) placeholder.style.display = "block";
                if (previewImg) previewImg.style.display = "none";
                if (metaBlock) metaBlock.style.display = "none";
            }
        }
    }

    /**
     * Initialize the AI Image Studio Artwork Gallery modal and full-resolution Lightbox.
     * Fetches artworks from /api/images/gallery, supports filtering by model,
     * search filtering, modal open/close, full-res preview, download, copy prompt, and deletion.
     * @returns {void}
     */
    initImageGallery() {
        const btnOpen = document.getElementById("btnOpenGallery");
        const modal = document.getElementById("imageGalleryModal");
        const btnClose = document.getElementById("btnCloseGalleryModal");
        const btnFooterClose = document.getElementById("btnFooterCloseGalleryModal");
        const btnRefresh = document.getElementById("btnRefreshGallery");
        const filterChips = document.getElementById("galleryFilterChips");
        const searchInput = document.getElementById("gallerySearchInput");

        // Lightbox elements
        const lightboxModal = document.getElementById("galleryLightboxModal");
        const btnCloseLightbox = document.getElementById("btnCloseLightbox");
        const lbCopyPrompt = document.getElementById("lightboxCopyPromptBtn");
        const lbDelete = document.getElementById("lightboxDeleteBtn");

        this.galleryImages = [];
        this.activeGalleryFilter = "all";
        this.gallerySearchQuery = "";
        this.activeLightboxImage = null;

        // Open Gallery Modal
        if (btnOpen && modal) {
            btnOpen.addEventListener("click", () => {
                modal.style.display = "flex";
                this.loadGalleryImages();
            });
        }

        // Close Gallery Modal
        const closeModal = () => {
            if (modal) modal.style.display = "none";
        };
        if (btnClose) btnClose.addEventListener("click", closeModal);
        if (btnFooterClose) btnFooterClose.addEventListener("click", closeModal);
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) closeModal();
            });
        }

        // Refresh Gallery
        if (btnRefresh) {
            btnRefresh.addEventListener("click", () => {
                this.loadGalleryImages();
            });
        }

        const btnRetryGallery = document.getElementById("btnRetryGallery");
        if (btnRetryGallery) {
            btnRetryGallery.addEventListener("click", () => this.loadGalleryImages());
        }

        // Filter Chips
        if (filterChips) {
            filterChips.addEventListener("click", (e) => {
                const chip = e.target.closest(".gallery-filter-chip");
                if (!chip) return;
                filterChips.querySelectorAll(".gallery-filter-chip").forEach(c => c.classList.remove("active"));
                chip.classList.add("active");
                this.activeGalleryFilter = chip.dataset.filter || "all";
                this.renderGalleryGrid();
            });
        }

        // Search Input
        if (searchInput) {
            searchInput.addEventListener("input", (e) => {
                this.gallerySearchQuery = (e.target.value || "").toLowerCase().trim();
                this.renderGalleryGrid();
            });
        }

        // Close Lightbox
        const closeLightbox = () => {
            if (lightboxModal) lightboxModal.style.display = "none";
            this.activeLightboxImage = null;
        };
        if (btnCloseLightbox) btnCloseLightbox.addEventListener("click", closeLightbox);
        if (lightboxModal) {
            lightboxModal.addEventListener("click", (e) => {
                if (e.target === lightboxModal) closeLightbox();
            });
        }

        // Copy Prompt
        if (lbCopyPrompt) {
            lbCopyPrompt.addEventListener("click", () => {
                if (this.activeLightboxImage && this.activeLightboxImage.prompt) {
                    navigator.clipboard.writeText(this.activeLightboxImage.prompt).then(() => {
                        const originalText = lbCopyPrompt.textContent;
                        lbCopyPrompt.textContent = "✓ Copied!";
                        setTimeout(() => { lbCopyPrompt.textContent = originalText; }, 1800);
                    }).catch(() => {});
                }
            });
        }

        // Delete Image
        if (lbDelete) {
            lbDelete.addEventListener("click", async () => {
                if (!this.activeLightboxImage) return;
                const fname = this.activeLightboxImage.filename;
                if (!confirm(`Are you sure you want to permanently delete '${fname}'?`)) return;

                try {
                    const res = await this.apiFetch(`/api/images/gallery/${encodeURIComponent(fname)}`, {
                        method: "DELETE"
                    });
                    const data = await res.json();
                    if (data.status === "ok") {
                        closeLightbox();
                        this.loadGalleryImages();
                    } else {
                        alert(`Failed to delete image: ${data.detail || data.message || "Unknown error"}`);
                    }
                } catch (err) {
                    alert(`Error deleting image: ${err.message}`);
                }
            });
        }

        // Initial load of gallery count
        this.loadGalleryImages(false);
    }

    /**
     * Switch the gallery body between its loading, ready, error and empty faces.
     * @param {"loading"|"ready"|"error"|"empty"} state - Which face to show.
     * @param {string} [message] - Detail line for the error face.
     * @returns {void}
     */
    setGalleryState(state, message = "") {
        const grid = document.getElementById("galleryGrid");
        if (!grid) return;
        const loading = document.getElementById("galleryLoadingState");
        const error = document.getElementById("galleryErrorState");
        const empty = document.getElementById("galleryEmptyState");
        const errorText = document.getElementById("galleryErrorText");

        if (loading) loading.hidden = state !== "loading";
        if (error) error.hidden = state !== "error";
        if (empty) empty.hidden = state !== "empty";
        grid.hidden = state !== "ready";

        if (state === "error" && errorText) {
            errorText.textContent = message || "The image service did not respond.";
        }
    }

    /**
     * Load image gallery items from the server API.
     * @param {boolean} [render=true] - Whether to re-render the gallery grid.
     * @returns {Promise<void>}
     */
    async loadGalleryImages(render = true) {
        if (render) this.setGalleryState("loading");
        try {
            const res = await this.apiFetch("/api/images/gallery?limit=150");
            if (!res.ok) throw new Error(`Gallery request failed (HTTP ${res.status})`);
            const data = await res.json();
            if (data.status !== "ok" || !Array.isArray(data.images)) {
                throw new Error(data.message || "The gallery returned an unexpected response.");
            }
            this.galleryImages = data.images;
            const total = data.total || this.galleryImages.length;

            const countBadge = document.getElementById("galleryCountBadge");
            if (countBadge) countBadge.textContent = total;

            const modalBadge = document.getElementById("modalGalleryCountBadge");
            if (modalBadge) modalBadge.textContent = `${total} ${total === 1 ? "ARTWORK" : "ARTWORKS"}`;

            const countAll = document.getElementById("galleryCountAll");
            if (countAll) countAll.textContent = total;

            if (render) {
                this.renderGalleryGrid();
            }
        } catch (err) {
            console.warn("Failed to load gallery images:", err);
            if (render) {
                const friendly = err instanceof TypeError
                    ? "Couldn’t reach the image service. Check that the API is running."
                    : err.message;
                this.setGalleryState("error", friendly);
            }
        }
    }

    /**
     * Condense an epoch timestamp into a card-sized age label.
     * @param {number} epoch - Unix timestamp in seconds.
     * @returns {string} Compact age such as "now", "12m", "3h", "4d", "6w" or "2y".
     */
    relativeStamp(epoch) {
        if (!Number.isFinite(epoch)) return "";
        const secs = Date.now() / 1000 - epoch;
        if (secs < 90) return "now";
        const mins = Math.floor(secs / 60);
        if (mins < 60) return `${mins}m`;
        const hours = Math.floor(mins / 60);
        if (hours < 24) return `${hours}h`;
        const days = Math.floor(hours / 24);
        if (days < 7) return `${days}d`;
        if (days < 365) return `${Math.floor(days / 7)}w`;
        return `${Math.floor(days / 365)}y`;
    }

    /**
     * Render the filtered gallery images into the DOM grid.
     * @returns {void}
     */
    renderGalleryGrid() {
        const grid = document.getElementById("galleryGrid");
        if (!grid) return;

        let filtered = this.galleryImages || [];

        // Model filter
        if (this.activeGalleryFilter && this.activeGalleryFilter !== "all") {
            const filterKey = this.activeGalleryFilter.toLowerCase();
            filtered = filtered.filter(img => {
                const m = (img.model || "").toLowerCase();
                const fn = (img.filename || "").toLowerCase();
                if (filterKey === "flux") return m.includes("flux") || fn.includes("flux");
                if (filterKey === "sd-turbo") return m.includes("sd-turbo") || fn.includes("sd_turbo");
                if (filterKey === "dreamshaper") return m.includes("dreamshaper") || fn.includes("dreamshaper");
                if (filterKey === "sdxl") return m.includes("sdxl") || fn.includes("sdxl");
                if (filterKey === "small-sd") return m.includes("small-sd") || fn.includes("small_sd");
                if (filterKey === "fast-procedural") return m.includes("procedural") || m.includes("fast");
                return m.includes(filterKey);
            });
        }

        // Search query filter
        if (this.gallerySearchQuery) {
            const q = this.gallerySearchQuery;
            filtered = filtered.filter(img => {
                const prompt = (img.prompt || "").toLowerCase();
                const model = (img.model || "").toLowerCase();
                const fn = (img.filename || "").toLowerCase();
                return prompt.includes(q) || model.includes(q) || fn.includes(q);
            });
        }

        if (filtered.length === 0) {
            grid.innerHTML = "";
            const hasArtwork = (this.galleryImages || []).length > 0;
            const title = document.getElementById("galleryEmptyTitle");
            const sub = document.getElementById("galleryEmptySub");
            const icon = document.getElementById("galleryEmptyIcon");
            if (title) title.textContent = hasArtwork ? "No matches" : "Nothing saved yet";
            if (sub) {
                sub.textContent = hasArtwork
                    ? "No artwork matches this filter and search. Clear them to see everything."
                    : "Generate visual artwork using the Image Studio to populate your persistent gallery.";
            }
            if (icon) icon.textContent = hasArtwork ? "🔍" : "🎨";
            this.setGalleryState("empty");
            return;
        }

        this.setGalleryState("ready");
        grid.innerHTML = filtered.map(img => {
            const promptEscaped = (img.prompt || "Generated Artwork")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
            const modelLabel = (img.model || "diffusion").toUpperCase();
            return `
                <div class="gallery-card" data-filename="${img.filename}" role="button" tabindex="0" aria-label="Open ${promptEscaped}">
                    <div class="gallery-thumb-wrap">
                        <img src="${img.url}" alt="${promptEscaped}" class="gallery-thumb" loading="lazy" decoding="async" />
                        <span class="gallery-card-badge">${modelLabel}</span>
                    </div>
                    <div class="gallery-card-info">
                        <div class="gallery-card-prompt" title="${promptEscaped}">${promptEscaped}</div>
                        <div class="gallery-card-meta">
                            <span class="gallery-card-size">${img.size_kb || ""}</span>
                            <span class="gallery-card-age" title="${img.created_at_iso || ""}">${this.relativeStamp(img.created_at)}</span>
                        </div>
                    </div>
                </div>
            `;
        }).join("");

        // Attach click and keyboard listeners to cards
        grid.querySelectorAll(".gallery-card").forEach(card => {
            const open = () => {
                const fname = card.dataset.filename;
                const imgItem = this.galleryImages.find(i => i.filename === fname);
                if (imgItem) {
                    this.openGalleryLightbox(imgItem);
                }
            };
            card.addEventListener("click", open);
            card.addEventListener("keydown", (e) => {
                if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    open();
                }
            });
        });
    }

    /**
     * Open the full-resolution artwork lightbox viewer modal.
     * @param {Object} img - Artwork metadata item.
     * @returns {void}
     */
    openGalleryLightbox(img) {
        this.activeLightboxImage = img;
        const modal = document.getElementById("galleryLightboxModal");
        const lbImg = document.getElementById("lightboxImage");
        const lbModel = document.getElementById("lightboxModelBadge");
        const lbDim = document.getElementById("lightboxDimensionsBadge");
        const lbSize = document.getElementById("lightboxSizeBadge");
        const lbPrompt = document.getElementById("lightboxPromptText");
        const lbDate = document.getElementById("lightboxDateText");
        const lbDownload = document.getElementById("lightboxDownloadBtn");

        if (lbImg) lbImg.src = img.url;
        if (lbModel) lbModel.textContent = (img.model || "DIFFUSION").toUpperCase();
        if (lbDim) lbDim.textContent = `${img.width || 512}×${img.height || 512}`;
        if (lbSize) lbSize.textContent = img.size_kb || "";
        if (lbPrompt) lbPrompt.textContent = img.prompt || "Generated Artwork";
        if (lbDate) lbDate.textContent = img.created_at_iso || "";
        if (lbDownload) {
            lbDownload.href = img.url;
            lbDownload.download = img.filename;
        }

        if (modal) modal.style.display = "flex";
    }

    /**
     * Poll HuggingFace model download progress and update telemetry in HUD buttons.
     * @returns {void}
     */
    initHfDownloadMonitor() {
        const updateDownloadBadges = async () => {
            try {
                const res = await this.apiFetch("/api/llm/hf/downloads");
                const data = await res.json();
                if (data.status === "ok" && Array.isArray(data.downloads)) {
                    const activeDownload = data.downloads.find(d => d.status === "downloading" || d.status === "in_progress");
                    const hfBtn = document.getElementById("btnTopNavLocalPicker");
                    if (activeDownload && hfBtn) {
                        const pct = Math.round(activeDownload.progress_percent || 0);
                        let badge = hfBtn.querySelector(".hf-dl-live-badge");
                        if (!badge) {
                            badge = document.createElement("span");
                            badge.className = "hf-dl-live-badge badge badge-accent";
                            badge.style.marginLeft = "6px";
                            badge.style.fontSize = "0.7rem";
                            hfBtn.appendChild(badge);
                        }
                        badge.textContent = `📥 ${pct}%`;
                    } else if (hfBtn) {
                        const badge = hfBtn.querySelector(".hf-dl-live-badge");
                        if (badge) badge.remove();
                    }
                }
            } catch (err) {
                // Silently ignore network hiccups during download polling
            }
        };

        this._interval(updateDownloadBadges, 4000);
        updateDownloadBadges();
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
                    const res = await this.apiFetch("/api/equalizer", {
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
                const res = await this.apiFetch("/api/music/status");
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
        this._interval(refreshMusicState, 5000);

        // 3. Volume slider (Direct REST endpoint)
        if (volSlider) {
            volSlider.addEventListener("input", () => {
                if (volVal) volVal.textContent = `${volSlider.value}%`;
            });
            volSlider.addEventListener("change", async () => {
                const volPct = parseInt(volSlider.value, 10);
                if (volVal) volVal.textContent = `${volPct}%`;
                try {
                    await this.apiFetch("/api/music/volume", {
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
                    const res = await this.apiFetch("/api/music/restart", { method: "POST" });
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
                    const res = await this.apiFetch("/api/music/toggle", { method: "POST" });
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
            const wsInput = this.chatTextarea;
            const wsSendBtn = this.btnChatSubmit;

            if (document.body.classList.contains("mode-chat") && wsInput && wsSendBtn) {
                wsInput.value = prompt;
                wsInput.dispatchEvent(new Event("input", { bubbles: true }));
                wsSendBtn.click();
            } else if (dualInput && dualSendBtn) {
                dualInput.value = prompt;
                dualInput.dispatchEvent(new Event("input", { bubbles: true }));
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
                if (this._stopped) return;
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
                const res = await this.apiFetch("/api/memory/stats");
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
                    this.apiFetch(searchUrl).then(r => r.json()),
                    q ? this.apiFetch(`/history/search?query=${encodeURIComponent(q)}&limit=6`).then(r => r.json()) : Promise.resolve({ results: [] })
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
            if (this._stopped) return;
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
     * Initialize Local LLM Model switcher modal, HuggingFace Hub search, and background downloader.
     * Allows hot-swapping local GGUF models on the RTX 3050 CUDA GPU and downloading new models from HuggingFace.
     * @returns {Promise<void>}
     */
    async initLocalLlmModelControls() {
        const topNavPicker = document.getElementById("btnTopNavLocalPicker");
        const topNavLabel = document.getElementById("topNavLocalPickerLabel");
        const modal = document.getElementById("localModelModal");
        const btnClose = document.getElementById("btnCloseLocalModelModal");
        const btnCancel = document.getElementById("btnCancelLocalModelModal");
        const modalHint = document.getElementById("localModelModalHint");

        // Tab Navigation Elements
        const tabBtnInstalled = document.getElementById("tabBtnLocalInstalled");
        const tabBtnHfHub = document.getElementById("tabBtnLocalHfHub");
        const tabBtnSettings = document.getElementById("tabBtnLocalSettings");
        const tabContentInstalled = document.getElementById("tabContentLocalInstalled");
        const tabContentHfHub = document.getElementById("tabContentLocalHfHub");
        const tabContentSettings = document.getElementById("tabContentLocalSettings");
        const btnGoToHfHub = document.getElementById("btnGoToHfHub");
        const installedCountBadge = document.getElementById("localInstalledCountBadge");
        const hfActiveDownloadsBadge = document.getElementById("hfActiveDownloadsBadge");
        const hfActiveDownloadsCount = document.getElementById("hfActiveDownloadsCount");

        // Installed Tab Elements
        const cardsGrid = document.getElementById("localModelCardsGrid");

        // HuggingFace Hub Tab Elements
        const hfSearchInput = document.getElementById("hfSearchInput");
        const btnHfSearchClear = document.getElementById("btnHfSearchClear");
        const btnHfSearchSubmit = document.getElementById("btnHfSearchSubmit");
        const hfPresetChips = document.querySelectorAll(".hf-preset-chip");
        const hfDownloadsPanel = document.getElementById("hfDownloadsPanel");
        const hfDownloadsList = document.getElementById("hfDownloadsList");
        const hfDownloadsStatusSub = document.getElementById("hfDownloadsStatusSub");
        const hfResultsStatus = document.getElementById("hfResultsStatus");
        const hfSearchResultsGrid = document.getElementById("hfSearchResultsGrid");

        this.activeLocalModelId = "qwen3.5-4b";
        this.localModelsList = [];
        this.activeHfDownloads = [];
        this.hfDownloadPollInterval = null;
        this.completedHfDownloadIds = new Set();

        /**
         * Switch between Installed Models, HuggingFace Hub, and LM Studio Settings tabs.
         * @param {"installed" | "hf" | "settings"} tabId - Target tab identifier.
         * @returns {void}
         */
        this.switchLocalModelModalTab = (tabId) => {
            if (tabId === "settings") {
                if (tabBtnInstalled) tabBtnInstalled.classList.remove("active");
                if (tabBtnHfHub) tabBtnHfHub.classList.remove("active");
                if (tabBtnSettings) tabBtnSettings.classList.add("active");
                if (tabContentInstalled) tabContentInstalled.style.display = "none";
                if (tabContentHfHub) tabContentHfHub.style.display = "none";
                if (tabContentSettings) tabContentSettings.style.display = "block";
                if (modalHint) modalHint.textContent = "Live configuration for llama-server process on RTX 3050 CUDA.";
                if (typeof this.loadLlamaSettings === "function") {
                    this.loadLlamaSettings();
                }
            } else if (tabId === "hf") {
                if (tabBtnInstalled) tabBtnInstalled.classList.remove("active");
                if (tabBtnSettings) tabBtnSettings.classList.remove("active");
                if (tabBtnHfHub) tabBtnHfHub.classList.add("active");
                if (tabContentInstalled) tabContentInstalled.style.display = "none";
                if (tabContentSettings) tabContentSettings.style.display = "none";
                if (tabContentHfHub) tabContentHfHub.style.display = "block";
                if (modalHint) modalHint.textContent = "Search, discover, and stream GGUF model quants directly into models/llm/.";
                if (hfSearchInput && !hfSearchInput.value) {
                    this.searchHuggingFace("instruct");
                }
            } else {
                if (tabBtnHfHub) tabBtnHfHub.classList.remove("active");
                if (tabBtnSettings) tabBtnSettings.classList.remove("active");
                if (tabBtnInstalled) tabBtnInstalled.classList.add("active");
                if (tabContentHfHub) tabContentHfHub.style.display = "none";
                if (tabContentSettings) tabContentSettings.style.display = "none";
                if (tabContentInstalled) tabContentInstalled.style.display = "block";
                if (modalHint) modalHint.textContent = "Hot-swapping pre-warms slot cache and updates active model.";
                this.refreshLocalModels();
            }
        };

        if (tabBtnInstalled) tabBtnInstalled.addEventListener("click", () => this.switchLocalModelModalTab("installed"));
        if (tabBtnHfHub) tabBtnHfHub.addEventListener("click", () => this.switchLocalModelModalTab("hf"));
        if (tabBtnSettings) tabBtnSettings.addEventListener("click", () => this.switchLocalModelModalTab("settings"));
        if (btnGoToHfHub) btnGoToHfHub.addEventListener("click", () => this.switchLocalModelModalTab("hf"));

        /**
         * Open Local Model Modal and render current cards.
         * @param {"installed" | "hf" | "settings"} [initialTab="installed"] - Initial tab to activate.
         * @returns {Promise<void>}
         */
        this.openLocalModelModal = async (initialTab = "installed") => {
            if (!modal) return;
            modal.style.display = "flex";
            modal.classList.add("open");
            this.switchLocalModelModalTab(initialTab);
            if (initialTab === "installed") {
                await this.refreshLocalModels();
            }
            await this.pollHfDownloads();
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

        if (topNavPicker) topNavPicker.addEventListener("click", () => this.openLocalModelModal("installed"));
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
            try {
                const res = await this.apiFetch("/api/llm/models");
                if (!res.ok) return;
                const data = await res.json();
                this.localModelsList = data.local_models || [];
                if (data.active_local_model) {
                    this.activeLocalModelId = data.active_local_model;
                    this.updateLocalModelDisplays(this.activeLocalModelId);
                }
                if (installedCountBadge) {
                    installedCountBadge.textContent = this.localModelsList.length.toString();
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
                const vramEst = m.parameters === "4B" ? "~3.4 GB" : (m.parameters === "2B" ? "~2.3 GB" : (m.parameters === "0.5B" ? "~0.8 GB" : (m.parameters === "1B" || m.parameters === "1.2B" || m.parameters === "1.5B" ? "~1.2 GB" : "~2.8 GB")));

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
                const res = await this.apiFetch("/api/models/local/switch", {
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
         * Update the top header GGUF model picker label for the active local model.
         * @param {string} modelId - Active local model identifier.
         * @returns {void}
         */
        this.updateLocalModelDisplays = (modelId) => {
            const shortName = (modelId || "").replace(".gguf", "").replace("-Q4_K_M", "").replace("qwen", "Qwen ").replace("llama", "Llama ");
            if (topNavLabel) topNavLabel.textContent = shortName;
        };

        // ─── HuggingFace Hub Search & Downloader Logic ───────────────────────

        /**
         * Search HuggingFace Hub for GGUF models and display results.
         * @param {string} query - Search term.
         * @returns {Promise<void>}
         */
        this.searchHuggingFace = async (query) => {
            if (!hfSearchResultsGrid) return;
            const cleanQuery = (query || "").trim();
            if (hfResultsStatus) {
                hfResultsStatus.innerHTML = `<span>⚡ Querying HuggingFace Hub for "<strong>${cleanQuery || "popular"}</strong>" GGUFs...</span>`;
            }

            try {
                const res = await this.apiFetch(`/api/llm/hf/search?q=${encodeURIComponent(cleanQuery)}&limit=24`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                const models = data.models || [];

                if (models.length === 0) {
                    if (hfResultsStatus) {
                        hfResultsStatus.innerHTML = `<span>No GGUF models found on HuggingFace Hub matching "<strong>${cleanQuery}</strong>".</span>`;
                    }
                    hfSearchResultsGrid.innerHTML = "";
                    return;
                }

                if (hfResultsStatus) {
                    hfResultsStatus.innerHTML = `<span>Found <strong>${models.length}</strong> GGUF repositories matching "${cleanQuery || "popular"}". Select a quantization below to download directly to <code>models/llm/</code>:</span>`;
                }
                this.renderHfModelCards(models);
            } catch (err) {
                if (hfResultsStatus) {
                    hfResultsStatus.innerHTML = `<span style="color: #ef4444;">Error searching HuggingFace Hub: ${err.message}</span>`;
                }
            }
        };

        /**
         * Render HuggingFace repository search result cards with quantization selectors.
         * @param {Array<object>} models - Array of model metadata objects.
         * @returns {void}
         */
        this.renderHfModelCards = (models) => {
            if (!hfSearchResultsGrid) return;
            hfSearchResultsGrid.innerHTML = "";

            models.forEach(m => {
                const card = document.createElement("div");
                card.className = "hf-model-card";
                card.dataset.repoId = m.repo_id;

                const downloadsFmt = m.downloads >= 1000 ? `${(m.downloads / 1000).toFixed(1)}k` : m.downloads;
                const likesFmt = m.likes >= 1000 ? `${(m.likes / 1000).toFixed(1)}k` : m.likes;

                let badgeHtml = "";
                if (m.is_vision) badgeHtml += '<span class="hf-badge badge-vision">VISION VLM</span>';
                if (m.is_uncensored) badgeHtml += '<span class="hf-badge badge-uncensored">UNCENSORED</span>';
                if (m.parameters && m.parameters !== "Unknown") badgeHtml += `<span class="hf-badge">${m.parameters}</span>`;
                badgeHtml += '<span class="hf-badge">GGUF</span>';

                card.innerHTML = `
                    <div class="hf-card-header">
                        <div class="hf-card-title-group">
                            <span class="hf-card-repo-title">${m.name || m.repo_id}</span>
                            <span class="hf-card-author">${m.author || m.repo_id.split('/')[0]}</span>
                        </div>
                        <div class="hf-card-badges">${badgeHtml}</div>
                    </div>
                    <div class="hf-card-meta">
                        <span>⬇ ${downloadsFmt} downloads</span>
                        <span>♥ ${likesFmt}</span>
                        <span>ID: ${m.repo_id}</span>
                    </div>
                    <div class="hf-file-picker-section">
                        <div class="hf-file-picker-label">
                            <span>QUANTIZATION / GGUF FILE:</span>
                            <span class="hf-loading-files-label" id="hfLoading_${m.repo_id.replace(/[^a-zA-Z0-9]/g, '_')}">⏳ Loading files...</span>
                        </div>
                        <select class="hf-quant-select" id="hfSelect_${m.repo_id.replace(/[^a-zA-Z0-9]/g, '_')}" disabled>
                            <option value="">Inspecting repository files...</option>
                        </select>
                        <div class="hf-file-meta-row" id="hfMetaRow_${m.repo_id.replace(/[^a-zA-Z0-9]/g, '_')}">
                            <span class="hf-size-info">—</span>
                            <span class="hf-vram-pill">—</span>
                        </div>
                    </div>
                    <div class="hf-card-footer">
                        <button class="btn-hf-download" id="hfBtnDl_${m.repo_id.replace(/[^a-zA-Z0-9]/g, '_')}" type="button" disabled>
                            <span>⬇</span> <span>DOWNLOAD GGUF</span>
                        </button>
                    </div>
                `;

                hfSearchResultsGrid.appendChild(card);

                // Asynchronously fetch repository files for this model card
                this.loadHfModelCardFiles(m);
            });
        };

        /**
         * Fetch GGUF files for a model card and populate its quantization dropdown.
         * @param {object} modelMeta - Model metadata object.
         * @returns {Promise<void>}
         */
        this.loadHfModelCardFiles = async (modelMeta) => {
            const safeId = modelMeta.repo_id.replace(/[^a-zA-Z0-9]/g, "_");
            const selectEl = document.getElementById(`hfSelect_${safeId}`);
            const btnEl = document.getElementById(`hfBtnDl_${safeId}`);
            const loadingLabel = document.getElementById(`hfLoading_${safeId}`);
            const metaRow = document.getElementById(`hfMetaRow_${safeId}`);

            try {
                const res = await this.apiFetch(`/api/llm/hf/files?repo_id=${encodeURIComponent(modelMeta.repo_id)}`);
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                const files = (data.files || []).filter(f => !f.is_mmproj);

                if (loadingLabel) loadingLabel.style.display = "none";
                if (!selectEl || !btnEl) return;

                if (files.length === 0) {
                    selectEl.innerHTML = '<option value="">No standalone GGUF files in main branch</option>';
                    return;
                }

                selectEl.innerHTML = "";
                files.forEach(f => {
                    const opt = document.createElement("option");
                    opt.value = f.filename;
                    opt.dataset.size = f.size_bytes;
                    opt.dataset.sizeFormatted = f.size_formatted;
                    opt.dataset.isDownloaded = f.is_downloaded ? "true" : "false";
                    opt.dataset.fitsVram = f.fits_vram ? "true" : "false";
                    opt.dataset.isRecommended = f.is_recommended ? "true" : "false";

                    const recTag = f.is_recommended ? "★ RECOMMENDED" : "";
                    const dlTag = f.is_downloaded ? "✓ (INSTALLED)" : "";
                    opt.textContent = `${f.filename} [${f.size_formatted}] ${recTag} ${dlTag}`.trim();
                    selectEl.appendChild(opt);
                });

                selectEl.disabled = false;
                btnEl.disabled = false;

                /**
                 * Update button state and size indicator when selected quantization changes.
                 */
                const updateSelectionState = () => {
                    const selectedOpt = selectEl.options[selectEl.selectedIndex];
                    if (!selectedOpt) return;

                    const isDownloaded = selectedOpt.dataset.isDownloaded === "true";
                    const fitsVram = selectedOpt.dataset.fitsVram === "true";
                    const sizeFmt = selectedOpt.dataset.sizeFormatted || "—";

                    if (metaRow) {
                        metaRow.innerHTML = `
                            <span class="hf-size-info">Size: <strong>${sizeFmt}</strong></span>
                            <span class="${fitsVram ? "hf-vram-ok" : "hf-vram-warn"}">
                                ${fitsVram ? "✓ Fits in RTX 3050 (4GB VRAM)" : "⚠️ High VRAM: Uses partial CPU offload"}
                            </span>
                        `;
                    }

                    if (isDownloaded) {
                        btnEl.className = "btn-hf-download is-installed";
                        btnEl.innerHTML = '<span>⚡</span> <span>ACTIVATE / LOAD MODEL</span>';
                        btnEl.onclick = () => {
                            // Find matching model ID
                            const stem = selectedOpt.value.replace(/\.gguf$/i, "");
                            this.switchLocalModel(stem);
                        };
                    } else {
                        btnEl.className = "btn-hf-download";
                        btnEl.innerHTML = `<span>⬇</span> <span>DOWNLOAD (${sizeFmt})</span>`;
                        btnEl.onclick = () => {
                            this.startHfDownload(
                                modelMeta.repo_id,
                                selectedOpt.value,
                                modelMeta.name,
                                modelMeta.parameters,
                                modelMeta.is_vision
                            );
                        };
                    }
                };

                selectEl.addEventListener("change", updateSelectionState);
                updateSelectionState();

            } catch (err) {
                if (loadingLabel) {
                    loadingLabel.textContent = `Error: ${err.message}`;
                    loadingLabel.style.color = "#ef4444";
                }
            }
        };

        /**
         * Start background download of a HuggingFace GGUF model.
         * @param {string} repoId - Model repository ID.
         * @param {string} filename - Target GGUF filename.
         * @param {string} [name] - Model display title.
         * @param {string} [parameters] - Parameter count.
         * @param {boolean} [isVision=false] - Vision support flag.
         * @returns {Promise<void>}
         */
        this.startHfDownload = async (repoId, filename, name, parameters, isVision = false) => {
            this.showDynamicIsland("HuggingFace Download", `Starting download of ${filename}...`, "⬇", "cyan", 4000);
            this.logEvent("HF_HUB", `Enqueuing download for ${filename} from ${repoId}...`, "system");

            try {
                const res = await this.apiFetch("/api/llm/hf/download", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        repo_id: repoId,
                        filename: filename,
                        name: name,
                        parameters: parameters,
                        is_vision: isVision,
                    }),
                });
                const data = await res.json();
                if (data.status === "ok") {
                    this.showDynamicIsland("Downloading GGUF", `Streaming ${filename} to models/llm/...`, "⚡", "cyan", 3000);
                    await this.pollHfDownloads();
                    // Scroll to top of tab to show active downloads
                    if (tabContentHfHub) tabContentHfHub.scrollTop = 0;
                } else {
                    this.showDynamicIsland("Download Error", data.message || "Failed to start download.", "✕", "rose", 4000);
                    this.logEvent("ERROR", `HuggingFace download failed: ${data.message}`, "error");
                }
            } catch (err) {
                this.showDynamicIsland("Network Error", err.message, "✕", "rose", 4000);
            }
        };

        /**
         * Cancel an in-progress HuggingFace download task.
         * @param {string} downloadId - Download task ID.
         * @returns {Promise<void>}
         */
        this.cancelHfDownload = async (downloadId) => {
            try {
                const res = await this.apiFetch("/api/llm/hf/cancel", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ download_id: downloadId }),
                });
                const data = await res.json();
                if (data.status === "ok") {
                    this.showDynamicIsland("Download Cancelled", "Download task stopped.", "✕", "amber", 3000);
                    await this.pollHfDownloads();
                }
            } catch (err) {
                console.error("Failed to cancel download:", err);
            }
        };

        /**
         * Poll active and completed downloads from the backend.
         * @returns {Promise<void>}
         */
        this.pollHfDownloads = async () => {
            try {
                const res = await this.apiFetch("/api/llm/hf/downloads");
                if (!res.ok) return;
                const data = await res.json();
                const downloads = data.downloads || [];
                this.activeHfDownloads = downloads;

                const activeCount = downloads.filter(d => d.status === "downloading" || d.status === "verifying" || d.status === "queued").length;

                // Update active downloads pill in modal strip
                if (hfActiveDownloadsBadge && hfActiveDownloadsCount) {
                    if (activeCount > 0) {
                        hfActiveDownloadsBadge.style.display = "inline-flex";
                        hfActiveDownloadsCount.textContent = activeCount.toString();
                    } else {
                        hfActiveDownloadsBadge.style.display = "none";
                    }
                }

                // Check for newly completed downloads to trigger auto-registration notification & refresh
                downloads.forEach(d => {
                    if (d.status === "completed" && !this.completedHfDownloadIds.has(d.download_id)) {
                        this.completedHfDownloadIds.add(d.download_id);
                        this.showDynamicIsland("Model Ready!", `✓ ${d.name} saved and registered locally!`, "🎉", "cyan", 5000);
                        this.logEvent("HF_HUB", `✓ Model ${d.name} (${d.filename}) successfully registered locally in models/llm/!`, "success");
                        this.refreshLocalModels();
                    }
                });

                this.renderHfDownloads(downloads);

                // Manage polling interval: poll every 700ms while active, otherwise every 5s
                if (activeCount > 0) {
                    if (!this.hfDownloadPollInterval) {
                        this.hfDownloadPollInterval = this._interval(() => this.pollHfDownloads(), 750);
                    }
                } else {
                    if (this.hfDownloadPollInterval) {
                        this._clearIv(this.hfDownloadPollInterval);
                        this.hfDownloadPollInterval = null;
                    }
                }
            } catch (err) {
                console.error("Failed to poll HuggingFace downloads:", err);
            }
        };

        /**
         * Render active and historical download cards inside the downloads drawer.
         * @param {Array<object>} downloads - Array of download tasks.
         * @returns {void}
         */
        this.renderHfDownloads = (downloads) => {
            if (!hfDownloadsPanel || !hfDownloadsList) return;

            if (downloads.length === 0) {
                hfDownloadsPanel.style.display = "none";
                return;
            }

            hfDownloadsPanel.style.display = "block";
            hfDownloadsList.innerHTML = "";

            downloads.forEach(d => {
                const card = document.createElement("div");
                card.className = "hf-download-card";

                const isDownloading = d.status === "downloading";
                const isCompleted = d.status === "completed";
                const isVerifying = d.status === "verifying";
                const isFailed = d.status === "failed";

                const etaFmt = d.eta_seconds > 60
                    ? `~${Math.round(d.eta_seconds / 60)}m ${(Math.round(d.eta_seconds % 60))}s`
                    : `~${Math.round(d.eta_seconds)}s`;

                card.innerHTML = `
                    <div class="hf-dl-card-top">
                        <div class="hf-dl-info">
                            <span class="hf-dl-name">${d.name || d.filename}</span>
                            <span class="hf-dl-file">${d.filename} • ${d.repo_id}</span>
                        </div>
                        <span class="hf-dl-badge status-${d.status}">${d.status}</span>
                    </div>
                    <div class="hf-dl-progress-wrap">
                        <div class="hf-progress-track">
                            <div class="hf-progress-fill" style="width: ${d.percent || 0}%;"></div>
                        </div>
                        <div class="hf-dl-metrics">
                            <span><strong>${d.downloaded_formatted}</strong> / ${d.total_formatted} (${d.percent}%)</span>
                            ${isDownloading ? `<span>⚡ ${d.speed_mb_s} MB/s • ${etaFmt} remaining</span>` : ""}
                            ${isVerifying ? `<span>🔍 Verifying GGUF header & registering...</span>` : ""}
                            ${isCompleted ? `<span style="color: #34d399;">✓ Saved to models/llm/ & registered</span>` : ""}
                            ${isFailed ? `<span style="color: #f87171;">✕ ${d.error || "Download error"}</span>` : ""}
                        </div>
                    </div>
                    <div class="hf-dl-actions">
                        ${isDownloading ? `<button class="btn-hf-cancel" type="button" data-id="${d.download_id}">✕ Cancel</button>` : ""}
                        ${isCompleted ? `<button class="btn-hf-activate" type="button" data-model="${d.registered_model_id || d.filename}">⚡ LOAD & ACTIVATE NOW</button>` : ""}
                    </div>
                `;

                // Bind Cancel button
                const btnCancelDl = card.querySelector(".btn-hf-cancel");
                if (btnCancelDl) {
                    btnCancelDl.addEventListener("click", () => this.cancelHfDownload(d.download_id));
                }

                // Bind Activate button
                const btnActivateDl = card.querySelector(".btn-hf-activate");
                if (btnActivateDl) {
                    btnActivateDl.addEventListener("click", () => {
                        const targetId = btnActivateDl.dataset.model;
                        this.switchLocalModel(targetId);
                    });
                }

                hfDownloadsList.appendChild(card);
            });
        };

        // Bind Search Input & Submit
        if (btnHfSearchSubmit && hfSearchInput) {
            btnHfSearchSubmit.addEventListener("click", () => {
                this.searchHuggingFace(hfSearchInput.value);
            });
            hfSearchInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") {
                    this.searchHuggingFace(hfSearchInput.value);
                }
            });
            hfSearchInput.addEventListener("input", () => {
                if (btnHfSearchClear) {
                    btnHfSearchClear.style.display = hfSearchInput.value ? "block" : "none";
                }
            });
        }

        if (btnHfSearchClear && hfSearchInput) {
            btnHfSearchClear.addEventListener("click", () => {
                hfSearchInput.value = "";
                btnHfSearchClear.style.display = "none";
                hfSearchInput.focus();
                this.searchHuggingFace("instruct");
            });
        }

        // Bind Preset Chips
        hfPresetChips.forEach(chip => {
            chip.addEventListener("click", () => {
                const query = chip.dataset.query || chip.textContent.trim();
                if (hfSearchInput) {
                    hfSearchInput.value = query;
                    if (btnHfSearchClear) btnHfSearchClear.style.display = "block";
                }
                this.searchHuggingFace(query);
            });
        });

        /**
         * Load advanced LM Studio / llama-server hardware settings from the backend.
         * Populates all sliders, input fields, checkboxes, and quantization selectors.
         * @returns {Promise<void>}
         */
        this.loadLlamaSettings = async () => {
            try {
                const res = await this.apiFetch("/api/llm/settings");
                if (!res.ok) return;
                const data = await res.json();
                const s = data.settings || {};

                const setVal = (id, val) => {
                    const el = document.getElementById(id);
                    if (el && val !== undefined && val !== null) el.value = val;
                };
                const setChecked = (id, val) => {
                    const el = document.getElementById(id);
                    if (el && val !== undefined) el.checked = Boolean(val);
                };

                const ngl = s.n_gpu_layers !== undefined ? s.n_gpu_layers : 99;
                setVal("cfgGpuLayers", ngl);
                setVal("cfgGpuLayersNum", ngl);
                const nglBadge = document.getElementById("cfgGpuLayersVal");
                if (nglBadge) nglBadge.textContent = ngl;

                setChecked("cfgFlashAttn", s.flash_attn !== false);
                setChecked("cfgNoKvOffload", s.no_kv_offload === true);
                setVal("cfgMainGpu", s.main_gpu !== undefined ? s.main_gpu : 0);
                setVal("cfgSplitMode", s.split_mode || "none");

                setVal("cfgCtxSize", s.ctx_size || 16384);
                setVal("cfgBatchSize", s.batch_size || 4096);
                setVal("cfgUbatchSize", s.ubatch_size || 4096);
                setVal("cfgParallel", s.parallel || 1);
                setChecked("cfgContBatching", s.cont_batching !== false);

                setVal("cfgCacheTypeK", s.cache_type_k || "q4_0");
                setVal("cfgCacheTypeV", s.cache_type_v || "q4_0");

                setVal("cfgThreads", s.threads || 8);
                setVal("cfgThreadsBatch", s.threads_batch || 8);
                setChecked("cfgMlock", s.mlock === true);
                setChecked("cfgNoMmap", s.no_mmap === true);
                setVal("cfgNuma", s.numa || "none");
                setChecked("cfgCpuStrict", s.cpu_strict === true);

                setVal("cfgRopeScaling", s.rope_scaling || "none");
                setVal("cfgRopeScale", s.rope_scale || "");
                setVal("cfgMmprojPath", s.mmproj_path || "");
                setChecked("cfgMmprojOffload", s.mmproj_offload !== false);

                setVal("cfgTemp", s.temp !== undefined ? s.temp : 0.7);
                setVal("cfgTopP", s.top_p !== undefined ? s.top_p : 0.9);
                setVal("cfgTopK", s.top_k !== undefined ? s.top_k : 40);
                setVal("cfgMinP", s.min_p !== undefined ? s.min_p : 0.05);
                setVal("cfgRepeatPenalty", s.repeat_penalty !== undefined ? s.repeat_penalty : 1.1);
                setVal("cfgReasoningBudget", s.reasoning_budget !== undefined ? s.reasoning_budget : -1);
                setVal("cfgCustomCliArgs", s.custom_cli_args || "");
            } catch (err) {
                console.error("Failed to load llama-server settings:", err);
            }
        };

        // Two-way sync for GPU layer slider and numeric input
        const sliderNgl = document.getElementById("cfgGpuLayers");
        const numNgl = document.getElementById("cfgGpuLayersNum");
        const badgeNgl = document.getElementById("cfgGpuLayersVal");
        if (sliderNgl && numNgl) {
            sliderNgl.addEventListener("input", (e) => {
                numNgl.value = e.target.value;
                if (badgeNgl) badgeNgl.textContent = e.target.value;
            });
            numNgl.addEventListener("input", (e) => {
                sliderNgl.value = e.target.value;
                if (badgeNgl) badgeNgl.textContent = e.target.value;
            });
        }

        // Bind preset chips for quick layer setting
        document.querySelectorAll(".quick-preset-chips .preset-chip-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const val = btn.dataset.val;
                if (sliderNgl) sliderNgl.value = val;
                if (numNgl) numNgl.value = val;
                if (badgeNgl) badgeNgl.textContent = val;
            });
        });

        /**
         * Collect all current settings form values into an updated options dictionary.
         * @returns {Object} LlamaSettings dictionary.
         */
        const collectLlamaSettings = () => {
            const getVal = (id, def = "") => {
                const el = document.getElementById(id);
                return el ? el.value : def;
            };
            const getNum = (id, def = 0) => {
                const el = document.getElementById(id);
                return el ? (parseFloat(el.value) || def) : def;
            };
            const getChecked = (id, def = false) => {
                const el = document.getElementById(id);
                return el ? el.checked : def;
            };

            const ropeScaleStr = getVal("cfgRopeScale");
            return {
                n_gpu_layers: parseInt(getVal("cfgGpuLayers", "99"), 10),
                main_gpu: parseInt(getVal("cfgMainGpu", "0"), 10),
                split_mode: getVal("cfgSplitMode", "none"),
                flash_attn: getChecked("cfgFlashAttn", true),
                no_kv_offload: getChecked("cfgNoKvOffload", false),
                ctx_size: parseInt(getVal("cfgCtxSize", "16384"), 10),
                batch_size: parseInt(getVal("cfgBatchSize", "4096"), 10),
                ubatch_size: parseInt(getVal("cfgUbatchSize", "4096"), 10),
                parallel: parseInt(getVal("cfgParallel", "1"), 10),
                cont_batching: getChecked("cfgContBatching", true),
                cache_type_k: getVal("cfgCacheTypeK", "q4_0"),
                cache_type_v: getVal("cfgCacheTypeV", "q4_0"),
                threads: parseInt(getVal("cfgThreads", "8"), 10),
                threads_batch: parseInt(getVal("cfgThreadsBatch", "8"), 10),
                cpu_strict: getChecked("cfgCpuStrict", false),
                mlock: getChecked("cfgMlock", false),
                no_mmap: getChecked("cfgNoMmap", false),
                numa: getVal("cfgNuma", "none"),
                rope_scaling: getVal("cfgRopeScaling", "none"),
                rope_scale: ropeScaleStr ? parseFloat(ropeScaleStr) : null,
                mmproj_path: getVal("cfgMmprojPath", "").trim(),
                mmproj_offload: getChecked("cfgMmprojOffload", true),
                temp: getNum("cfgTemp", 0.7),
                top_p: getNum("cfgTopP", 0.9),
                top_k: parseInt(getVal("cfgTopK", "40"), 10),
                min_p: getNum("cfgMinP", 0.05),
                repeat_penalty: getNum("cfgRepeatPenalty", 1.1),
                reasoning_budget: parseInt(getVal("cfgReasoningBudget", "-1"), 10),
                custom_cli_args: getVal("cfgCustomCliArgs", "").trim(),
            };
        };

        /**
         * Display status feedback message in the settings modal.
         * @param {string} msg - Message to display.
         * @param {"success" | "error" | "loading"} [type="success"] - Message style type.
         * @returns {void}
         */
        const showLlamaStatus = (msg, type = "success") => {
            const box = document.getElementById("llamaSettingsStatusBox");
            const txt = document.getElementById("llamaSettingsStatusText");
            if (!box || !txt) return;
            box.style.display = "flex";
            box.className = `llama-settings-status-box ${type}`;
            txt.textContent = msg;
            if (type === "success") {
                setTimeout(() => { box.style.display = "none"; }, 5000);
            }
        };

        const btnApply = document.getElementById("btnApplyLlamaSettingsReload");
        const btnSave = document.getElementById("btnSaveLlamaSettings");
        const btnReset = document.getElementById("btnResetLlamaSettings");

        if (btnApply) {
            btnApply.addEventListener("click", async () => {
                try {
                    btnApply.disabled = true;
                    btnApply.textContent = "⚡ Restarting llama-server...";
                    showLlamaStatus("Stopping llama-server, applying hardware parameters, and offloading layers to RTX 3050...", "loading");
                    const payload = { settings: collectLlamaSettings(), reload: true };
                    const res = await this.apiFetch("/api/llm/settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });
                    const data = await res.json();
                    if (data.status === "reloaded" || data.status === "ok") {
                        showLlamaStatus("✨ Llama-server reloaded successfully with new hardware offload parameters!", "success");
                        this.logEvent("LLM", "llama-server restarted with updated hardware/offload parameters.", "system");
                        await this.refreshLlmStatus();
                    } else {
                        showLlamaStatus(`⚠️ Reload failed: ${data.message || JSON.stringify(data)}`, "error");
                    }
                } catch (e) {
                    showLlamaStatus(`❌ Error reloading llama-server: ${e.message}`, "error");
                } finally {
                    btnApply.disabled = false;
                    btnApply.textContent = "⚡ Apply & Reload Model Now";
                }
            });
        }

        if (btnSave) {
            btnSave.addEventListener("click", async () => {
                try {
                    btnSave.disabled = true;
                    const payload = { settings: collectLlamaSettings(), reload: false };
                    const res = await this.apiFetch("/api/llm/settings", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
                    });
                    if (res.ok) {
                        showLlamaStatus("💾 Settings saved successfully. They will apply on the next model switch or server start.", "success");
                    } else {
                        showLlamaStatus("⚠️ Failed to save settings.", "error");
                    }
                } catch (e) {
                    showLlamaStatus(`❌ Error saving: ${e.message}`, "error");
                } finally {
                    btnSave.disabled = false;
                }
            });
        }

        if (btnReset) {
            btnReset.addEventListener("click", async () => {
                if (!confirm("Reset all llama.cpp and offloading settings to RTX 3050 defaults and reload?")) return;
                try {
                    btnReset.disabled = true;
                    showLlamaStatus("Resetting settings to defaults and reloading...", "loading");
                    const res = await this.apiFetch("/api/llm/settings/reset", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ reload: true }),
                    });
                    const data = await res.json();
                    if (data.status === "reloaded" || data.status === "reset") {
                        showLlamaStatus("↺ Settings reset to optimal defaults and llama-server reloaded!", "success");
                        await this.loadLlamaSettings();
                        await this.refreshLlmStatus();
                    }
                } catch (e) {
                    showLlamaStatus(`❌ Reset failed: ${e.message}`, "error");
                } finally {
                    btnReset.disabled = false;
                }
            });
        }

        // Fetch initial local model status and check for any background downloads
        await this.refreshLocalModels();
        await this.pollHfDownloads();
    }

    /**
     * Initialize Whisper STT model manager, top nav trigger, audio deck chips, and modal catalog.
     * Manages whisper-server daemon on AMD Radeon 680M iGPU via Vulkan backend.
     * @returns {Promise<void>}
     */
    async initSttModelControls() {
        const topNavPicker = document.getElementById("btnTopNavSttPicker");
        const topNavLabel = document.getElementById("topNavSttPickerLabel");
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
                const res = await this.apiFetch("/api/stt/models");
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
                const res = await this.apiFetch("/api/stt/switch", {
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
         * Update the top header STT picker label and the audio deck chips for the active Whisper model.
         * @param {string} modelId - Active model ID.
         * @param {number} [sizeMb] - Model file size in MB.
         * @returns {void}
         */
        this.updateSttDisplays = (modelId, sizeMb = null) => {
            if (topNavLabel) topNavLabel.textContent = modelId;
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
        this.ttsPreviewUrl = null;
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
            this._stopTtsPreview();
            if (this.ttsActivePreviewBtn) {
                this.ttsActivePreviewBtn.classList.remove("playing");
                this.ttsActivePreviewBtn.textContent = "🔊 Preview";
                this.ttsActivePreviewBtn = null;
            }
        };

        if (topNavPicker) topNavPicker.addEventListener("click", () => this.openTtsModal());
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
                    this.apiFetch("/api/tts/status"),
                    this.apiFetch("/api/tts/voices"),
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
                this._stopTtsPreview();
                btn.classList.remove("playing");
                btn.textContent = "🔊 Preview";
                this.ttsActivePreviewBtn = null;
                return;
            }

            // Switching voices mid-playback: release the old clip (audio + blob URL) first.
            this._stopTtsPreview();
            if (this.ttsActivePreviewBtn) {
                this.ttsActivePreviewBtn.classList.remove("playing");
                this.ttsActivePreviewBtn.textContent = "🔊 Preview";
            }

            btn.classList.add("playing");
            btn.textContent = "⏳ Generating...";
            this.ttsActivePreviewBtn = btn;

            let url = null;
            let audio = null;
            // Releases this clip's blob URL even when play() rejects and no ended/error event fires.
            const cleanupPreview = () => {
                if (this.ttsPreviewAudio === audio) {
                    this._stopTtsPreview();
                } else if (url) {
                    try {
                        URL.revokeObjectURL(url);
                    } catch (_) {}
                    if (this.ttsPreviewUrl === url) this.ttsPreviewUrl = null;
                }
            };

            try {
                const sampleText = engine === "kokoro"
                    ? "Hello! This is Kokoro, an ultra-realistic neural voice running locally on the robot."
                    : "Hello! This is Piper, a high-speed local speech synthesis voice.";

                const res = await this.apiFetch("/api/tts/test", {
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
                url = URL.createObjectURL(blob);
                this.ttsPreviewUrl = url;
                audio = new Audio(url);
                this.ttsPreviewAudio = audio;

                btn.textContent = "⏹️ Stop";

                audio.onended = () => {
                    btn.classList.remove("playing");
                    btn.textContent = "🔊 Preview";
                    if (this.ttsActivePreviewBtn === btn) this.ttsActivePreviewBtn = null;
                    cleanupPreview();
                };

                audio.onerror = () => {
                    btn.classList.remove("playing");
                    btn.textContent = "🔊 Preview";
                    if (this.ttsActivePreviewBtn === btn) this.ttsActivePreviewBtn = null;
                    cleanupPreview();
                };

                await audio.play();
            } catch (err) {
                console.error("Preview failed:", err);
                cleanupPreview();
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
                const res = await this.apiFetch("/api/tts/switch", {
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

// ─── Frontend tuning constants ───────────────────────────────────────────────
RobotUI.CHAT_MAX_ROWS = 200;   // hard cap on rendered chat message rows
RobotUI.STREAM_RENDER_MS = 100; // minimum interval between streaming re-renders
RobotUI.WS_BACKOFF_MIN_MS = 1000;
RobotUI.WS_BACKOFF_MAX_MS = 30000;
RobotUI.WS_MAX_FAST_RETRIES = 8;  // then fall back to a slow probe + offline badge
RobotUI.WS_SLOW_RETRY_MS = 60000;
RobotUI.WS_POLL_SLOWDOWN = 12;    // poll every Nth tick while /ws/events is connected

/**
 * Matrix Rain Ambient Visual Effect
 */
class MatrixRain {
    constructor() {
        this.canvas = document.getElementById("matrixRainCanvas");
        this._stopped = false;
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
        if (!this.ctx || !this.canvas || this._stopped) return;

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

// The HTML ships without a camera src so the stream is only ever requested with
// credentials (cookie/bearer). Arm it after load, never before first paint.
window.addEventListener("load", () => {
    const ui = window.robotUI;
    if (ui && ui.cameraImg && ui.cameraEnabled && !ui.cameraImg.getAttribute("src")) {
        ui.armCameraStream("/camera/stream");
    }
}, { once: true });
