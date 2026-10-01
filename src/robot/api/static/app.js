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
        this.metricStt = document.getElementById("metricStt");
        this.metricTtft = document.getElementById("metricTtft");
        this.metricTts = document.getElementById("metricTts");
        this.metricE2e = document.getElementById("metricE2e");
        this.barStt = document.getElementById("barStt");
        this.barTtft = document.getElementById("barTtft");
        this.barTts = document.getElementById("barTts");
        this.barE2e = document.getElementById("barE2e");

        this.initViewControls();
        this.initPersonalitySelector();
        this.init3DCharacter();
        this.initTextChat();
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

        const sendMessage = async () => {
            const text = input.value.trim();
            if (!text) return;

            // Show user message in bubble
            this.bubbleTag.textContent = "USER (TEXT)";
            this.bubbleTag.style.color = "var(--primary-cyan)";
            this.bubbleContent.textContent = text;
            this.logEvent("USER", text, "user");

            // Clear input and disable while processing
            input.value = "";
            btn.classList.add("sending");
            input.disabled = true;
            this.setRobotMode("thinking");

            try {
                const resp = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text }),
                });
                const data = await resp.json();
                const response = data.response || data.error || "No response.";

                // Show robot response
                this.bubbleTag.textContent = "ROBOT RESPONSE";
                this.bubbleTag.style.color = "var(--accent-emerald)";
                this.bubbleContent.textContent = response;
                this.logEvent("ROBOT", response, "robot");
                this.setRobotMode("speaking");

                // Trigger lip-sync on all 3D engines
                this._forEachEngine((eng) => {
                    eng.speakStart();
                    setTimeout(() => eng.speakStop(), Math.min(8000, response.length * 80));
                });

                // Return to idle after speech completes
                setTimeout(() => this.setRobotMode("idle"), Math.min(10000, response.length * 80 + 1000));
            } catch (err) {
                this.bubbleContent.textContent = `Error: ${err.message}`;
                this.logEvent("ERROR", err.message, "system");
                this.setRobotMode("idle");
            } finally {
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

        // ─── Realistic 3D Character Engine (created lazily on tab click) ───
        this.realisticContainer = document.getElementById("realisticContainer");
        this.realisticEngine = null;
        this._activeTab = "3d"; // "3d" | "realistic" | "face"

        // Avatar Tab Switcher (3D Humanoid vs Realistic Girl vs 2D Visor)
        const tab3D = document.getElementById("tabBtn3D");
        const tabRealGirl = document.getElementById("tabBtnRealGirl");
        const tabFace = document.getElementById("tabBtnFace");

        const resetTabs = () => {
            [tab3D, tabRealGirl, tabFace].forEach(t => t && t.classList.remove("active"));
            if (this.characterContainer) this.characterContainer.style.display = "none";
            if (this.realisticContainer) this.realisticContainer.style.display = "none";
            if (this.faceStage) this.faceStage.style.display = "none";
        };

        if (tab3D) {
            tab3D.addEventListener("click", () => {
                resetTabs();
                tab3D.classList.add("active");
                if (this.characterContainer) this.characterContainer.style.display = "flex";
                if (this.characterEngine) this.characterEngine.onResize();
                this._activeTab = "3d";
            });
        }

        if (tabRealGirl) {
            tabRealGirl.addEventListener("click", () => {
                resetTabs();
                tabRealGirl.classList.add("active");
                if (this.realisticContainer) this.realisticContainer.style.display = "flex";
                this._activeTab = "realistic";

                // Lazy-init the realistic engine on first tab click
                if (!this.realisticEngine && typeof CharacterEngine !== "undefined" && this.realisticContainer) {
                    this.realisticEngine = new CharacterEngine(this.realisticContainer, {
                        variant: "realistic",
                        modelUrl: "/static/models/avatar.glb",
                    });
                    this.logEvent("3D-REAL", "Realistic 3D character initialized (portrait lighting).", "system");
                } else if (this.realisticEngine) {
                    this.realisticEngine.onResize();
                }
            });
        }
        if (tabFace) {
            tabFace.addEventListener("click", () => {
                resetTabs();
                tabFace.classList.add("active");
                if (this.faceStage) this.faceStage.style.display = "flex";
                this._activeTab = "face";
            });
        }

        // Character Action Debug Buttons — dispatch to the active engine
        document.querySelectorAll(".dbg-act-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const act = btn.getAttribute("data-act");
                const side = btn.getAttribute("data-side") || null;
                const emo = btn.getAttribute("data-emo") || null;
                this.executeSemanticAction({ action: act, side: side, emotion: emo });
            });
        });
    }

    /** Returns whichever CharacterEngine is currently visible, or the animated one as default */
    _activeCharEngine() {
        if (this._activeTab === "realistic" && this.realisticEngine) return this.realisticEngine;
        return this.characterEngine;
    }

    /** Execute a callback on every initialized CharacterEngine */
    _forEachEngine(fn) {
        if (this.characterEngine) fn(this.characterEngine);
        if (this.realisticEngine) fn(this.realisticEngine);
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
        const btnFace = document.getElementById("btnViewFace");
        const btnDual = document.getElementById("btnViewDual");
        const btnDev = document.getElementById("btnViewDev");
        const btnFs = document.getElementById("btnFullscreen");
        const btnClear = document.getElementById("btnClearLog");

        btnFace.addEventListener("click", () => this.setView("face"));
        btnDual.addEventListener("click", () => this.setView("dual"));
        btnDev.addEventListener("click", () => this.setView("dev"));

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
                        const label = p === "neo" ? "NEO (Confident / Direct)"
                                    : p === "aria" ? "ARIA (Empathetic / Warm)"
                                    : p === "glados" ? "GLADOS (Sarcastic / Clinical)"
                                    : p.toUpperCase();
                        opt.textContent = label;
                        this.personaSelect.appendChild(opt);
                    });
                }
                if (data.active) {
                    this.personaSelect.value = data.active;
                    if (this.personaName) this.personaName.textContent = data.active.toUpperCase();
                }
            }
        } catch (e) {}

        this.personaSelect.addEventListener("change", async (e) => {
            const chosen = e.target.value;
            try {
                const switchRes = await fetch(`/personality/${chosen}`, { method: "POST" });
                if (switchRes.ok) {
                    this.logEvent("PERSONA", `Personality switched to ${chosen.toUpperCase()}`, "system");
                    if (this.personaName) this.personaName.textContent = chosen.toUpperCase();
                }
            } catch (err) {
                this.logEvent("ERROR", `Failed to switch personality: ${err.message}`, "error");
            }
        });
    }

    setView(mode) {
        document.body.className = `mode-${mode}`;
        document.querySelectorAll(".view-btn").forEach(btn => btn.classList.remove("active"));
        if (mode === "face") document.getElementById("btnViewFace").classList.add("active");
        if (mode === "dual") document.getElementById("btnViewDual").classList.add("active");
        if (mode === "dev") document.getElementById("btnViewDev").classList.add("active");
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
        if (this.currentMode === mode) return;
        this.currentMode = mode;

        // Reset stage state classes
        this.faceStage.classList.remove("state-idle", "state-listening", "state-thinking", "state-speaking");
        this.faceStage.classList.add(`state-${mode}`);

        // Update Status Pill
        this.statusPill.className = `status-pill status-${mode}`;
        this.statusText.textContent = mode.toUpperCase();

        // Update Face Caption & 3D Character Posture / Expression (all engines)
        if (this.charStateBadge) this.charStateBadge.textContent = `STATE: ${mode.toUpperCase()}`;
        if (this.dbgCharState) this.dbgCharState.textContent = mode.toUpperCase();

        switch (mode) {
            case "listening":
                this.faceCaption.textContent = "LISTENING FOR VOICE";
                this.stopMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.setEmotion("neutral");
                    eng.lookAt(0, 1.45, 2.0);
                });
                break;
            case "thinking":
                this.faceCaption.textContent = "PROCESSING & REASONING";
                this.stopMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.setEmotion("thinking");
                    eng.lookAt(0.3, 1.6, 2.2);
                });
                break;
            case "speaking":
                this.faceCaption.textContent = "SYNTHESIZING & SPEAKING";
                this.startMouthSpeaking();
                this._forEachEngine((eng) => {
                    eng.speakStart();
                    eng.setEmotion("happy");
                });
                break;
            default:
                this.faceCaption.textContent = "OBSERVING ENVIRONMENT";
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
        if (data.mode) {
            this.setRobotMode(data.mode);
        }

        if (data.user_text) {
            this.bubbleTag.textContent = "USER SPEECH";
            this.bubbleTag.style.color = "var(--primary-cyan)";
            this.bubbleContent.textContent = data.user_text;
            this.logEvent("USER", data.user_text, "user");
        }

        if (data.robot_phrase) {
            this.bubbleTag.textContent = "ROBOT RESPONSE";
            this.bubbleTag.style.color = "var(--accent-emerald)";
            this.bubbleContent.textContent = data.robot_phrase;
            this.logEvent("ROBOT", data.robot_phrase, "robot");

            // Trigger lip-sync on ALL 3D character engines
            this._forEachEngine((eng) => {
                eng.speakStart();
                setTimeout(() => {
                    if (this.currentMode !== "speaking") {
                        eng.speakStop();
                    }
                }, Math.min(8000, data.robot_phrase.length * 80));
            });
        }

        if (data.barge_in) {
            this.logEvent("BARGE-IN", "Interrupted bot playback via voice activity!", "barge");
            this._forEachEngine((eng) => eng.stopMotion());
            this.setRobotMode("listening");
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

        if (data.metrics) {
            this.updateMetrics(data.metrics);
        }

        if (data.event === "observability.event") {
            this.handleObservabilityEvent(data);
        }
    }

    updateMetrics(metrics) {
        if (metrics.stt !== undefined) {
            this.metricStt.textContent = `${Math.round(metrics.stt)} ms`;
            this.barStt.style.width = `${Math.min(100, (metrics.stt / 1000) * 100)}%`;
        }
        if (metrics.ttft !== undefined) {
            this.metricTtft.textContent = `${Math.round(metrics.ttft)} ms`;
            this.hudTtft.textContent = `${Math.round(metrics.ttft)} ms`;
            this.barTtft.style.width = `${Math.min(100, (metrics.ttft / 1500) * 100)}%`;
        }
        if (metrics.synth !== undefined) {
            this.metricTts.textContent = `${Math.round(metrics.synth)} ms`;
            this.barTts.style.width = `${Math.min(100, (metrics.synth / 500) * 100)}%`;
        }
        if (metrics.e2e !== undefined) {
            this.metricE2e.textContent = `${Math.round(metrics.e2e)} ms`;
            this.hudE2e.textContent = `${Math.round(metrics.e2e)} ms`;
            this.barE2e.style.width = `${Math.min(100, (metrics.e2e / 2000) * 100)}%`;
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
                    if (data.telemetry) {
                        const t = data.telemetry;
                        const metrics = {};
                        if (t.STT_FINAL_LATENCY && t.STT_FINAL_LATENCY.last) metrics.stt = t.STT_FINAL_LATENCY.last;
                        if (t.LLM_TTFT && t.LLM_TTFT.last) metrics.ttft = t.LLM_TTFT.last;
                        if (t.TTS_SYNTHESIS_LATENCY && t.TTS_SYNTHESIS_LATENCY.last) metrics.synth = t.TTS_SYNTHESIS_LATENCY.last;
                        else if (t.TTS_FIRST_AUDIO_LATENCY && t.TTS_FIRST_AUDIO_LATENCY.last) metrics.synth = t.TTS_FIRST_AUDIO_LATENCY.last;
                        if (t.END_OF_SPEECH_TO_FIRST_AUDIO && t.END_OF_SPEECH_TO_FIRST_AUDIO.last) metrics.e2e = t.END_OF_SPEECH_TO_FIRST_AUDIO.last;
                        this.updateMetrics(metrics);
                    }
                }
            } catch (e) {}
        }, 1200);

        // Poll World State & Perception at 4Hz (250ms) for responsive face tracking
        setInterval(async () => {
            try {
                const res = await fetch("/world-state");
                if (res.ok) {
                    const ws = await res.json();
                    if (!ws.error) {
                        this.worldStateCode.textContent = JSON.stringify(ws, null, 2);
                        document.getElementById("hudLighting").textContent = `LIGHT: ${(ws.lighting || "NORMAL").toUpperCase()}`;
                        document.getElementById("hudMotion").textContent = `MOTION: ${(ws.motion || "NONE").toUpperCase()}`;
                        document.getElementById("hudPeople").textContent = `PEOPLE: ${ws.people_count || 0}`;

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
                    if (hw.cpu_percent !== undefined) document.getElementById("hwCpuVal").textContent = `${hw.cpu_percent.toFixed(1)} %`;
                    if (hw.system_ram_gb !== undefined) document.getElementById("hwRamVal").textContent = `${hw.system_ram_gb.toFixed(1)} GB`;
                    if (hw.gpu_vram_mb !== undefined) document.getElementById("hwVramVal").textContent = `${Math.round(hw.gpu_vram_mb)} MB`;
                    if (hw.gpu_temp_c !== undefined) document.getElementById("hwTempVal").textContent = `${Math.round(hw.gpu_temp_c)} °C`;
                    if (hw.amd_utilization !== undefined) document.getElementById("hwAmdVal").textContent = `${Math.round(hw.amd_utilization)} %`;
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
            if (total) document.getElementById("traceTotalDuration").textContent = `DURATION: ${total.toFixed(1)} ms`;
        }
        if (evt.type === "pipeline_updated" && evt.pipeline) {
            this.updatePipelineStatus(evt.pipeline);
        }
    }

    updatePipelineStatus(pipeline) {
        const nodes = ["mic", "vad", "stt", "context", "llm", "tools", "tts", "avatar", "image"];
        for (const key of nodes) {
            const el = document.getElementById(`node${key.charAt(0).toUpperCase() + key.slice(1)}`);
            if (el) {
                const status = pipeline[key];
                el.className = "pipe-node";
                el.querySelector(".node-status").textContent = (status || "IDLE").toUpperCase();
                if (status === "active") {
                    el.classList.add("active");
                } else if (status === "done") {
                    el.classList.add("done");
                } else if (status === "error") {
                    el.classList.add("error");
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

        if (btnGen && input) {
            btnGen.addEventListener("click", async () => {
                const prompt = input.value.trim();
                if (!prompt) return;
                try {
                    const res = await fetch("/images/generate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ prompt: prompt, steps: 8 })
                    });
                    if (res.ok) {
                        input.value = "";
                        this.logEvent("IMAGE", `Generation job submitted: ${prompt}`, "system");
                    }
                } catch (e) {
                    console.error("Image gen failed", e);
                }
            });
        }

        if (btnCancel) {
            btnCancel.addEventListener("click", async () => {
                if (this.activeImageJob) {
                    await fetch(`/images/cancel/${this.activeImageJob}`, { method: "POST" });
                    this.logEvent("IMAGE", `Cancelled job ${this.activeImageJob}`, "system");
                }
            });
        }
    }

    updateImageJobsUI(jobs) {
        const badge = document.getElementById("imgJobBadge");
        const btnCancel = document.getElementById("btnCancelImage");
        const previewImg = document.getElementById("imgPreviewImg");
        const placeholder = document.getElementById("imgPlaceholder");
        const genStatus = document.getElementById("imgGeneratingStatus");
        const metaBlock = document.getElementById("imgMetaBlock");

        let activeJob = jobs.find(j => j.status === "queued" || j.status === "running");
        let completedJob = [...jobs].reverse().find(j => j.status === "completed" || j.status === "COMPLETED");
        if (!activeJob && jobs.length > 0) activeJob = jobs[jobs.length - 1]; // Use last job if none active

        this.activeImageJob = activeJob ? activeJob.job_id : null;
        
        if (activeJob && (activeJob.status === "queued" || activeJob.status === "running" || activeJob.status === "QUEUED" || activeJob.status === "RUNNING")) {
            badge.textContent = `STATUS: ${activeJob.status.toUpperCase()}`;
            if (btnCancel) btnCancel.disabled = false;
            
            // Show generating status
            placeholder.style.display = "none";
            previewImg.style.display = "none";
            metaBlock.style.display = "none";
            genStatus.style.display = "block";
            
            document.getElementById("imgGenStep").textContent = `Step ${Math.floor((activeJob.progress / 100) * activeJob.steps)}/${activeJob.steps}`;
            document.getElementById("imgGenBackend").textContent = `Backend: ${activeJob.backend || '--'}`;
            document.getElementById("imgGenDevice").textContent = `Device: ${activeJob.device || '--'}`;
            const elapsed = activeJob.started_at ? ((Date.now() / 1000) - activeJob.started_at).toFixed(1) : 0.0;
            document.getElementById("imgGenElapsed").textContent = `Elapsed: ${elapsed}s`;
            
            const pbar = document.getElementById("imgProgressBar");
            const pwrap = document.getElementById("imgProgressWrap");
            if (pbar && pwrap) {
                pwrap.style.display = "block";
                pbar.style.width = `${activeJob.progress}%`;
            }
        } else {
            badge.textContent = "STATUS: READY";
            if (btnCancel) btnCancel.disabled = true;
            genStatus.style.display = "none";
            
            const pwrap = document.getElementById("imgProgressWrap");
            if (pwrap) pwrap.style.display = "none";
            
            if (completedJob && completedJob.output_path) {
                placeholder.style.display = "none";
                previewImg.style.display = "block";
                previewImg.src = completedJob.output_path;
                
                metaBlock.style.display = "block";
                document.getElementById("metaModel").textContent = completedJob.model || '--';
                document.getElementById("metaBackend").textContent = completedJob.backend || '--';
                document.getElementById("metaDevice").textContent = completedJob.device || '--';
                document.getElementById("metaTime").textContent = `${(completedJob.duration_s || 0).toFixed(2)}s`;
                document.getElementById("metaRes").textContent = `${completedJob.width}x${completedJob.height}`;
                document.getElementById("metaSteps").textContent = completedJob.steps || '--';
            } else {
                placeholder.style.display = "block";
                previewImg.style.display = "none";
                metaBlock.style.display = "none";
            }
        }
    }
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    window.robotUI = new RobotUI();
});
