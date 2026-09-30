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
        this.init3DCharacter();
        this.initBlinking();
        this.initWebsocket();
        this.startPollers();
    }

    init3DCharacter() {
        if (typeof CharacterEngine !== "undefined" && this.characterContainer) {
            this.characterEngine = new CharacterEngine(this.characterContainer);
            this.logEvent("3D-CORE", "3D Humanoid Character initialized on AMD 680M WebGL2.", "system");

            // Update debug metrics every 500ms
            setInterval(() => {
                if (this.characterEngine && this.dbgCharFps) {
                    this.dbgCharFps.textContent = `FPS: ${this.characterEngine.fps} (${this.characterEngine.frameTimeMs.toFixed(1)}ms)`;
                }
            }, 500);
        }

        // Avatar Tab Switcher (3D Humanoid vs Realistic Girl vs 2D Visor)
        const tab3D = document.getElementById("tabBtn3D");
        const tabRealGirl = document.getElementById("tabBtnRealGirl");
        const tabFace = document.getElementById("tabBtnFace");
        const sketchfabContainer = document.getElementById("sketchfabContainer");

        const resetTabs = () => {
            [tab3D, tabRealGirl, tabFace].forEach(t => t && t.classList.remove("active"));
            if (this.characterContainer) this.characterContainer.style.display = "none";
            if (sketchfabContainer) sketchfabContainer.style.display = "none";
            if (this.faceStage) this.faceStage.style.display = "none";
        };

        if (tab3D) {
            tab3D.addEventListener("click", () => {
                resetTabs();
                tab3D.classList.add("active");
                if (this.characterContainer) this.characterContainer.style.display = "flex";
                if (this.characterEngine) this.characterEngine.onResize();
            });
        }
        if (tabRealGirl) {
            tabRealGirl.addEventListener("click", () => {
                resetTabs();
                tabRealGirl.classList.add("active");
                if (sketchfabContainer) sketchfabContainer.style.display = "block";
            });
        }
        if (tabFace) {
            tabFace.addEventListener("click", () => {
                resetTabs();
                tabFace.classList.add("active");
                if (this.faceStage) this.faceStage.style.display = "flex";
            });
        }

        // Character Action Debug Buttons
        document.querySelectorAll(".dbg-act-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                const act = btn.getAttribute("data-act");
                const side = btn.getAttribute("data-side") || null;
                const emo = btn.getAttribute("data-emo") || null;
                this.executeSemanticAction({ action: act, side: side, emotion: emo });
            });
        });
    }

    async executeSemanticAction(actionObj) {
        if (this.characterEngine) {
            this.characterEngine.executeAction(actionObj);
            if (this.charActionBadge) {
                this.charActionBadge.textContent = `MOTION: ${(actionObj.action || "").toUpperCase()}`;
            }
            if (this.dbgCharMotion) {
                this.dbgCharMotion.textContent = `${(actionObj.action || "").toUpperCase()} ${(actionObj.side || "")}`.trim();
            }
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

        // Update Face Caption & 3D Character Posture / Expression
        if (this.characterEngine) {
            if (this.charStateBadge) this.charStateBadge.textContent = `STATE: ${mode.toUpperCase()}`;
            if (this.dbgCharState) this.dbgCharState.textContent = mode.toUpperCase();
        }

        switch (mode) {
            case "listening":
                this.faceCaption.textContent = "LISTENING FOR VOICE";
                this.stopMouthSpeaking();
                if (this.characterEngine) {
                    this.characterEngine.setEmotion("neutral");
                    this.characterEngine.lookAt(0, 1.45, 2.0); // attentive forward lean
                }
                break;
            case "thinking":
                this.faceCaption.textContent = "PROCESSING & REASONING";
                this.stopMouthSpeaking();
                if (this.characterEngine) {
                    this.characterEngine.setEmotion("thinking");
                    this.characterEngine.lookAt(0.3, 1.6, 2.2); // subtle contemplative head tilt
                }
                break;
            case "speaking":
                this.faceCaption.textContent = "SYNTHESIZING & SPEAKING";
                this.startMouthSpeaking();
                if (this.characterEngine) {
                    this.characterEngine.speakStart();
                    this.characterEngine.setEmotion("happy");
                }
                break;
            default:
                this.faceCaption.textContent = "OBSERVING ENVIRONMENT";
                this.stopMouthSpeaking();
                if (this.characterEngine) {
                    this.characterEngine.speakStop();
                    this.characterEngine.setEmotion("neutral");
                }
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

            // Trigger lip-sync on 3D character
            if (this.characterEngine) {
                this.characterEngine.speakStart();
                setTimeout(() => {
                    if (this.characterEngine && this.currentMode !== "speaking") {
                        this.characterEngine.speakStop();
                    }
                }, Math.min(8000, data.robot_phrase.length * 80));
            }
        }

        if (data.barge_in) {
            this.logEvent("BARGE-IN", "Interrupted bot playback via voice activity!", "barge");
            if (this.characterEngine) {
                this.characterEngine.stopMotion();
            }
        }

        // Real-time Semantic Character Actions dispatched from Robot Core / LLM
        if (data.event === "character.action" || data.action) {
            const act = data.action;
            if (act && this.characterEngine) {
                this.characterEngine.executeAction(act);
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
            if (this.characterEngine) {
                this.characterEngine.stopMotion();
                if (this.charActionBadge) this.charActionBadge.textContent = "MOTION: STOPPED";
                if (this.dbgCharMotion) this.dbgCharMotion.textContent = "INTERRUPTED (IDLE)";
            }
            this.logEvent("CHAR", "Motion interrupted. Blending to neutral idle.", "barge");
        }

        if (data.metrics) {
            this.updateMetrics(data.metrics);
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
                    if (data.personality) this.personaName.textContent = data.personality.toUpperCase();
                    if (data.telemetry) {
                        const t = data.telemetry;
                        const metrics = {};
                        if (t.STT_FINAL_LATENCY && t.STT_FINAL_LATENCY.last) metrics.stt = t.STT_FINAL_LATENCY.last;
                        if (t.LLM_TTFT && t.LLM_TTFT.last) metrics.ttft = t.LLM_TTFT.last;
                        if (t.TTS_FIRST_AUDIO_LATENCY && t.TTS_FIRST_AUDIO_LATENCY.last) metrics.synth = t.TTS_FIRST_AUDIO_LATENCY.last;
                        if (t.END_OF_SPEECH_TO_FIRST_AUDIO && t.END_OF_SPEECH_TO_FIRST_AUDIO.last) metrics.e2e = t.END_OF_SPEECH_TO_FIRST_AUDIO.last;
                        this.updateMetrics(metrics);
                    }
                }
            } catch (e) {}
        }, 1200);

        // Poll World State & Perception every 1.5 seconds
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
    }
}

// Initialize on page load
window.addEventListener("DOMContentLoaded", () => {
    window.robotUI = new RobotUI();
});
