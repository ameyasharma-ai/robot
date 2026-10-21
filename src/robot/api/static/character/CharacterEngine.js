/**
 * CharacterEngine — Central humanoid 3D visual avatar engine.
 * Native WebGL2/WebGPU rendering surface integrated directly into browser interface.
 * Operates on integrated GPU (AMD Radeon 680M) without touching the discrete RTX 3050.
 *
 * Supports multiple instances: one for animated 3D humanoid, one for realistic model.
 * Pass `options.variant = "realistic"` for portrait-style warm lighting and no cyber stage.
 */

class CharacterEngine {
    constructor(containerElement, options = {}) {
        this.container = containerElement;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.clock = new THREE.Clock();

        this.rig = new CharacterRig();
        this.loader = new CharacterLoader();
        this.animationSystem = null;
        this.gazeSystem = null;
        this.facialSystem = null;

        // Options
        this.modelUrl = options.modelUrl || "/static/models/avatar.glb?v=9.0-skinny-cute-pro";
        this.variant = options.variant || "animated"; // "animated" | "realistic"

        // Diagnostics
        this.fps = 60;
        this.frameTimeMs = 16.6;
        this.frameCount = 0;
        this.lastFpsUpdate = performance.now();
        this.isLoaded = false;
        this.backend = "WebGL2";

        this.init();
    }

    async init() {
        const width = this.container.clientWidth || 600;
        const height = this.container.clientHeight || 700;

        // 1. Three.js Scene Setup
        this.scene = new THREE.Scene();
        this.scene.background = null; // Transparent to blend seamlessly into UI

        // 2. Camera Setup — variant-dependent framing
        if (this.variant === "realistic") {
            // Portrait-style camera: tighter framing, slightly higher angle
            this.camera = new THREE.PerspectiveCamera(30, width / height, 0.1, 100);
            this.camera.position.set(0, 0.65, 2.6);
            this.camera.lookAt(0, 0.55, 0);
        } else {
            // Default animated: full body + holographic pedestal
            this.camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
            this.camera.position.set(0, 0.60, 2.95);
            this.camera.lookAt(0, 0.52, 0);
        }

        // 3. Renderer Setup (WebGL2 with high-performance power preference)
        try {
            this.renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: true,
                powerPreference: "high-performance", // Prefers standard browser GPU (AMD 680M)
            });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            this.renderer.outputEncoding = THREE.sRGBEncoding;
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            this.container.appendChild(this.renderer.domElement);
            this.backend = "WebGL2 (AMD Radeon 680M / Browser)";
        } catch (e) {
            console.error("[CharacterEngine] WebGL2 init failed:", e);
            return;
        }

        // 4. Lighting & Stage (variant-dependent)
        this._setupLighting();

        // 5. Load Humanoid Avatar
        await this._loadCharacter();

        // 6. Window resize listener and container ResizeObserver
        window.addEventListener("resize", () => this.onResize());
        if (window.ResizeObserver) {
            this._resizeObserver = new ResizeObserver(() => this.onResize());
            this._resizeObserver.observe(this.container);
        }

        // 7. Start render loop
        this.animate();
    }

    _setupLighting() {
        if (this.variant === "realistic") {
            // ─── Realistic Portrait Lighting ───────────────────────────────────
            // Warm three-point studio setup for natural skin rendering
            const ambient = new THREE.AmbientLight(0xfff5eb, 0.55);
            this.scene.add(ambient);

            // Key light: warm directional (simulates softbox at 45° upper-right)
            const keyLight = new THREE.DirectionalLight(0xfff0e0, 1.2);
            keyLight.position.set(2.5, 3.5, 3);
            keyLight.castShadow = true;
            keyLight.shadow.mapSize.width = 1024;
            keyLight.shadow.mapSize.height = 1024;
            keyLight.shadow.bias = -0.001;
            this.scene.add(keyLight);

            // Fill light: soft blue-ish from left (adds dimension and separation)
            const fillLight = new THREE.DirectionalLight(0xc8ddf0, 0.6);
            fillLight.position.set(-2, 2, 2);
            this.scene.add(fillLight);

            // Rim/back light: subtle warm edge light for hair/shoulder separation
            const rimLight = new THREE.DirectionalLight(0xffe8cc, 0.45);
            rimLight.position.set(0, 3, -2);
            this.scene.add(rimLight);

            // Subtle ground reflection (point light from below, very soft)
            const groundBounce = new THREE.PointLight(0xe8e0d8, 0.2, 5);
            groundBounce.position.set(0, -0.5, 1);
            this.scene.add(groundBounce);

            // ─── Elegant Minimal Stage (subtle glass-like disc) ───────────────
            const discGeo = new THREE.CylinderGeometry(0.52, 0.55, 0.018, 48);
            const discMat = new THREE.MeshStandardMaterial({
                color: 0x1a1a2e,
                roughness: 0.15,
                metalness: 0.7,
                transparent: true,
                opacity: 0.6,
            });
            const disc = new THREE.Mesh(discGeo, discMat);
            disc.position.set(0, -0.009, 0);
            disc.receiveShadow = true;
            this.scene.add(disc);

            // Thin accent ring
            const ringGeo = new THREE.TorusGeometry(0.53, 0.004, 16, 64);
            const ringMat = new THREE.MeshStandardMaterial({
                color: 0xffe8cc,
                emissive: 0xffe8cc,
                emissiveIntensity: 0.3,
                roughness: 0.1,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = Math.PI / 2;
            ring.position.set(0, 0.002, 0);
            this.scene.add(ring);
        } else {
            // ─── Cinematic Realistic Portrait Studio Lighting ───────────────────
            // Warm Ambient Base (fills micro-creases and removes harsh under-chin dropouts)
            const ambient = new THREE.AmbientLight(0xfff6ed, 0.50);
            this.scene.add(ambient);

            // Frontal Soft Beauty Ring / Portrait Light
            // Positioned right at face/chest level (Y=1.25, Z=2.2) to illuminate radiant facial skin,
            // eliminate any dark lower face/chin shadow band, and create sparkling cornea catchlights
            const portraitBeautyLight = new THREE.DirectionalLight(0xfff7f0, 0.40);
            portraitBeautyLight.position.set(0.0, 1.25, 2.2);
            this.scene.add(portraitBeautyLight);

            // Key Light: Warm 3800K directional softbox at natural 30° upper-right
            const keyLight = new THREE.DirectionalLight(0xfff2e6, 0.85);
            keyLight.position.set(1.6, 2.2, 2.6);
            keyLight.castShadow = true;
            keyLight.shadow.mapSize.width = 1024;
            keyLight.shadow.mapSize.height = 1024;
            keyLight.shadow.bias = -0.0005;
            this.scene.add(keyLight);

            // Fill Light: Soft cool daylight (6000K) from left at eye/chest level
            const fillLight = new THREE.DirectionalLight(0xd0e2f5, 0.35);
            fillLight.position.set(-1.8, 1.6, 2.0);
            this.scene.add(fillLight);

            // Rim / Hair Light: Warm backlight separating hair and shoulders from dark backdrop
            const rimLight = new THREE.DirectionalLight(0xffe4cc, 0.65);
            rimLight.position.set(0.0, 3.2, -2.5);
            this.scene.add(rimLight);

            // Ground Bounce: Soft warm floor bounce softening lower contours
            const groundBounce = new THREE.PointLight(0xebe2d8, 0.25, 5);
            groundBounce.position.set(0, -0.3, 1.2);
            this.scene.add(groundBounce);

            // ─── Sleek Minimalist Obsidian & Smoked Glass Pedestal ─────────────
            // Core Platform
            const pedestalBaseGeo = new THREE.CylinderGeometry(0.62, 0.65, 0.035, 48);
            const pedestalBaseMat = new THREE.MeshStandardMaterial({
                color: 0x0c0e14,
                roughness: 0.25,
                metalness: 0.85,
            });
            const pedestalBase = new THREE.Mesh(pedestalBaseGeo, pedestalBaseMat);
            pedestalBase.position.set(0, -0.018, 0);
            pedestalBase.receiveShadow = true;
            this.scene.add(pedestalBase);

            // Smoked Glass Stage Disc
            const coreGeo = new THREE.CylinderGeometry(0.55, 0.58, 0.025, 48);
            const coreMat = new THREE.MeshPhysicalMaterial({
                color: 0x141824,
                roughness: 0.15,
                metalness: 0.7,
                clearcoat: 0.8,
                clearcoatRoughness: 0.1,
            });
            const coreMesh = new THREE.Mesh(coreGeo, coreMat);
            coreMesh.position.set(0, 0.005, 0);
            coreMesh.receiveShadow = true;
            this.scene.add(coreMesh);

            // Elegant Thin Luminous Accent Ring (Warm Rose Gold / Champagne)
            const outerRingGeo = new THREE.TorusGeometry(0.63, 0.005, 16, 64);
            const outerRingMat = new THREE.MeshStandardMaterial({
                color: 0xf5d0b0,
                emissive: 0xf5d0b0,
                emissiveIntensity: 0.45,
                roughness: 0.1,
            });
            const outerRing = new THREE.Mesh(outerRingGeo, outerRingMat);
            outerRing.name = "pedestalAccentRing";
            outerRing.rotation.x = Math.PI / 2;
            outerRing.position.set(0, 0.002, 0);
            this.scene.add(outerRing);

            // Inner Accent Ring
            const innerRingGeo = new THREE.TorusGeometry(0.53, 0.003, 16, 64);
            const innerRingMat = new THREE.MeshStandardMaterial({
                color: 0xe8b59b,
                emissive: 0xe8b59b,
                emissiveIntensity: 0.35,
                roughness: 0.1,
            });
            const innerRing = new THREE.Mesh(innerRingGeo, innerRingMat);
            innerRing.name = "pedestalAccentRing";
            innerRing.rotation.x = Math.PI / 2;
            innerRing.position.set(0, 0.018, 0);
            this.scene.add(innerRing);
        }
    }

    async _loadCharacter() {
        try {
            const { scene, isProcedural } = await this.loader.load(this.modelUrl);
            this.scene.add(scene);

            // Bind CharacterRig
            this.rig.bind(scene);

            // Initialize sub-systems
            if (typeof AnimationSystem !== "undefined") {
                this.animationSystem = new AnimationSystem(this.rig);
            }
            if (typeof GazeSystem !== "undefined") {
                this.gazeSystem = new GazeSystem(this.rig);
            }
            if (typeof FacialSystem !== "undefined") {
                this.facialSystem = new FacialSystem(this.rig);
            }

            this.isLoaded = true;
            this.onResize();
            console.log(`[CharacterEngine:${this.variant}] Humanoid character system online! (procedural: ${isProcedural})`);
        } catch (e) {
            console.error(`[CharacterEngine:${this.variant}] Failed loading avatar GLB:`, e);
            try {
                const { scene } = this.loader.buildProceduralHumanoid();
                this.scene.add(scene);
                this.rig.bind(scene);
                if (typeof AnimationSystem !== "undefined") this.animationSystem = new AnimationSystem(this.rig);
                if (typeof GazeSystem !== "undefined") this.gazeSystem = new GazeSystem(this.rig);
                this.isLoaded = true;
                this.onResize();
                console.log(`[CharacterEngine:${this.variant}] Fallback procedural humanoid active!`);
            } catch (fallbackErr) {
                console.error(`[CharacterEngine:${this.variant}] Critical avatar error:`, fallbackErr);
            }
        }
    }

    /**
     * Active AnimationSystem instance.
     * @returns {AnimationSystem|null}
     */
    get animation() {
        return this.animationSystem;
    }

    /**
     * Active GazeSystem instance.
     * @returns {GazeSystem|null}
     */
    get gaze() {
        return this.gazeSystem;
    }

    onResize() {
        if (!this.container || !this.renderer || !this.camera) return;
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        if (width === 0 || height === 0) return;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        const t0 = performance.now();
        const delta = Math.min(0.06, this.clock.getDelta());

        // Update Systems
        if (this.isLoaded) {
            if (this.animationSystem) this.animationSystem.update(delta);
            if (this.gazeSystem) this.gazeSystem.update(delta);
            if (this.facialSystem) this.facialSystem.update(delta);
        }

        // Render
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }

        const t1 = performance.now();
        this.frameTimeMs = t1 - t0;
        this.frameCount++;

        if (t1 - this.lastFpsUpdate >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsUpdate = t1;
        }
    }

    /**
     * Dynamically update 3D WebGL lighting, pedestal glow rings, and avatar materials per persona.
     * @param {string} personaName - Persona key: "luna", "neo", "cypher", "aria", "glados"
     */
    setPersonaTheme(personaName) {
        if (!personaName) return;
        const persona = personaName.toLowerCase();
        this.activePersona = persona;

        const themes = {
            luna: { keyLight: 0xff3399, fillLight: 0xff66b2, ring: 0xff3399 },
            neo: { keyLight: 0xff1744, fillLight: 0xff5252, ring: 0xff1744 },
            cypher: { keyLight: 0xff1744, fillLight: 0xff5252, ring: 0xff1744 },
            aria: { keyLight: 0xa855f7, fillLight: 0xc084fc, ring: 0xa855f7 },
            glados: { keyLight: 0x00ff66, fillLight: 0x66ff99, ring: 0x00ff66 }
        };
        const t = themes[persona] || themes.neo;

        this.scene.traverse((child) => {
            if (child.isDirectionalLight) {
                // Keep key light warm natural portrait light (x > 0, z > 0)
                if (child.position.x > 0 && child.position.z > 0) {
                    child.color.setHex(0xfff2e6);
                } else if (child.position.z < 0) {
                    // Rim light behind character carries the persona theme edge halo
                    child.color.setHex(t.ring);
                } else {
                    // Soft fill light
                    child.color.setHex(0xd0e2f5);
                }
            }
            if (child.isMesh && (child.name === "pedestalAccentRing" || child.name === "characterEyesVisor")) {
                if (child.material && child.material.emissive) {
                    child.material.emissive.setHex(t.ring);
                }
            }
        });
    }

    // ─── Semantic Action API ────────────────────────────────────────────────

    idle() {
        if (this.animationSystem) this.animationSystem.interrupt();
        if (this.facialSystem) this.facialSystem.setEmotion("neutral");
    }

    reset() {
        if (this.animationSystem) this.animationSystem.interrupt();
        if (this.rig) {
            this.rig.resetPose();
            if (this.rig.root) {
                this.rig.root.rotation.set(0, 0, 0);
            }
        }
        if (this.gazeSystem) {
            this.gazeSystem.setTarget(0, 1.15, 2.2);
        }
        if (this.facialSystem) {
            this.facialSystem.setEmotion("neutral");
            this.facialSystem.stopSpeaking();
        }
    }

    raiseHand(side = "right", duration = 1.2) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null; // cancel any in-progress idle blend
        this.animationSystem.activeActions.upperBody = {
            type: "raise_hand",
            side: side,
            duration: duration,
            elapsed: 0,
        };
    }

    wave(side = "right", repetitions = 3, duration = 2.0) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null; // cancel any in-progress idle blend
        this.animationSystem.activeActions.upperBody = {
            type: "wave",
            side: side,
            repetitions: repetitions,
            duration: duration,
            elapsed: 0,
        };
    }

    point(side = "right", target = null, duration = 2.0) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null; // cancel any in-progress idle blend
        this.animationSystem.activeActions.upperBody = {
            type: "point",
            side: side,
            target: target,
            duration: duration,
            elapsed: 0,
        };
    }

    nod(duration = 1.4) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "nod", duration, elapsed: 0 };
    }

    shakeHead(duration = 1.4) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "shake_head", duration, elapsed: 0 };
    }

    tiltHead(duration = 1.5) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "tilt_head", duration, elapsed: 0 };
    }

    shrug(duration = 1.6) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "shrug", duration, elapsed: 0 };
    }

    bow(duration = 2.2) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "bow", duration, elapsed: 0 };
    }

    clap(duration = 2.2) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "clap", duration, elapsed: 0 };
    }

    think(duration = 2.5) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "think", duration, elapsed: 0 };
    }

    cheer(duration = 2.0) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "cheer", duration, elapsed: 0 };
    }

    handsOnHips(duration = 2.5) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "hands_on_hips", duration, elapsed: 0 };
    }

    heartGesture(duration = 2.2) {
        if (!this.animationSystem) return;
        this.animationSystem._returnBlend = null;
        this.animationSystem.activeActions.upperBody = { type: "heart_gesture", duration, elapsed: 0 };
    }

    kick(side = "right", direction = "forward", intensity = 0.85, duration = 1.1) {
        if (!this.animationSystem) return;
        this.animationSystem.activeActions.lowerBody = {
            type: "kick",
            side: side,
            direction: direction,
            intensity: intensity,
            duration: duration,
            elapsed: 0,
        };
    }

    walk(direction = "forward", distance = 1.5, duration = 3.0) {
        if (!this.animationSystem) return;
        this.animationSystem.activeActions.locomotion = {
            type: "walk",
            direction: direction === "backward" ? -1.0 : 1.0,
            distance: distance,
            duration: duration,
            speed: 1.8,
            progress: 0,
        };
    }

    turn(angle = 180, duration = 2.0) {
        if (!this.animationSystem) return;
        this.animationSystem.activeActions.locomotion = {
            type: "turn",
            targetAngle: (angle * Math.PI) / 180,
            duration: duration,
            progress: 0,
        };
    }

    sit() {
        const hips = this.rig.getBone("hips");
        const leftLeg = this.rig.getBone("leftUpperLeg");
        const rightLeg = this.rig.getBone("rightUpperLeg");
        const leftKnee = this.rig.getBone("leftLowerLeg");
        const rightKnee = this.rig.getBone("rightLowerLeg");

        if (hips && this.rig.initialPositions["hips"]) {
            hips.position.y = this.rig.initialPositions["hips"].y * 0.65;
        }
        if (leftLeg && this.rig.initialRotations["leftUpperLeg"]) {
            const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2.3, 0, 0));
            leftLeg.quaternion.copy(this.rig.initialRotations["leftUpperLeg"]).multiply(delta);
        }
        if (rightLeg && this.rig.initialRotations["rightUpperLeg"]) {
            const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2.3, 0, 0));
            rightLeg.quaternion.copy(this.rig.initialRotations["rightUpperLeg"]).multiply(delta);
        }
        if (leftKnee && this.rig.initialRotations["leftLowerLeg"]) {
            const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2.2, 0, 0));
            leftKnee.quaternion.copy(this.rig.initialRotations["leftLowerLeg"]).multiply(delta);
        }
        if (rightKnee && this.rig.initialRotations["rightLowerLeg"]) {
            const delta = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2.2, 0, 0));
            rightKnee.quaternion.copy(this.rig.initialRotations["rightLowerLeg"]).multiply(delta);
        }
    }

    stand() {
        this.idle();
    }

    lookAt(x, y, z) {
        if (this.gazeSystem) this.gazeSystem.setTarget(x, y, z);
    }

    setOpenCVUserFace(normX, normY) {
        if (this.gazeSystem) this.gazeSystem.setOpenCVFace(normX, normY);
    }

    setEmotion(emotion) {
        if (this.facialSystem) this.facialSystem.setEmotion(emotion);
    }

    speakStart() {
        if (this.facialSystem) this.facialSystem.startSpeaking();
        if (this.animationSystem) {
            this.animationSystem._returnBlend = null; // cancel any idle-return blend immediately
            this.animationSystem.isSpeaking = true;
        }
    }

    speakStop() {
        if (this.facialSystem) this.facialSystem.stopSpeaking();
        if (this.animationSystem) this.animationSystem.isSpeaking = false;
    }

    stopMotion() {
        if (this.animationSystem) {
            this.animationSystem.interrupt();
            this.animationSystem.isSpeaking = false;
        }
        if (this.facialSystem) this.facialSystem.stopSpeaking();
    }

    /**
     * Dispatch validated semantic action dictionary
     */
    executeAction(actionObj) {
        if (!actionObj || !actionObj.action) return;
        const act = actionObj.action.toLowerCase();
        const side = actionObj.side || "right";
        const duration = actionObj.duration || 1.2;
        const intensity = actionObj.intensity || 1.0;
        const repetitions = actionObj.repetitions || 3;

        console.log(`[CharacterEngine] Executing semantic action: ${act}`, actionObj);

        switch (act) {
            case "raise_hand":
                this.raiseHand(side, duration);
                break;
            case "wave":
                this.wave(side, repetitions, duration);
                break;
            case "kick":
                this.kick(side, actionObj.direction || "forward", intensity, duration);
                break;
            case "walk":
                this.walk(actionObj.direction || "forward", actionObj.distance || 1.5, duration);
                break;
            case "turn":
                this.turn(actionObj.angle || 180, duration);
                break;
            case "point":
                this.point(side, actionObj.target, duration);
                break;
            case "look_at":
                if (actionObj.target) {
                    this.lookAt(actionObj.target.x || 0, actionObj.target.y || 1.45, actionObj.target.z || 2.0);
                }
                break;
            case "sit":
                this.sit();
                break;
            case "stand":
                this.stand();
                break;
            case "nod":
                this.nod(duration);
                break;
            case "shake_head":
                this.shakeHead(duration);
                break;
            case "tilt_head":
                this.tiltHead(duration);
                break;
            case "shrug":
                this.shrug(duration);
                break;
            case "bow":
                this.bow(duration);
                break;
            case "clap":
                this.clap(duration);
                break;
            case "think":
                this.think(duration);
                break;
            case "cheer":
                this.cheer(duration);
                break;
            case "hands_on_hips":
                this.handsOnHips(duration);
                break;
            case "heart_gesture":
                this.heartGesture(duration);
                break;
            case "set_emotion":
                if (actionObj.emotion) this.setEmotion(actionObj.emotion);
                break;
            case "stop":
                this.stopMotion();
                break;
            case "idle":
                this.idle();
                break;
            case "reset":
            case "reset_position":
                this.reset();
                break;
            default:
                console.warn(`[CharacterEngine] Unknown semantic action: ${act}`);
                break;
        }
    }
}

window.CharacterEngine = CharacterEngine;
