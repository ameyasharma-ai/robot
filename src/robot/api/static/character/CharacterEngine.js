/**
 * CharacterEngine — Central humanoid 3D visual avatar engine.
 * Native WebGL2/WebGPU rendering surface integrated directly into browser interface.
 * Operates on integrated GPU (AMD Radeon 680M) without touching the discrete RTX 3050.
 */

class CharacterEngine {
    constructor(containerElement) {
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

        // 2. Camera Setup (Designer framing: full body + elevated holographic pedestal)
        this.camera = new THREE.PerspectiveCamera(34, width / height, 0.1, 100);
        this.camera.position.set(0, 0.90, 3.10);
        this.camera.lookAt(0, 0.78, 0);

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

        // 4. Studio & Futuristic Lighting Rig
        this._setupLighting();

        // 5. Load Humanoid Avatar
        await this._loadCharacter();

        // 6. Window resize listener
        window.addEventListener("resize", () => this.onResize());

        // 7. Start render loop
        this.animate();
    }

    _setupLighting() {
        // Soft ambient rim illumination
        const ambient = new THREE.AmbientLight(0xffffff, 0.65);
        this.scene.add(ambient);

        // Key light (Cyan tinted futuristic rim)
        const keyLight = new THREE.DirectionalLight(0x00f2fe, 1.1);
        keyLight.position.set(2, 4, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 1024;
        keyLight.shadow.mapSize.height = 1024;
        keyLight.shadow.bias = -0.001;
        this.scene.add(keyLight);

        // Fill light (Deep blue/violet)
        const fillLight = new THREE.DirectionalLight(0x9d4edd, 0.7);
        fillLight.position.set(-2, 2, 2);
        this.scene.add(fillLight);

        // Ground reflection / holographic pedestal disk
        const pedestalGeo = new THREE.CylinderGeometry(0.85, 0.85, 0.04, 32);
        const pedestalMat = new THREE.MeshStandardMaterial({
            color: 0x0a101d,
            roughness: 0.2,
            metalness: 0.8,
            emissive: 0x00f2fe,
            emissiveIntensity: 0.15,
        });
        const pedestal = new THREE.Mesh(pedestalGeo, pedestalMat);
        pedestal.position.set(0, -0.02, 0);
        pedestal.receiveShadow = true;
        this.scene.add(pedestal);

        // Pedestal holographic neon border ring
        const ringGeo = new THREE.TorusGeometry(0.86, 0.015, 12, 48);
        const ringMat = new THREE.MeshStandardMaterial({
            color: 0x00f2fe,
            emissive: 0x00f2fe,
            emissiveIntensity: 0.8,
        });
        const ring = new THREE.Mesh(ringGeo, ringMat);
        ring.rotation.x = Math.PI / 2;
        ring.position.set(0, 0, 0);
        this.scene.add(ring);
    }

    async _loadCharacter() {
        const { scene, isProcedural } = await this.loader.load("/static/models/avatar.glb");
        this.scene.add(scene);

        // Bind CharacterRig
        this.rig.bind(scene);

        // Initialize sub-systems
        this.animationSystem = new AnimationSystem(this.rig);
        this.gazeSystem = new GazeSystem(this.rig);
        this.facialSystem = new FacialSystem(this.rig);

        this.isLoaded = true;
        console.log("[CharacterEngine] Humanoid character system online!");
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

    // ─── Semantic Action API ────────────────────────────────────────────────

    idle() {
        if (this.animationSystem) this.animationSystem.interrupt();
        if (this.facialSystem) this.facialSystem.setEmotion("neutral");
    }

    raiseHand(side = "right", duration = 1.2) {
        if (!this.animationSystem) return;
        this.animationSystem.activeActions.upperBody = {
            type: "raise_hand",
            side: side,
            duration: duration,
            elapsed: 0,
        };
    }

    wave(side = "right", repetitions = 3, duration = 2.0) {
        if (!this.animationSystem) return;
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
        this.animationSystem.activeActions.upperBody = {
            type: "point",
            side: side,
            target: target,
            duration: duration,
            elapsed: 0,
        };
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

        if (hips) hips.position.y = 0.6;
        if (leftLeg) leftLeg.rotation.x = Math.PI / 2.2;
        if (rightLeg) rightLeg.rotation.x = Math.PI / 2.2;
        if (leftKnee) leftKnee.rotation.x = -Math.PI / 2.2;
        if (rightKnee) rightKnee.rotation.x = -Math.PI / 2.2;
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
    }

    speakStop() {
        if (this.facialSystem) this.facialSystem.stopSpeaking();
    }

    stopMotion() {
        if (this.animationSystem) this.animationSystem.interrupt();
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
            case "set_emotion":
                if (actionObj.emotion) this.setEmotion(actionObj.emotion);
                break;
            case "stop":
                this.stopMotion();
                break;
            case "idle":
                this.idle();
                break;
            default:
                console.warn(`[CharacterEngine] Unknown semantic action: ${act}`);
                break;
        }
    }
}

window.CharacterEngine = CharacterEngine;
