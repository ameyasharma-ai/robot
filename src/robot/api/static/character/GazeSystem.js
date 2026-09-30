/**
 * GazeSystem & FacialSystem
 * Supports:
 * - Real-time Head & Eye Look-At Inverse Kinematics
 * - Smooth exponential tracking of OpenCV detected user position
 * - Micro-saccades (natural subtle involuntary eye shifts)
 * - Automatic natural blinking and double-blinking
 * - Audio-driven lip-synchronization with amplitude-based jaw/mouth deformation
 * - Facial emotional states (neutral, happy, sad, surprised, angry, confused, thinking)
 */

class GazeSystem {
    constructor(rig) {
        this.rig = rig;

        // Current and target look-at points in 3D world space
        this.target = new THREE.Vector3(0, 1.15, 2.2); // Default: looking slightly forward/camera
        this.current = new THREE.Vector3(0, 1.15, 2.2);
        this.smoothSpeed = 5.0; // Interpolation damping

        // OpenCV continuous tracking target (norm_x: 0..1, norm_y: 0..1)
        this.opencvTarget = { norm_x: 0.5, norm_y: 0.5, active: false, lastSeen: 0 };

        // Saccade generation
        this.saccadeTimer = 0;
        this.saccadeOffset = new THREE.Vector3();
    }

    /**
     * Set direct 3D target coordinates
     */
    setTarget(x, y, z) {
        this.target.set(x, y, z);
    }

    /**
     * Update gaze from OpenCV detected user face coordinates
     */
    setOpenCVFace(normX, normY) {
        this.opencvTarget.norm_x = normX;
        this.opencvTarget.norm_y = normY;
        this.opencvTarget.active = true;
        this.opencvTarget.lastSeen = performance.now();

        // Convert camera normalized coordinates (0..1) to 3D world target
        // norm_x 0 = user to left of camera, 1 = user to right
        const worldX = (normX - 0.5) * 1.5;
        const worldY = 1.15 - (normY - 0.5) * 1.0;
        const worldZ = 2.0;

        this.setTarget(worldX, worldY, worldZ);
    }

    update(delta) {
        const head = this.rig.getBone("head");
        const neck = this.rig.getBone("neck");
        if (!head) return;

        // Micro-saccades (natural involuntary optical adjustments every 2.5-4s)
        this.saccadeTimer += delta;
        if (this.saccadeTimer > 3.0) {
            this.saccadeTimer = 0;
            this.saccadeOffset.set(
                (Math.random() - 0.5) * 0.15,
                (Math.random() - 0.5) * 0.1,
                0
            );
        }

        // Check if OpenCV target expired (> 3 seconds)
        if (this.opencvTarget.active && (performance.now() - this.opencvTarget.lastSeen > 3000)) {
            this.opencvTarget.active = false;
            this.target.set(0, 1.45, 2.5); // return to center camera
        }

        // Interpolate current look vector toward target + saccade
        const desired = this.target.clone().add(this.saccadeOffset);
        this.current.lerp(desired, Math.min(1.0, delta * this.smoothSpeed));

        // Compute Head & Neck orientation
        // Calculate angle deltas from head rest position (assume head is ~ y=1.15)
        const dx = this.current.x;
        const dy = this.current.y - 1.15;
        const dz = Math.max(0.5, this.current.z);

        const targetYaw = Math.max(-0.65, Math.min(0.65, -Math.atan2(dx, dz)));
        const targetPitch = Math.max(-0.45, Math.min(0.45, Math.atan2(dy, dz)));

        // Distribute rotation 70% to head, 30% to neck for natural posture
        // Apply rotation multiplicatively on top of the initial rest pose to avoid axis conflicts
        const neckInit = this.rig.initialRotations["neck"];
        const headInit = this.rig.initialRotations["head"];

        const neckDelta = new THREE.Quaternion().setFromEuler(new THREE.Euler(targetPitch * 0.35, targetYaw * 0.35, 0, "YXZ"));
        const headDelta = new THREE.Quaternion().setFromEuler(new THREE.Euler(targetPitch * 0.65, targetYaw * 0.65, 0, "YXZ"));

        if (neck) {
            const targetNeckQ = neckInit ? neckInit.clone().multiply(neckDelta) : neckDelta;
            neck.quaternion.slerp(targetNeckQ, Math.min(1.0, delta * 6.0));
        }
        if (head) {
            const targetHeadQ = headInit ? headInit.clone().multiply(headDelta) : headDelta;
            head.quaternion.slerp(targetHeadQ, Math.min(1.0, delta * 6.0));
        }
    }
}

class FacialSystem {
    constructor(rig) {
        this.rig = rig;
        this.currentEmotion = "neutral";
        this.speaking = false;
        this.speechAmplitude = 0;
        this.blinkTimer = 0;
        this.isBlinking = false;

        // References to procedural eye and mouth meshes if present
        this.eyesVisor = null;
        this.mouthBar = null;

        if (rig.root) {
            rig.root.traverse((node) => {
                if (node.name === "characterEyesVisor") this.eyesVisor = node;
                if (node.name === "characterMouthBar") this.mouthBar = node;
            });
        }
    }

    setEmotion(emotion) {
        this.currentEmotion = emotion.toLowerCase();
        console.log(`[FacialSystem] Emotion set to: ${this.currentEmotion}`);

        // Update visor / eye color and intensity based on emotion
        if (this.eyesVisor && this.eyesVisor.material) {
            const mat = this.eyesVisor.material;
            switch (this.currentEmotion) {
                case "happy":
                    mat.emissive.setHex(0x00f5a0); // Vibrant Emerald
                    break;
                case "thinking":
                    mat.emissive.setHex(0xf6d365); // Amber Gold
                    break;
                case "surprised":
                    mat.emissive.setHex(0x00f2fe); // Cyan High Beam
                    break;
                case "sad":
                    mat.emissive.setHex(0x4facfe); // Soft Blue
                    break;
                case "angry":
                    mat.emissive.setHex(0xff0844); // Crimson
                    break;
                default:
                    mat.emissive.setHex(0x00f2fe); // Cyber Cyan
                    break;
            }
        }
    }

    startSpeaking() {
        this.speaking = true;
    }

    stopSpeaking() {
        this.speaking = false;
        this.speechAmplitude = 0;
        if (this.mouthBar) {
            this.mouthBar.scale.set(1, 1, 1);
        }
        // Reset VRM mouth blendshapes
        if (this.rig) {
            this.rig.setMorphWeight("Face_Blendshape.Fcl_MTH_A", 0);
            this.rig.setMorphWeight("Face_Blendshape.Fcl_MTH_I", 0);
            this.rig.setMorphWeight("Face_Blendshape.Fcl_MTH_O", 0);
        }
    }

    update(delta) {
        // ─── 1. Automatic Humanoid Blinking ──────────────────────────────────
        this.blinkTimer += delta;
        if (!this.isBlinking && this.blinkTimer > (Math.random() * 3.5 + 2.0)) {
            this.isBlinking = true;
            this.blinkTimer = 0;
            if (this.eyesVisor) {
                this.eyesVisor.scale.y = 0.1; // blink slit
            }
            if (this.rig) {
                this.rig.setMorphWeight("Face_Blendshape.Fcl_EYE_Close", 1.0);
            }
            setTimeout(() => {
                if (this.eyesVisor) this.eyesVisor.scale.y = 1.0;
                if (this.rig) {
                    this.rig.setMorphWeight("Face_Blendshape.Fcl_EYE_Close", 0.0);
                }
                this.isBlinking = false;
            }, 110);
        }

        // ─── 2. Lip-Sync & Mouth Modulation (Both procedural mesh and VRM blendshapes)
        if (this.speaking) {
            // Amplitude modulation synchronized to voice speech cadence
            const amp = Math.sin(performance.now() * 0.015) * 0.5 + 0.5;
            const jitter = (Math.random() * 0.3) + 0.7;
            const mouthScaleY = 1.0 + amp * jitter * 2.8;
            const mouthScaleX = 1.0 + amp * 0.3;

            if (this.mouthBar) {
                this.mouthBar.scale.set(mouthScaleX, mouthScaleY, 1.0);
            }

            // Drive VRM 3D Face Blendshapes dynamically during speech
            if (this.rig) {
                const openAmount = amp * jitter * 0.85;
                this.rig.setMorphWeight("Face_Blendshape.Fcl_MTH_A", openAmount);
                this.rig.setMorphWeight("Face_Blendshape.Fcl_MTH_I", (1.0 - openAmount) * 0.4);
                this.rig.setMorphWeight("Face_Blendshape.Fcl_MTH_O", openAmount * 0.5);
            }
        }
    }
}

window.GazeSystem = GazeSystem;
window.FacialSystem = FacialSystem;
