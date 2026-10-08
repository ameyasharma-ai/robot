/**
 * AnimationSystem & ProceduralMotionSystem
 * Implements 8 layered animation tracks:
 * - Layer 0: Base Idle
 * - Layer 1: Breathing / subtle spine & chest motion
 * - Layer 2: Locomotion (Walk / Turn cycles)
 * - Layer 3: Lower-body actions (Kicking, Weight shifting)
 * - Layer 4: Upper-body gestures (Hand raise, Waving, Pointing)
 * - Layer 5: Head movement & look reactions
 * - Layer 6: Facial expressions & Visor eye morphing
 * - Layer 7: Gaze & Look-At IK
 */

class AnimationSystem {
    constructor(rig) {
        this.rig = rig;
        this.time = 0;

        // Current active actions per layer
        this.activeActions = {
            locomotion: null,  // { type: "walk", progress: 0, speed: 1.0, direction: 1 }
            lowerBody: null,   // { type: "kick", side: "right", progress: 0, duration: 1.0, intensity: 1.0 }
            upperBody: null,   // { type: "raise_hand" | "wave" | "point", progress: 0, duration: 1.0, side: "right" }
            headReaction: null,// { yaw: 0, pitch: 0, progress: 0 }
        };

        // Procedural Idle Configuration
        this.procedural = {
            breathingRate: 0.35,      // Hz (~21 breaths per minute)
            breathingIntensity: 1.0,
            weightShiftRate: 0.12,    // Hz subtle lateral sway
            blinkTimer: 0,
            isBlinking: false,
        };

        // Cache for smooth interpolation
        this.currentUpperArmR = new THREE.Euler();
        this.currentUpperArmL = new THREE.Euler();

        // Smooth return-to-idle blend state
        // Set by interrupt() and consumed each frame until complete
        this._returnBlend = null; // { elapsed: 0, duration: 2.0, fromRotations: { boneName: Quaternion } }
    }

    /**
     * Update all animation layers for delta seconds
     */
    update(delta) {
        this.time += delta;
        const rig = this.rig;
        if (!rig || !rig.hasBone("hips")) return;

        // ─── Return-to-Idle Smooth Blend (post-interrupt) ─────────────────────
        if (this._returnBlend) {
            const rb = this._returnBlend;
            rb.elapsed += delta;
            const t = Math.min(1.0, rb.elapsed / rb.duration);
            // Ease-out cubic: fast start, gently settles
            const ease = 1.0 - Math.pow(1.0 - t, 3);
            for (const [boneName, fromQ] of Object.entries(rb.fromRotations)) {
                const bone = rig.getBone(boneName);
                const toQ = rig.initialRotations[boneName];
                if (bone && toQ) {
                    bone.quaternion.slerpQuaternions(fromQ, toQ, ease);
                }
            }
            // Also lerp positions (hips x/y)
            if (rb.fromPositions) {
                for (const [boneName, fromP] of Object.entries(rb.fromPositions)) {
                    const bone = rig.getBone(boneName);
                    const toP = rig.initialPositions[boneName];
                    if (bone && toP) {
                        bone.position.lerpVectors(fromP, toP, ease);
                    }
                }
            }
            if (t >= 1.0) {
                this._returnBlend = null; // blend complete
            }
            this._syncSecondarySleeves();
            // While blending back, skip the normal layer updates
            // so we don't fight the blend with fresh gestures
            return;
        }

        // ─── Layer 1: Breathing & Subtle Weight Shift ────────────────────────
        const breath = Math.sin(this.time * Math.PI * 2 * this.procedural.breathingRate);
        const sway = Math.sin(this.time * Math.PI * 2 * this.procedural.weightShiftRate);

        const chest = rig.getBone("chest");
        const spine = rig.getBone("spine");
        const hips = rig.getBone("hips");

        if (chest && rig.initialRotations["chest"]) {
            const deltaChest = new THREE.Quaternion().setFromEuler(new THREE.Euler(breath * 0.02 * this.procedural.breathingIntensity, sway * 0.015, 0));
            chest.quaternion.copy(rig.initialRotations["chest"]).multiply(deltaChest);
        }
        if (spine && rig.initialRotations["spine"]) {
            const deltaSpine = new THREE.Quaternion().setFromEuler(new THREE.Euler(breath * 0.015 * this.procedural.breathingIntensity, 0, sway * 0.015));
            spine.quaternion.copy(rig.initialRotations["spine"]).multiply(deltaSpine);
        }
        if (hips && !this.activeActions.locomotion && !this.activeActions.lowerBody) {
            hips.position.y = (rig.initialPositions["hips"] ? rig.initialPositions["hips"].y : 1.0) + (breath * 0.005);
            if (rig.initialRotations["hips"]) {
                const deltaHips = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -sway * 0.015));
                hips.quaternion.copy(rig.initialRotations["hips"]).multiply(deltaHips);
            }
        }

        // ─── Layer 2: Locomotion (Walk / Turn) ────────────────────────────────
        if (this.activeActions.locomotion) {
            this._updateLocomotion(delta);
        }

        // ─── Layer 3: Lower Body Actions (Kick / Weight shift) ───────────────
        if (this.activeActions.lowerBody) {
            this._updateLowerBody(delta);
        }

        // ─── Layer 4: Upper Body Gestures (Raise Hand, Wave, Point, or Expressive Speaking) ──────────
        if (this.activeActions.upperBody) {
            this._updateUpperBody(delta);
        } else if (this.isSpeaking) {
            // Expressive conversational gestures while speaking
            // Modulate right forearm & spine with cadence of speech
            const speakCycle = Math.sin(this.time * 4.2);
            const speakCycleSlow = Math.sin(this.time * 2.1);
            const leftArm = rig.getBone("leftUpperArm");
            const rightArm = rig.getBone("rightUpperArm");
            const rightForearm = rig.getBone("rightLowerArm");
            const leftForearm = rig.getBone("leftLowerArm");

            if (leftArm && rig.initialRotations["leftUpperArm"]) {
                const deltaArmL = new THREE.Quaternion().setFromEuler(new THREE.Euler(sway * 0.05 + speakCycleSlow * 0.04, 0, -1.15 + breath * 0.02));
                leftArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(deltaArmL);
            }
            if (rightArm && rig.initialRotations["rightUpperArm"]) {
                // Lift right arm slightly forward and gesture with subtle harmonic pulse
                const deltaArmR = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.25 + speakCycleSlow * 0.12, speakCycle * 0.06, 0.95 + speakCycle * 0.08));
                rightArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(deltaArmR);
            }
            if (rightForearm && rig.initialRotations["rightLowerArm"]) {
                const deltaForearmR = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -0.45 - speakCycle * 0.22));
                rightForearm.quaternion.copy(rig.initialRotations["rightLowerArm"]).multiply(deltaForearmR);
            }
            if (leftForearm && rig.initialRotations["leftLowerArm"]) {
                const deltaForearmL = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0.18 + speakCycleSlow * 0.1));
                leftForearm.quaternion.copy(rig.initialRotations["leftLowerArm"]).multiply(deltaForearmL);
            }
        } else {
            // Natural arm hanging with subtle harmonic breathing response
            // For VRM canonical bones (T-pose extending out along X):
            // Left arm rotates down with negative Z (-1.25 rad)
            // Right arm rotates down with positive Z (+1.25 rad)
            const leftArm = rig.getBone("leftUpperArm");
            const rightArm = rig.getBone("rightUpperArm");
            const rightForearm = rig.getBone("rightLowerArm");
            const leftForearm = rig.getBone("leftLowerArm");

            if (leftArm && rig.initialRotations["leftUpperArm"]) {
                const deltaArmL = new THREE.Quaternion().setFromEuler(new THREE.Euler(sway * 0.02, 0, -1.25 + breath * 0.015));
                leftArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(deltaArmL);
            }
            if (rightArm && rig.initialRotations["rightUpperArm"]) {
                const deltaArmR = new THREE.Quaternion().setFromEuler(new THREE.Euler(-sway * 0.02, 0, 1.25 - breath * 0.015));
                rightArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(deltaArmR);
            }
            if (rightForearm && rig.initialRotations["rightLowerArm"]) {
                rightForearm.quaternion.copy(rig.initialRotations["rightLowerArm"]);
            }
            if (leftForearm && rig.initialRotations["leftLowerArm"]) {
                leftForearm.quaternion.copy(rig.initialRotations["leftLowerArm"]);
            }
        }

        // Keep secondary sleeve and clothing aim bones synchronized with upper arm kinematics
        this._syncSecondarySleeves();
    }

    /**
     * Synchronize secondary clothing and sleeve aim bones with the upper arms.
     */
    _syncSecondarySleeves() {
        const rig = this.rig;
        if (!rig) return;
        const leftArm = rig.getBone("leftUpperArm");
        const rightArm = rig.getBone("rightUpperArm");
        const leftSleeve = rig.getBone("leftSleeveAim");
        const rightSleeve = rig.getBone("rightSleeveAim");
        if (leftSleeve && leftArm) leftSleeve.quaternion.copy(leftArm.quaternion);
        if (rightSleeve && rightArm) rightSleeve.quaternion.copy(rightArm.quaternion);
    }

    _updateLocomotion(delta) {
        const loc = this.activeActions.locomotion;
        loc.progress += delta * (loc.speed || 1.8);
        const cycle = Math.sin(loc.progress * Math.PI * 2);
        const rig = this.rig;

        if (loc.type === "walk") {
            const leftLeg = rig.getBone("leftUpperLeg");
            const rightLeg = rig.getBone("rightUpperLeg");
            const leftKnee = rig.getBone("leftLowerLeg");
            const rightKnee = rig.getBone("rightLowerLeg");
            const hips = rig.getBone("hips");

            const amp = 0.45 * (loc.direction || 1.0);
            if (leftLeg && rig.initialRotations["leftUpperLeg"]) {
                const initQ = rig.initialRotations["leftUpperLeg"];
                leftLeg.quaternion.copy(initQ).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-cycle * amp, 0, 0)));
            }
            if (rightLeg && rig.initialRotations["rightUpperLeg"]) {
                const initQ = rig.initialRotations["rightUpperLeg"];
                rightLeg.quaternion.copy(initQ).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(cycle * amp, 0, 0)));
            }

            // Knee flexion on back swing
            const leftFlex = Math.max(0, -cycle * 0.55);
            const rightFlex = Math.max(0, cycle * 0.55);
            if (leftKnee && rig.initialRotations["leftLowerLeg"]) {
                const initQ = rig.initialRotations["leftLowerLeg"];
                leftKnee.quaternion.copy(initQ).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-leftFlex, 0, 0)));
            }
            if (rightKnee && rig.initialRotations["rightLowerLeg"]) {
                const initQ = rig.initialRotations["rightLowerLeg"];
                rightKnee.quaternion.copy(initQ).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-rightFlex, 0, 0)));
            }

            // Pelvic bounce and counter-rotation
            if (hips && rig.initialPositions["hips"]) {
                hips.position.y = (rig.initialPositions["hips"].y || 1.0) + Math.abs(Math.cos(loc.progress * Math.PI * 2)) * 0.03;
            }

            // Arm swinging if not gesturing
            if (!this.activeActions.upperBody) {
                const lArm = rig.getBone("leftUpperArm");
                const rArm = rig.getBone("rightUpperArm");
                if (lArm && rig.initialRotations["leftUpperArm"]) {
                    lArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(cycle * 0.4, 0, 1.25)));
                }
                if (rArm && rig.initialRotations["rightUpperArm"]) {
                    rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-cycle * 0.4, 0, -1.25)));
                }
            }

            if (loc.duration && loc.progress * 0.5 >= loc.duration) {
                this.activeActions.locomotion = null;
                this.resetLimb("leftUpperLeg");
                this.resetLimb("rightUpperLeg");
                this.resetLimb("leftLowerLeg");
                this.resetLimb("rightLowerLeg");
                if (hips && rig.initialPositions["hips"]) {
                    hips.position.y = rig.initialPositions["hips"].y;
                }
            }
        } else if (loc.type === "turn") {
            const rigRoot = rig.root;
            if (rigRoot) {
                const step = (loc.targetAngle * delta) / (loc.duration || 2.0);
                rigRoot.rotation.y += step;
                if (Math.abs(rigRoot.rotation.y) >= Math.abs(loc.targetAngle)) {
                    this.activeActions.locomotion = null;
                }
            }
        }
    }

    _updateLowerBody(delta) {
        const act = this.activeActions.lowerBody;
        act.elapsed = (act.elapsed || 0) + delta;
        const t = Math.min(1.0, act.elapsed / (act.duration || 1.0));
        const rig = this.rig;

        const isRight = act.side !== "left";
        const kickLeg = rig.getBone(isRight ? "rightUpperLeg" : "leftUpperLeg");
        const kickKnee = rig.getBone(isRight ? "rightLowerLeg" : "leftLowerLeg");
        const supportLeg = rig.getBone(isRight ? "leftUpperLeg" : "rightUpperLeg");
        const hips = rig.getBone("hips");

        if (act.type === "kick") {
            // Kinetic kick trajectory: chamber knee -> snap kick forward -> retract -> return
            let legLift = 0;
            let kneeSnap = 0;

            if (t < 0.25) {
                // Chamber leg
                const sub = t / 0.25;
                legLift = sub * 0.7;
                kneeSnap = sub * 1.1; // bend knee
            } else if (t < 0.55) {
                // Snap kick forward
                const sub = (t - 0.25) / 0.3;
                legLift = 0.7 + sub * 0.45 * (act.intensity || 1.0);
                kneeSnap = 1.1 - sub * 1.05; // extend knee
            } else if (t < 0.8) {
                // Retract
                const sub = (t - 0.55) / 0.25;
                legLift = 1.15 - sub * 0.6;
                kneeSnap = 0.05 + sub * 0.6;
            } else {
                // Return to ground
                const sub = (t - 0.8) / 0.2;
                legLift = 0.55 * (1.0 - sub);
                kneeSnap = 0.65 * (1.0 - sub);
            }

            if (kickLeg && rig.initialRotations[isRight ? "rightUpperLeg" : "leftUpperLeg"]) {
                const initQ = rig.initialRotations[isRight ? "rightUpperLeg" : "leftUpperLeg"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-legLift, 0, 0));
                kickLeg.quaternion.copy(initQ).multiply(deltaQ);
            }
            if (kickKnee && rig.initialRotations[isRight ? "rightLowerLeg" : "leftLowerLeg"]) {
                const initQ = rig.initialRotations[isRight ? "rightLowerLeg" : "leftLowerLeg"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(-kneeSnap, 0, 0));
                kickKnee.quaternion.copy(initQ).multiply(deltaQ);
            }

            // Stabilize pelvic balance on supporting leg
            if (hips && rig.initialPositions["hips"]) {
                const shiftSign = isRight ? 1 : -1;
                hips.position.x = (rig.initialPositions["hips"].x || 0) + Math.sin(t * Math.PI) * 0.08 * shiftSign;
            }
        }

        if (t >= 1.0) {
            this.activeActions.lowerBody = null;
            this.resetLimb(isRight ? "rightUpperLeg" : "leftUpperLeg");
            this.resetLimb(isRight ? "rightLowerLeg" : "leftLowerLeg");
            if (hips && rig.initialPositions["hips"]) {
                hips.position.x = rig.initialPositions["hips"].x || 0;
            }
        }
    }

    _updateUpperBody(delta) {
        const act = this.activeActions.upperBody;
        act.elapsed = (act.elapsed || 0) + delta;
        const t = Math.min(1.0, act.elapsed / (act.duration || 1.2));
        const rig = this.rig;

        const isRight = act.side !== "left";
        const arm = rig.getBone(isRight ? "rightUpperArm" : "leftUpperArm");
        const forearm = rig.getBone(isRight ? "rightLowerArm" : "leftLowerArm");
        const hand = rig.getBone(isRight ? "rightHand" : "leftHand");

        // In VRM T-pose:
        // Left arm hangs at Z = -1.25 rad. Lifting up = rotating towards 0 or positive Z.
        // Right arm hangs at Z = +1.25 rad. Lifting up = rotating towards 0 or negative Z.
        const armHangZ = isRight ? 1.25 : -1.25;

        if (act.type === "raise_hand") {
            const ease = 0.5 - 0.5 * Math.cos(Math.min(1.0, t * 1.5) * Math.PI);
            if (arm && rig.initialRotations[isRight ? "rightUpperArm" : "leftUpperArm"]) {
                const initQ = rig.initialRotations[isRight ? "rightUpperArm" : "leftUpperArm"];
                // Lift arm forward and up
                const targetZ = armHangZ * (1.0 - ease) + (isRight ? -0.4 : 0.4) * ease;
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(ease * 1.2, 0, targetZ));
                arm.quaternion.copy(initQ).multiply(deltaQ);
            }
            if (forearm && rig.initialRotations[isRight ? "rightLowerArm" : "leftLowerArm"]) {
                const initQ = rig.initialRotations[isRight ? "rightLowerArm" : "leftLowerArm"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, ease * (isRight ? -0.6 : 0.6)));
                forearm.quaternion.copy(initQ).multiply(deltaQ);
            }
        } else if (act.type === "wave") {
            const raiseProgress = Math.min(1.0, t * 2.5);
            const easeRaise = 0.5 - 0.5 * Math.cos(raiseProgress * Math.PI);
            const waveCycles = (act.repetitions || 3) * 2;
            const waveProgress = Math.max(0, (act.elapsed - 0.3) * 6.0);
            const waveOsc = Math.sin(waveProgress * Math.PI);

            if (arm && rig.initialRotations[isRight ? "rightUpperArm" : "leftUpperArm"]) {
                const initQ = rig.initialRotations[isRight ? "rightUpperArm" : "leftUpperArm"];
                const targetZ = armHangZ * (1.0 - easeRaise) + (isRight ? -0.5 : 0.5) * easeRaise;
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(easeRaise * 0.9, 0, targetZ));
                arm.quaternion.copy(initQ).multiply(deltaQ);
            }
            if (forearm && rig.initialRotations[isRight ? "rightLowerArm" : "leftLowerArm"]) {
                const initQ = rig.initialRotations[isRight ? "rightLowerArm" : "leftLowerArm"];
                const waveAngle = (0.7 + waveOsc * 0.4) * (isRight ? -1 : 1);
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, waveAngle));
                forearm.quaternion.copy(initQ).multiply(deltaQ);
            }
        } else if (act.type === "point") {
            const ease = 0.5 - 0.5 * Math.cos(Math.min(1.0, t * 2.0) * Math.PI);
            if (arm && rig.initialRotations[isRight ? "rightUpperArm" : "leftUpperArm"]) {
                const initQ = rig.initialRotations[isRight ? "rightUpperArm" : "leftUpperArm"];
                const targetZ = armHangZ * (1.0 - ease) + (isRight ? -0.1 : 0.1) * ease;
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(ease * 1.5, 0, targetZ));
                arm.quaternion.copy(initQ).multiply(deltaQ);
            }
            if (forearm && rig.initialRotations[isRight ? "rightLowerArm" : "leftLowerArm"]) {
                const initQ = rig.initialRotations[isRight ? "rightLowerArm" : "leftLowerArm"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0));
                forearm.quaternion.copy(initQ).multiply(deltaQ);
            }
        } else if (act.type === "nod") {
            const head = rig.getBone("head");
            const ease = Math.sin(t * Math.PI * 4) * (1.0 - t * 0.3);
            if (head && rig.initialRotations["head"]) {
                const initQ = rig.initialRotations["head"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(ease * 0.28, 0, 0));
                head.quaternion.copy(initQ).multiply(deltaQ);
            }
        } else if (act.type === "shake_head") {
            const head = rig.getBone("head");
            const ease = Math.sin(t * Math.PI * 4) * (1.0 - t * 0.3);
            if (head && rig.initialRotations["head"]) {
                const initQ = rig.initialRotations["head"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, ease * 0.32, 0));
                head.quaternion.copy(initQ).multiply(deltaQ);
            }
        } else if (act.type === "tilt_head") {
            const head = rig.getBone("head");
            const ease = Math.sin(t * Math.PI);
            if (head && rig.initialRotations["head"]) {
                const initQ = rig.initialRotations["head"];
                const deltaQ = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.05 * ease, 0, ease * 0.25));
                head.quaternion.copy(initQ).multiply(deltaQ);
            }
        } else if (act.type === "shrug") {
            const ease = Math.sin(t * Math.PI);
            const chest = rig.getBone("chest");
            const lArm = rig.getBone("leftUpperArm");
            const rArm = rig.getBone("rightUpperArm");
            const lFore = rig.getBone("leftLowerArm");
            const rFore = rig.getBone("rightLowerArm");
            if (chest && rig.initialRotations["chest"]) {
                chest.quaternion.copy(rig.initialRotations["chest"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.06 * ease, 0, 0)));
            }
            if (lArm && rig.initialRotations["leftUpperArm"]) {
                lArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.2 * ease, 0, -1.25 + 0.45 * ease)));
            }
            if (rArm && rig.initialRotations["rightUpperArm"]) {
                rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.2 * ease, 0, 1.25 - 0.45 * ease)));
            }
            if (lFore && rig.initialRotations["leftLowerArm"]) {
                lFore.quaternion.copy(rig.initialRotations["leftLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0.5 * ease)));
            }
            if (rFore && rig.initialRotations["rightLowerArm"]) {
                rFore.quaternion.copy(rig.initialRotations["rightLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -0.5 * ease)));
            }
        } else if (act.type === "bow") {
            const ease = Math.sin(t * Math.PI);
            const spine = rig.getBone("spine");
            const chest = rig.getBone("chest");
            const head = rig.getBone("head");
            if (spine && rig.initialRotations["spine"]) {
                spine.quaternion.copy(rig.initialRotations["spine"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.35 * ease, 0, 0)));
            }
            if (chest && rig.initialRotations["chest"]) {
                chest.quaternion.copy(rig.initialRotations["chest"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.2 * ease, 0, 0)));
            }
            if (head && rig.initialRotations["head"]) {
                head.quaternion.copy(rig.initialRotations["head"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.15 * ease, 0, 0)));
            }
        } else if (act.type === "clap") {
            const ease = Math.sin(t * Math.PI);
            const clapOsc = Math.sin(t * Math.PI * 8);
            const lArm = rig.getBone("leftUpperArm");
            const rArm = rig.getBone("rightUpperArm");
            const lFore = rig.getBone("leftLowerArm");
            const rFore = rig.getBone("rightLowerArm");
            if (lArm && rig.initialRotations["leftUpperArm"]) {
                lArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.7 * ease, 0.3 * ease, -0.6 * ease)));
            }
            if (rArm && rig.initialRotations["rightUpperArm"]) {
                rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.7 * ease, -0.3 * ease, 0.6 * ease)));
            }
            if (lFore && rig.initialRotations["leftLowerArm"]) {
                lFore.quaternion.copy(rig.initialRotations["leftLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, (0.6 + clapOsc * 0.2) * ease)));
            }
            if (rFore && rig.initialRotations["rightLowerArm"]) {
                rFore.quaternion.copy(rig.initialRotations["rightLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, (-0.6 - clapOsc * 0.2) * ease)));
            }
        } else if (act.type === "think") {
            const ease = Math.sin(t * Math.PI);
            const rArm = rig.getBone("rightUpperArm");
            const rFore = rig.getBone("rightLowerArm");
            const head = rig.getBone("head");
            if (rArm && rig.initialRotations["rightUpperArm"]) {
                rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.9 * ease, 0.4 * ease, 0.2 * ease)));
            }
            if (rFore && rig.initialRotations["rightLowerArm"]) {
                rFore.quaternion.copy(rig.initialRotations["rightLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.2 * ease)));
            }
            if (head && rig.initialRotations["head"]) {
                head.quaternion.copy(rig.initialRotations["head"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.15 * ease, 0.1 * ease, 0.18 * ease)));
            }
        } else if (act.type === "cheer") {
            const ease = Math.sin(t * Math.PI);
            const bounce = Math.abs(Math.sin(t * Math.PI * 4)) * 0.04 * ease;
            const lArm = rig.getBone("leftUpperArm");
            const rArm = rig.getBone("rightUpperArm");
            const hips = rig.getBone("hips");
            if (lArm && rig.initialRotations["leftUpperArm"]) {
                lArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.5 * ease, 0, (-1.25 + 2.5 * ease))));
            }
            if (rArm && rig.initialRotations["rightUpperArm"]) {
                rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.5 * ease, 0, (1.25 - 2.5 * ease))));
            }
            if (hips && rig.initialPositions["hips"]) {
                hips.position.y = (rig.initialPositions["hips"].y || 1.0) + bounce;
            }
        } else if (act.type === "hands_on_hips") {
            const ease = Math.sin(t * Math.PI);
            const lArm = rig.getBone("leftUpperArm");
            const rArm = rig.getBone("rightUpperArm");
            const lFore = rig.getBone("leftLowerArm");
            const rFore = rig.getBone("rightLowerArm");
            if (lArm && rig.initialRotations["leftUpperArm"]) {
                lArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.2 * ease, 0, -0.6 * ease)));
            }
            if (rArm && rig.initialRotations["rightUpperArm"]) {
                rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.2 * ease, 0, 0.6 * ease)));
            }
            if (lFore && rig.initialRotations["leftLowerArm"]) {
                lFore.quaternion.copy(rig.initialRotations["leftLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 1.3 * ease)));
            }
            if (rFore && rig.initialRotations["rightLowerArm"]) {
                rFore.quaternion.copy(rig.initialRotations["rightLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.3 * ease)));
            }
        } else if (act.type === "heart_gesture") {
            const ease = Math.sin(t * Math.PI);
            const lArm = rig.getBone("leftUpperArm");
            const rArm = rig.getBone("rightUpperArm");
            const lFore = rig.getBone("leftLowerArm");
            const rFore = rig.getBone("rightLowerArm");
            if (lArm && rig.initialRotations["leftUpperArm"]) {
                lArm.quaternion.copy(rig.initialRotations["leftUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.85 * ease, 0.4 * ease, -0.3 * ease)));
            }
            if (rArm && rig.initialRotations["rightUpperArm"]) {
                rArm.quaternion.copy(rig.initialRotations["rightUpperArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0.85 * ease, -0.4 * ease, 0.3 * ease)));
            }
            if (lFore && rig.initialRotations["leftLowerArm"]) {
                lFore.quaternion.copy(rig.initialRotations["leftLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 1.4 * ease)));
            }
            if (rFore && rig.initialRotations["rightLowerArm"]) {
                rFore.quaternion.copy(rig.initialRotations["rightLowerArm"]).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.4 * ease)));
            }
        }

        const isBilateral = ["shrug", "bow", "clap", "cheer", "hands_on_hips", "heart_gesture", "nod", "shake_head", "tilt_head", "think"].includes(act.type);
        if (!isBilateral) {
            // Keep the contralateral (other) arm hanging down naturally during gesture
            const otherArm = rig.getBone(isRight ? "leftUpperArm" : "rightUpperArm");
            const otherInit = rig.initialRotations[isRight ? "leftUpperArm" : "rightUpperArm"];
            if (otherArm && otherInit) {
                const otherZ = isRight ? -1.25 : 1.25;
                const deltaOther = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, otherZ));
                otherArm.quaternion.copy(otherInit).multiply(deltaOther);
            }
        }

        if (t >= 1.0) {
            // Return to idle smoothly after duration
            this.activeActions.upperBody = null;
            this.resetLimb("rightUpperArm");
            this.resetLimb("leftUpperArm");
            this.resetLimb("rightLowerArm");
            this.resetLimb("leftLowerArm");
            this.resetLimb("head");
            this.resetLimb("spine");
            this.resetLimb("chest");
            if (rig.initialPositions["hips"]) {
                const hips = rig.getBone("hips");
                if (hips) hips.position.y = rig.initialPositions["hips"].y || 1.0;
            }
        }
    }

    resetLimb(boneName) {
        const b = this.rig.getBone(boneName);
        if (b && this.rig.initialRotations[boneName]) {
            b.quaternion.copy(this.rig.initialRotations[boneName]);
        }
    }

    interrupt() {
        // Capture current bone rotations as blend-from snapshots
        const fromRotations = {};
        const fromPositions = {};
        const rig = this.rig;
        for (const boneName of Object.keys(rig.initialRotations)) {
            const bone = rig.getBone(boneName);
            if (bone) {
                fromRotations[boneName] = bone.quaternion.clone();
            }
        }
        for (const boneName of Object.keys(rig.initialPositions)) {
            const bone = rig.getBone(boneName);
            if (bone) {
                fromPositions[boneName] = bone.position.clone();
            }
        }

        // Clear all active actions
        this.activeActions.locomotion = null;
        this.activeActions.lowerBody = null;
        this.activeActions.upperBody = null;
        this.isSpeaking = false;

        // Kick off a smooth 2-second slerp back to rest pose
        this._returnBlend = {
            elapsed: 0,
            duration: 2.0,
            fromRotations,
            fromPositions,
        };
    }
}

window.AnimationSystem = AnimationSystem;
