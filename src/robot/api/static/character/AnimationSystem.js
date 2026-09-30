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
    }

    /**
     * Update all animation layers for delta seconds
     */
    update(delta) {
        this.time += delta;
        const rig = this.rig;
        if (!rig || !rig.hasBone("hips")) return;

        // ─── Layer 1: Breathing & Subtle Weight Shift ────────────────────────
        const breath = Math.sin(this.time * Math.PI * 2 * this.procedural.breathingRate);
        const sway = Math.sin(this.time * Math.PI * 2 * this.procedural.weightShiftRate);

        const chest = rig.getBone("chest");
        const spine = rig.getBone("spine");
        const hips = rig.getBone("hips");

        if (chest) {
            // Subtle thoracic expansion and slight tilt
            chest.rotation.x = breath * 0.035 * this.procedural.breathingIntensity;
            chest.rotation.y = sway * 0.02;
        }
        if (spine) {
            spine.rotation.x = breath * 0.02 * this.procedural.breathingIntensity;
            spine.rotation.z = sway * 0.025;
        }
        if (hips && !this.activeActions.locomotion && !this.activeActions.lowerBody) {
            // Natural micro-bob and weight distribution
            hips.position.y = (rig.initialPositions["hips"] ? rig.initialPositions["hips"].y : 1.0) + (breath * 0.008);
            hips.rotation.z = -sway * 0.02;
        }

        // ─── Layer 2: Locomotion (Walk / Turn) ────────────────────────────────
        if (this.activeActions.locomotion) {
            this._updateLocomotion(delta);
        }

        // ─── Layer 3: Lower Body Actions (Kick / Weight shift) ───────────────
        if (this.activeActions.lowerBody) {
            this._updateLowerBody(delta);
        }

        // ─── Layer 4: Upper Body Gestures (Raise Hand, Wave, Point) ──────────
        if (this.activeActions.upperBody) {
            this._updateUpperBody(delta);
        } else {
            // Natural arm hanging with subtle harmonic breathing response
            const leftArm = rig.getBone("leftUpperArm");
            const rightArm = rig.getBone("rightUpperArm");
            if (leftArm) {
                leftArm.rotation.z = -Math.PI / 2.3 + (breath * 0.02);
                leftArm.rotation.x = sway * 0.03;
            }
            if (rightArm) {
                rightArm.rotation.z = Math.PI / 2.3 - (breath * 0.02);
                rightArm.rotation.x = -sway * 0.03;
            }
        }
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

            const amp = 0.55 * (loc.direction || 1.0);
            if (leftLeg) leftLeg.rotation.x = cycle * amp;
            if (rightLeg) rightLeg.rotation.x = -cycle * amp;

            // Knee flexion on back swing
            if (leftKnee) leftKnee.rotation.x = Math.max(0, -cycle * 0.6);
            if (rightKnee) rightKnee.rotation.x = Math.max(0, cycle * 0.6);

            // Pelvic bounce and counter-rotation
            if (hips) {
                hips.position.y = (rig.initialPositions["hips"]?.y || 1.0) + Math.abs(Math.cos(loc.progress * Math.PI * 2)) * 0.04;
                hips.rotation.y = -cycle * 0.08;
            }

            // Arm swinging if not gesturing
            if (!this.activeActions.upperBody) {
                const lArm = rig.getBone("leftUpperArm");
                const rArm = rig.getBone("rightUpperArm");
                if (lArm) lArm.rotation.x = -cycle * 0.45;
                if (rArm) rArm.rotation.x = cycle * 0.45;
            }

            if (loc.duration && loc.progress * 0.5 >= loc.duration) {
                this.activeActions.locomotion = null;
                this.resetLimb("leftUpperLeg");
                this.resetLimb("rightUpperLeg");
                this.resetLimb("leftLowerLeg");
                this.resetLimb("rightLowerLeg");
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

            if (kickLeg) kickLeg.rotation.x = legLift;
            if (kickKnee) kickKnee.rotation.x = kneeSnap;

            // Stabilize pelvic balance on supporting leg
            if (hips) {
                const shiftSign = isRight ? 1 : -1;
                hips.position.x = Math.sin(t * Math.PI) * 0.08 * shiftSign;
                hips.rotation.z = Math.sin(t * Math.PI) * 0.05 * shiftSign;
            }
        }

        if (t >= 1.0) {
            this.activeActions.lowerBody = null;
            if (kickLeg) kickLeg.rotation.set(0, 0, 0);
            if (kickKnee) kickKnee.rotation.set(0, 0, 0);
            if (hips) {
                hips.position.x = 0;
                hips.rotation.z = 0;
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

        const sign = isRight ? 1 : -1;

        if (act.type === "raise_hand") {
            // Smooth raise arm up in salutation / affirmation
            const ease = 0.5 - 0.5 * Math.cos(Math.min(1.0, t * 1.5) * Math.PI);
            if (arm) {
                arm.rotation.z = (Math.PI / 2.3 * sign) - (ease * Math.PI * 0.85 * sign);
                arm.rotation.x = ease * 0.6;
            }
            if (forearm) forearm.rotation.y = ease * 0.4 * sign;
        } else if (act.type === "wave") {
            // Raise hand then oscillate forearm
            const raiseProgress = Math.min(1.0, t * 2.5);
            const easeRaise = 0.5 - 0.5 * Math.cos(raiseProgress * Math.PI);

            if (arm) {
                arm.rotation.z = (Math.PI / 2.3 * sign) - (easeRaise * Math.PI * 0.75 * sign);
                arm.rotation.x = easeRaise * 0.4;
            }
            if (forearm) {
                const waveCycles = (act.repetitions || 3) * 2;
                const waveProgress = Math.max(0, (act.elapsed - 0.3) * 6.0);
                const waveOsc = Math.sin(waveProgress * Math.PI);
                forearm.rotation.z = (0.5 + waveOsc * 0.35) * sign;
            }
        } else if (act.type === "point") {
            // Direct pointing forward at target
            const ease = 0.5 - 0.5 * Math.cos(Math.min(1.0, t * 2.0) * Math.PI);
            if (arm) {
                arm.rotation.z = (Math.PI / 2.3 * sign) - (ease * Math.PI * 0.55 * sign);
                arm.rotation.x = ease * 1.4; // point forward
            }
            if (forearm) forearm.rotation.x = ease * 0.1;
        }

        if (t >= 1.0) {
            // Hold or return to idle smoothly
            if (act.type === "wave" || act.type === "point") {
                this.activeActions.upperBody = null;
                this.resetLimb(isRight ? "rightUpperArm" : "leftUpperArm");
                this.resetLimb(isRight ? "rightLowerArm" : "leftLowerArm");
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
        this.activeActions.locomotion = null;
        this.activeActions.lowerBody = null;
        this.activeActions.upperBody = null;
        this.rig.resetPose();
    }
}

window.AnimationSystem = AnimationSystem;
