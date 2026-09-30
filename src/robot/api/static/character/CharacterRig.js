/**
 * CharacterRig — Standardized Humanoid Rig & Skeleton Abstraction
 * Supports standard humanoid bone mappings, VRM humanoid conventions,
 * and procedural bone fallback.
 */

class CharacterRig {
    constructor() {
        this.root = null;
        this.skeleton = null;
        this.bones = {};
        this.initialRotations = {};
        this.initialPositions = {};
        this.morphTargetMeshes = [];
        this.morphTargetDictionary = {};

        // Standard humanoid bone mapping names
        this.HUMANOID_BONE_NAMES = [
            "hips", "spine", "chest", "neck", "head",
            "leftUpperArm", "leftLowerArm", "leftHand",
            "rightUpperArm", "rightLowerArm", "rightHand",
            "leftUpperLeg", "leftLowerLeg", "leftFoot",
            "rightUpperLeg", "rightLowerLeg", "rightFoot"
        ];
    }

    /**
     * Bind a Three.js Object3D / SkinnedMesh / VRM root to humanoid rig
     */
    bind(rootObject) {
        this.root = rootObject;
        this.bones = {};
        this.initialRotations = {};
        this.initialPositions = {};
        this.morphTargetMeshes = [];
        this.morphTargetDictionary = {};

        // Scan for bones in skeleton or scene graph
        const boneMap = new Map();
        rootObject.traverse((node) => {
            if (node.isBone) {
                boneMap.set(node.name.toLowerCase().replace(/[-_ ]/g, ""), node);
            }
            if (node.isSkinnedMesh) {
                if (node.skeleton && !this.skeleton) {
                    this.skeleton = node.skeleton;
                }
                if (node.morphTargetDictionary) {
                    this.morphTargetMeshes.push(node);
                    Object.assign(this.morphTargetDictionary, node.morphTargetDictionary);
                }
            }
        });

        // Bone name aliases dictionary with exact VRM standard names prioritized
        const ALIASES = {
            hips: ["j_bip_c_hips", "hips", "pelvis", "root", "hip"],
            spine: ["j_bip_c_spine", "spine", "spine1", "lowertorso"],
            chest: ["j_bip_c_chest", "chest", "spine2", "uppertorso", "torso"],
            neck: ["j_bip_c_neck", "neck"],
            head: ["j_bip_c_head", "head"],
            leftUpperArm: ["j_bip_l_upperarm", "leftupperarm", "leftarm", "armleft", "l_upperarm", "upperarm_l"],
            leftLowerArm: ["j_bip_l_lowerarm", "leftlowerarm", "leftforearm", "forearmleft", "l_forearm", "lowerarm_l"],
            leftHand: ["j_bip_l_hand", "lefthand", "handleft", "l_hand", "hand_l", "wrist_l"],
            rightUpperArm: ["j_bip_r_upperarm", "rightupperarm", "rightarm", "armright", "r_upperarm", "upperarm_r"],
            rightLowerArm: ["j_bip_r_lowerarm", "rightlowerarm", "rightforearm", "forearmright", "r_forearm", "lowerarm_r"],
            rightHand: ["j_bip_r_hand", "righthand", "handright", "r_hand", "hand_r", "wrist_r"],
            leftUpperLeg: ["j_bip_l_upperleg", "leftupperleg", "leftup_leg", "leftthigh", "l_thigh", "upperleg_l"],
            leftLowerLeg: ["j_bip_l_lowerleg", "leftlowerleg", "leftleg", "leftcalf", "l_calf", "lowerleg_l"],
            leftFoot: ["j_bip_l_foot", "leftfoot", "footleft", "l_foot", "foot_l", "ankle_l"],
            rightUpperLeg: ["j_bip_r_upperleg", "rightupperleg", "rightup_leg", "rightthigh", "r_thigh", "upperleg_r"],
            rightLowerLeg: ["j_bip_r_lowerleg", "rightlowerleg", "rightleg", "rightcalf", "r_calf", "lowerleg_r"],
            rightFoot: ["j_bip_r_foot", "rightfoot", "footright", "r_foot", "foot_r", "ankle_r"],
        };

        // Match bones with exact-match priority first
        for (const [canonical, aliases] of Object.entries(ALIASES)) {
            for (const alias of aliases) {
                const norm = alias.toLowerCase().replace(/[-_ ]/g, "");
                for (const [bName, bObj] of boneMap.entries()) {
                    if (bName === norm) {
                        this.bones[canonical] = bObj;
                        break;
                    }
                }
                if (this.bones[canonical]) break;
            }
            // Fallback substring search if exact match didn't find bone
            if (!this.bones[canonical]) {
                for (const alias of aliases) {
                    const norm = alias.toLowerCase().replace(/[-_ ]/g, "");
                    for (const [bName, bObj] of boneMap.entries()) {
                        if (bName.includes(norm)) {
                            this.bones[canonical] = bObj;
                            break;
                        }
                    }
                    if (this.bones[canonical]) break;
                }
            }
        }

        // Cache rest/initial transform poses
        for (const [name, bone] of Object.entries(this.bones)) {
            this.initialRotations[name] = bone.quaternion.clone();
            this.initialPositions[name] = bone.position.clone();
        }

        console.log(`[CharacterRig] Mapped ${Object.keys(this.bones).length} bones:`, Object.keys(this.bones));
        return Object.keys(this.bones).length > 0;
    }

    getBone(name) {
        return this.bones[name] || null;
    }

    hasBone(name) {
        return !!this.bones[name];
    }

    resetPose() {
        for (const [name, bone] of Object.entries(this.bones)) {
            if (this.initialRotations[name]) {
                bone.quaternion.copy(this.initialRotations[name]);
            }
            if (this.initialPositions[name]) {
                bone.position.copy(this.initialPositions[name]);
            }
        }
    }

    setMorphWeight(name, weight) {
        for (const mesh of this.morphTargetMeshes) {
            const idx = mesh.morphTargetDictionary[name];
            if (idx !== undefined && mesh.morphTargetInfluences) {
                mesh.morphTargetInfluences[idx] = Math.max(0, Math.min(1, weight));
            }
        }
    }
}

window.CharacterRig = CharacterRig;
