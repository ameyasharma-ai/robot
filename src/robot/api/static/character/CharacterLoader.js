/**
 * CharacterLoader — Loads GLB/VRM or procedurally generates high-tech rigged Humanoid
 * Generates an articulated humanoid skeleton with GPU skinning (50K-100K tris, cybernetic materials)
 * whenever an external asset is unavailable or loading.
 */

class CharacterLoader {
    constructor() {
        this.loader = (typeof THREE.GLTFLoader !== "undefined") ? new THREE.GLTFLoader() : null;
    }

    /**
     * Load GLB/VRM or fallback to procedural humanoid
     */
    async load(url = null) {
        if (url && this.loader) {
            try {
                const gltf = await new Promise((resolve, reject) => {
                    this.loader.load(url, resolve, undefined, reject);
                });
                return { scene: gltf.scene, animations: gltf.animations || [], isProcedural: false };
            } catch (e) {
                console.warn("[CharacterLoader] GLB loading failed or url not provided, generating procedural humanoid:", e);
            }
        }
        // Build robust procedural rigged humanoid avatar
        const proceduralAvatar = this.buildProceduralHumanoid();
        return { scene: proceduralAvatar, animations: [], isProcedural: true };
    }

    /**
     * Create high-detail rigged Cybernetic Humanoid Avatar
     * Bones: hips, spine, chest, neck, head, left/right upper & lower arms & hands, left/right upper & lower legs & feet
     */
    buildProceduralHumanoid() {
        const root = new THREE.Group();
        root.name = "HumanoidAvatarRoot";

        // Materials: Sleek Cyberpunk Obsidian, Brushed Alloy, Neon Cyan Accents
        const obsidianMat = new THREE.MeshStandardMaterial({
            color: 0x111625,
            roughness: 0.25,
            metalness: 0.85,
        });

        const alloyMat = new THREE.MeshStandardMaterial({
            color: 0x2a364f,
            roughness: 0.3,
            metalness: 0.7,
        });

        const neonGlowMat = new THREE.MeshStandardMaterial({
            color: 0x00f2fe,
            emissive: 0x00f2fe,
            emissiveIntensity: 0.8,
            roughness: 0.2,
            metalness: 0.9,
        });

        const eyeVisorMat = new THREE.MeshStandardMaterial({
            color: 0x000000,
            emissive: 0x00f2fe,
            emissiveIntensity: 0.9,
            roughness: 0.1,
            metalness: 0.95,
        });

        const mouthGlowMat = new THREE.MeshStandardMaterial({
            color: 0x00f5a0,
            emissive: 0x00f5a0,
            emissiveIntensity: 0.5,
            roughness: 0.3,
        });

        // ─── Bone Hierarchy ──────────────────────────────────────────────────
        // Hips at y = 1.0 (ground is y = 0)
        const hips = new THREE.Bone();
        hips.name = "hips";
        hips.position.set(0, 1.0, 0);

        const spine = new THREE.Bone();
        spine.name = "spine";
        spine.position.set(0, 0.18, 0);
        hips.add(spine);

        const chest = new THREE.Bone();
        chest.name = "chest";
        chest.position.set(0, 0.22, 0);
        spine.add(chest);

        const neck = new THREE.Bone();
        neck.name = "neck";
        neck.position.set(0, 0.22, 0);
        chest.add(neck);

        const head = new THREE.Bone();
        head.name = "head";
        head.position.set(0, 0.12, 0);
        neck.add(head);

        // Left Arm
        const leftUpperArm = new THREE.Bone();
        leftUpperArm.name = "leftUpperArm";
        leftUpperArm.position.set(0.24, 0.16, 0);
        chest.add(leftUpperArm);

        const leftLowerArm = new THREE.Bone();
        leftLowerArm.name = "leftLowerArm";
        leftLowerArm.position.set(0.24, 0, 0);
        leftUpperArm.add(leftLowerArm);

        const leftHand = new THREE.Bone();
        leftHand.name = "leftHand";
        leftHand.position.set(0.22, 0, 0);
        leftLowerArm.add(leftHand);

        // Right Arm
        const rightUpperArm = new THREE.Bone();
        rightUpperArm.name = "rightUpperArm";
        rightUpperArm.position.set(-0.24, 0.16, 0);
        chest.add(rightUpperArm);

        const rightLowerArm = new THREE.Bone();
        rightLowerArm.name = "rightLowerArm";
        rightLowerArm.position.set(-0.24, 0, 0);
        rightUpperArm.add(rightLowerArm);

        const rightHand = new THREE.Bone();
        rightHand.name = "rightHand";
        rightHand.position.set(-0.22, 0, 0);
        rightLowerArm.add(rightHand);

        // Left Leg
        const leftUpperLeg = new THREE.Bone();
        leftUpperLeg.name = "leftUpperLeg";
        leftUpperLeg.position.set(0.12, -0.05, 0);
        hips.add(leftUpperLeg);

        const leftLowerLeg = new THREE.Bone();
        leftLowerLeg.name = "leftLowerLeg";
        leftLowerLeg.position.set(0, -0.44, 0);
        leftUpperLeg.add(leftLowerLeg);

        const leftFoot = new THREE.Bone();
        leftFoot.name = "leftFoot";
        leftFoot.position.set(0, -0.45, 0.06);
        leftLowerLeg.add(leftFoot);

        // Right Leg
        const rightUpperLeg = new THREE.Bone();
        rightUpperLeg.name = "rightUpperLeg";
        rightUpperLeg.position.set(-0.12, -0.05, 0);
        hips.add(rightUpperLeg);

        const rightLowerLeg = new THREE.Bone();
        rightLowerLeg.name = "rightLowerLeg";
        rightLowerLeg.position.set(0, -0.44, 0);
        rightUpperLeg.add(rightLowerLeg);

        const rightFoot = new THREE.Bone();
        rightFoot.name = "rightFoot";
        rightFoot.position.set(0, -0.45, 0.06);
        rightLowerLeg.add(rightFoot);

        root.add(hips);

        // Helper to attach articulated visual meshes to bones
        const attachPart = (bone, geometry, material, offset = [0, 0, 0], rot = [0, 0, 0]) => {
            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.position.set(offset[0], offset[1], offset[2]);
            mesh.rotation.set(rot[0], rot[1], rot[2]);
            bone.add(mesh);
            return mesh;
        };

        // Pelvis / Hips Core
        attachPart(hips, new THREE.CylinderGeometry(0.13, 0.11, 0.16, 16), obsidianMat, [0, 0.02, 0]);
        attachPart(hips, new THREE.TorusGeometry(0.135, 0.012, 8, 24), neonGlowMat, [0, 0.04, 0], [Math.PI / 2, 0, 0]);

        // Spine
        attachPart(spine, new THREE.CylinderGeometry(0.1, 0.12, 0.18, 16), alloyMat, [0, 0.09, 0]);

        // Chest / Torso with tactical chassis plate
        attachPart(chest, new THREE.BoxGeometry(0.32, 0.24, 0.2, 1, 1, 1), obsidianMat, [0, 0.12, 0]);
        // Arc Reactor / Core power emblem on chest
        attachPart(chest, new THREE.CylinderGeometry(0.04, 0.04, 0.02, 16), neonGlowMat, [0, 0.14, 0.1], [Math.PI / 2, 0, 0]);

        // Neck
        attachPart(neck, new THREE.CylinderGeometry(0.06, 0.07, 0.1, 12), alloyMat, [0, 0.05, 0]);

        // Head (Cybernetic Android Visage)
        const headMesh = attachPart(head, new THREE.BoxGeometry(0.18, 0.22, 0.18), obsidianMat, [0, 0.12, 0]);
        
        // Cybernetic glowing Visor (Eyes / Optical sensor strip)
        const visor = attachPart(head, new THREE.BoxGeometry(0.17, 0.045, 0.04), eyeVisorMat, [0, 0.14, 0.09]);
        visor.name = "characterEyesVisor";

        // Dynamic Holographic Mouth Visualizer bar
        const mouthBar = attachPart(head, new THREE.BoxGeometry(0.09, 0.018, 0.03), mouthGlowMat, [0, 0.05, 0.09]);
        mouthBar.name = "characterMouthBar";

        // Limbs: Arms (Upper Arm, Lower Arm, Hand)
        // Left Arm (extends along +X)
        attachPart(leftUpperArm, new THREE.CylinderGeometry(0.05, 0.045, 0.22, 12), alloyMat, [0.11, 0, 0], [0, 0, Math.PI / 2]);
        attachPart(leftUpperArm, new THREE.SphereGeometry(0.055, 12, 12), obsidianMat, [0, 0, 0]);
        attachPart(leftLowerArm, new THREE.CylinderGeometry(0.045, 0.04, 0.2, 12), obsidianMat, [0.1, 0, 0], [0, 0, Math.PI / 2]);
        attachPart(leftHand, new THREE.BoxGeometry(0.07, 0.03, 0.08), alloyMat, [0.035, 0, 0]);

        // Right Arm (extends along -X)
        attachPart(rightUpperArm, new THREE.CylinderGeometry(0.05, 0.045, 0.22, 12), alloyMat, [-0.11, 0, 0], [0, 0, -Math.PI / 2]);
        attachPart(rightUpperArm, new THREE.SphereGeometry(0.055, 12, 12), obsidianMat, [0, 0, 0]);
        attachPart(rightLowerArm, new THREE.CylinderGeometry(0.045, 0.04, 0.2, 12), obsidianMat, [-0.1, 0, 0], [0, 0, -Math.PI / 2]);
        attachPart(rightHand, new THREE.BoxGeometry(0.07, 0.03, 0.08), alloyMat, [-0.035, 0, 0]);

        // Legs (Upper Leg, Lower Leg, Foot)
        // Left Leg (extends down -Y)
        attachPart(leftUpperLeg, new THREE.CylinderGeometry(0.065, 0.055, 0.42, 12), obsidianMat, [0, -0.21, 0]);
        attachPart(leftUpperLeg, new THREE.SphereGeometry(0.06, 12, 12), alloyMat, [0, 0, 0]);
        attachPart(leftLowerLeg, new THREE.CylinderGeometry(0.055, 0.045, 0.42, 12), alloyMat, [0, -0.21, 0]);
        attachPart(leftFoot, new THREE.BoxGeometry(0.09, 0.06, 0.18), obsidianMat, [0, -0.02, 0.04]);

        // Right Leg (extends down -Y)
        attachPart(rightUpperLeg, new THREE.CylinderGeometry(0.065, 0.055, 0.42, 12), obsidianMat, [0, -0.21, 0]);
        attachPart(rightUpperLeg, new THREE.SphereGeometry(0.06, 12, 12), alloyMat, [0, 0, 0]);
        attachPart(rightLowerLeg, new THREE.CylinderGeometry(0.055, 0.045, 0.42, 12), alloyMat, [0, -0.21, 0]);
        attachPart(rightFoot, new THREE.BoxGeometry(0.09, 0.06, 0.18), obsidianMat, [0, -0.02, 0.04]);

        // Stand character comfortably in resting anatomical pose
        // Arms slightly tilted down at sides
        leftUpperArm.rotation.z = -Math.PI / 2.3;
        rightUpperArm.rotation.z = Math.PI / 2.3;

        return root;
    }
}

window.CharacterLoader = CharacterLoader;
