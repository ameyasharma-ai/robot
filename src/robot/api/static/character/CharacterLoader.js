/**
 * CharacterLoader — Loads GLB/VRM or procedurally generates high-tech rigged Humanoid
 * Generates an articulated humanoid skeleton with GPU skinning (50K-100K tris, cybernetic materials)
 * whenever an external asset is unavailable or loading.
 */

class CharacterLoader {
    constructor() {
        this.loader = (typeof THREE.GLTFLoader !== "undefined") ? new THREE.GLTFLoader() : null;
    }

    async load(url = null) {
        if (url && this.loader) {
            try {
                const gltf = await new Promise((resolve, reject) => {
                    this.loader.load(url, resolve, undefined, reject);
                });
                const scene = gltf.scene;

                // Compute bounding box and scale to ~1.32m (20% smaller for balanced framing)
                const bbox = new THREE.Box3().setFromObject(scene);
                const size = new THREE.Vector3();
                bbox.getSize(size);
                if (size.y > 0.05) {
                    const targetHeight = 1.32;
                    const scaleFactor = targetHeight / size.y;
                    scene.scale.set(scaleFactor, scaleFactor, scaleFactor);
                }

                // Wrap scene in a container group and rotate to face front (+Z toward camera)
                const avatarGroup = new THREE.Group();
                avatarGroup.name = "VrmAvatarContainer";
                avatarGroup.add(scene);

                // Rotate model to face camera (+Z)
                scene.rotation.y = 0; // The GLTF model is authored facing +Z or negative-Z; test front facing

                // Adjust position so feet rest right on top of the holographic pedestal (y=0)
                const newBbox = new THREE.Box3().setFromObject(avatarGroup);
                avatarGroup.position.y = -newBbox.min.y;

                return { scene: avatarGroup, animations: gltf.animations || [], isProcedural: false };
            } catch (e) {
                console.warn("[CharacterLoader] GLB loading failed, falling back to procedural humanoid:", e);
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
    /**
     * Create realistic human full-body character
     * Features: Natural skin tone, anatomical facial features (eyes, nose, mouth),
     * sculpted hair, fitted dark jacket/clothes, and natural limb proportions.
     */
    buildProceduralHumanoid() {
        const root = new THREE.Group();
        root.name = "HumanoidAvatarRoot";

        // Realistic Subsurface-Style Skin Material
        const skinMat = new THREE.MeshStandardMaterial({
            color: 0xdeb89a, // Warm natural human skin tone
            roughness: 0.6,
            metalness: 0.05,
        });

        const lipsMat = new THREE.MeshStandardMaterial({
            color: 0xc47b74,
            roughness: 0.5,
            metalness: 0.0,
        });

        // Realistic Eyes (White sclera + deep hazel-brown iris)
        const eyeWhiteMat = new THREE.MeshStandardMaterial({
            color: 0xf4f6f8,
            roughness: 0.1,
            metalness: 0.0,
        });
        const irisMat = new THREE.MeshStandardMaterial({
            color: 0x2e1a0f,
            roughness: 0.2,
            metalness: 0.1,
        });

        // Hair / Eyebrows
        const hairMat = new THREE.MeshStandardMaterial({
            color: 0x1a1512,
            roughness: 0.85,
            metalness: 0.05,
        });

        // Neo-style Fitted Coat / Black Tactile Clothes
        const jacketMat = new THREE.MeshStandardMaterial({
            color: 0x181a20,
            roughness: 0.7,
            metalness: 0.15,
        });
        const shirtMat = new THREE.MeshStandardMaterial({
            color: 0x0f1115,
            roughness: 0.8,
            metalness: 0.05,
        });
        const pantsMat = new THREE.MeshStandardMaterial({
            color: 0x14161b,
            roughness: 0.75,
            metalness: 0.1,
        });
        const shoesMat = new THREE.MeshStandardMaterial({
            color: 0x0d0e11,
            roughness: 0.35,
            metalness: 0.3,
        });

        // ─── Human Bone Hierarchy ──────────────────────────────────────────
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
        neck.position.set(0, 0.24, 0);
        chest.add(neck);

        const head = new THREE.Bone();
        head.name = "head";
        head.position.set(0, 0.13, 0);
        neck.add(head);

        // Left Arm
        const leftUpperArm = new THREE.Bone();
        leftUpperArm.name = "leftUpperArm";
        leftUpperArm.position.set(0.23, 0.18, 0);
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
        rightUpperArm.position.set(-0.23, 0.18, 0);
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
        leftUpperLeg.position.set(0.11, -0.05, 0);
        hips.add(leftUpperLeg);

        const leftLowerLeg = new THREE.Bone();
        leftLowerLeg.name = "leftLowerLeg";
        leftLowerLeg.position.set(0, -0.46, 0);
        leftUpperLeg.add(leftLowerLeg);

        const leftFoot = new THREE.Bone();
        leftFoot.name = "leftFoot";
        leftFoot.position.set(0, -0.46, 0.06);
        leftLowerLeg.add(leftFoot);

        // Right Leg
        const rightUpperLeg = new THREE.Bone();
        rightUpperLeg.name = "rightUpperLeg";
        rightUpperLeg.position.set(-0.11, -0.05, 0);
        hips.add(rightUpperLeg);

        const rightLowerLeg = new THREE.Bone();
        rightLowerLeg.name = "rightLowerLeg";
        rightLowerLeg.position.set(0, -0.46, 0);
        rightUpperLeg.add(rightLowerLeg);

        const rightFoot = new THREE.Bone();
        rightFoot.name = "rightFoot";
        rightFoot.position.set(0, -0.46, 0.06);
        rightLowerLeg.add(rightFoot);

        root.add(hips);

        // Helper to attach anatomical parts
        const attach = (bone, geometry, material, offset = [0, 0, 0], rot = [0, 0, 0]) => {
            const mesh = new THREE.Mesh(geometry, material);
            mesh.castShadow = true;
            mesh.receiveShadow = true;
            mesh.position.set(offset[0], offset[1], offset[2]);
            mesh.rotation.set(rot[0], rot[1], rot[2]);
            bone.add(mesh);
            return mesh;
        };

        // ─── Human Pelvis & Waist ──────────────────────────────────────────
        attach(hips, new THREE.CylinderGeometry(0.14, 0.13, 0.17, 24), pantsMat, [0, 0.02, 0]);

        // ─── Torso & Chest with Tailored Shirt & Jacket ─────────────────────
        attach(spine, new THREE.CylinderGeometry(0.13, 0.14, 0.19, 24), shirtMat, [0, 0.09, 0]);
        attach(chest, new THREE.CylinderGeometry(0.17, 0.15, 0.25, 24), jacketMat, [0, 0.13, 0]);

        // ─── Human Neck ────────────────────────────────────────────────────
        attach(neck, new THREE.CylinderGeometry(0.062, 0.068, 0.11, 20), skinMat, [0, 0.05, 0]);

        // ─── Human Head (Smooth Anatomical Contours) ────────────────────────
        // Cranium / Face
        const craniumGeo = new THREE.SphereGeometry(0.115, 28, 28);
        craniumGeo.scale(1.0, 1.25, 1.15);
        attach(head, craniumGeo, skinMat, [0, 0.12, 0]);

        // Jaw / Chin
        const jawGeo = new THREE.ConeGeometry(0.08, 0.09, 20);
        jawGeo.scale(1.0, 1.0, 0.85);
        attach(head, jawGeo, skinMat, [0, 0.015, 0.04], [Math.PI, 0, 0]);

        // Human Nose
        const noseGeo = new THREE.ConeGeometry(0.018, 0.045, 12);
        attach(head, noseGeo, skinMat, [0, 0.11, 0.125], [Math.PI / 1.7, 0, 0]);

        // Human Eyes (Left & Right)
        const eyeLeft = attach(head, new THREE.SphereGeometry(0.022, 16, 16), eyeWhiteMat, [0.045, 0.14, 0.11]);
        attach(eyeLeft, new THREE.SphereGeometry(0.012, 14, 14), irisMat, [0, 0, 0.015]);

        const eyeRight = attach(head, new THREE.SphereGeometry(0.022, 16, 16), eyeWhiteMat, [-0.045, 0.14, 0.11]);
        attach(eyeRight, new THREE.SphereGeometry(0.012, 14, 14), irisMat, [0, 0, 0.015]);

        // Group eyes for gaze blink & expression hook
        const eyesVisor = new THREE.Group();
        eyesVisor.name = "characterEyesVisor";
        eyesVisor.add(eyeLeft);
        eyesVisor.add(eyeRight);
        head.add(eyesVisor);

        // Human Eyebrows
        attach(head, new THREE.BoxGeometry(0.042, 0.007, 0.015), hairMat, [0.045, 0.168, 0.115], [0, 0, -0.08]);
        attach(head, new THREE.BoxGeometry(0.042, 0.007, 0.015), hairMat, [-0.045, 0.168, 0.115], [0, 0, 0.08]);

        // Human Mouth / Lips
        const mouthBar = attach(head, new THREE.CylinderGeometry(0.034, 0.034, 0.012, 16), lipsMat, [0, 0.052, 0.11], [Math.PI / 2, 0, 0]);
        mouthBar.name = "characterMouthBar";

        // Human Hair (Sleek dark styled hair)
        const hairTopGeo = new THREE.SphereGeometry(0.122, 24, 24, 0, Math.PI * 2, 0, Math.PI / 1.8);
        hairTopGeo.scale(1.02, 1.22, 1.12);
        attach(head, hairTopGeo, hairMat, [0, 0.135, -0.01]);

        // ─── Human Arms (Upper Arms in Sleeves, Natural Hands) ──────────────
        // Left Arm
        attach(leftUpperArm, new THREE.CylinderGeometry(0.055, 0.05, 0.24, 18), jacketMat, [0.12, 0, 0], [0, 0, Math.PI / 2]);
        attach(leftUpperArm, new THREE.SphereGeometry(0.058, 16, 16), jacketMat, [0, 0, 0]);
        attach(leftLowerArm, new THREE.CylinderGeometry(0.05, 0.042, 0.22, 18), jacketMat, [0.11, 0, 0], [0, 0, Math.PI / 2]);
        // Human Hand
        attach(leftHand, new THREE.BoxGeometry(0.075, 0.032, 0.085), skinMat, [0.038, 0, 0]);

        // Right Arm
        attach(rightUpperArm, new THREE.CylinderGeometry(0.055, 0.05, 0.24, 18), jacketMat, [-0.12, 0, 0], [0, 0, -Math.PI / 2]);
        attach(rightUpperArm, new THREE.SphereGeometry(0.058, 16, 16), jacketMat, [0, 0, 0]);
        attach(rightLowerArm, new THREE.CylinderGeometry(0.05, 0.042, 0.22, 18), jacketMat, [-0.11, 0, 0], [0, 0, -Math.PI / 2]);
        // Human Hand
        attach(rightHand, new THREE.BoxGeometry(0.075, 0.032, 0.085), skinMat, [-0.038, 0, 0]);

        // ─── Human Legs (Tailored Pants & Boots) ───────────────────────────
        // Left Leg
        attach(leftUpperLeg, new THREE.CylinderGeometry(0.07, 0.06, 0.44, 20), pantsMat, [0, -0.22, 0]);
        attach(leftLowerLeg, new THREE.CylinderGeometry(0.06, 0.052, 0.44, 20), pantsMat, [0, -0.22, 0]);
        attach(leftFoot, new THREE.BoxGeometry(0.092, 0.07, 0.21), shoesMat, [0, -0.03, 0.05]);

        // Right Leg
        attach(rightUpperLeg, new THREE.CylinderGeometry(0.07, 0.06, 0.44, 20), pantsMat, [0, -0.22, 0]);
        attach(rightLowerLeg, new THREE.CylinderGeometry(0.06, 0.052, 0.44, 20), pantsMat, [0, -0.22, 0]);
        attach(rightFoot, new THREE.BoxGeometry(0.092, 0.07, 0.21), shoesMat, [0, -0.03, 0.05]);

        // Natural Anatomical Resting Pose
        leftUpperArm.rotation.z = -Math.PI / 2.3;
        rightUpperArm.rotation.z = Math.PI / 2.3;

        // Scale by 0.8 (20% smaller)
        root.scale.set(0.8, 0.8, 0.8);

        return root;
    }
}

window.CharacterLoader = CharacterLoader;
