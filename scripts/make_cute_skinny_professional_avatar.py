"""Generate perfected Skinny, Cute, and Professional avatar assets and pack into avatar.glb.

Aesthetics:
1. Tops: Slender, tailored Midnight Slate / Royal Navy Cashmere-Wool Blazer with satin peak lapels,
         metallic gold edge piping, crisp Ivory Silk high-neck blouse, cute dusty-rose silk ribbon tie
         with crystalline pearl & gold brooch, embossed gold crest buttons, and folded silk pocket square.
2. Bottoms: High-waisted tailored pencil trousers/skirt with crisp front creases, rich Italian cognac leather
            belt, brushed gold buckle, and slant pocket welts.
3. Shoes: Burnished Italian espresso-black leather designer pumps with gold buckle accent and stacked leather sole.
4. Body: Pure radiant porcelain-peach human skin with matching trousers & cognac belt seamlessly painted
         across the hips and pelvis (zero bra/underwear/waistband artifacts).
5. Face: Glowing, soft peach skin with delicate cheek blush, nose-tip blush, professional winged eyeliner,
         feathered individual lashes, and sweet glossy hydrating strawberry-rose lips.
6. Eyes: Breathtaking crystalline gemstone eyes with deep velvet limbal ring, royal sapphire blending
         to warm golden-amber honey, radiating starburst fibers, and sparkling diamond catchlights.
7. Geometry: Slenderized Tops and Bottoms mesh vertices tightly hugging the humanoid body.
"""

from __future__ import annotations

import json
import math
import os
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURE_DIR = Path("src/robot/api/static/models/textures")
TEXTURE_DIR.mkdir(parents=True, exist_ok=True)


def generate_normal_map_from_height(height_arr: np.ndarray, strength: float = 3.0) -> Image.Image:
    """Generate a tangent-space normal map from a 2D float height/bump map using Sobel filters.

    Args:
        height_arr: 2D numpy array of float height values [0, 255].
        strength: Gradient amplification factor.

    Returns:
        PIL Image in RGB format representing normal map (flat = 128, 128, 255).
    """
    h, w = height_arr.shape
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    padded = np.pad(height_arr, 1, mode="edge")

    dx = (
        padded[:-2, 2:] * sobel_x[0, 2] + padded[1:-1, 2:] * sobel_x[1, 2] + padded[2:, 2:] * sobel_x[2, 2]
        + padded[:-2, :-2] * sobel_x[0, 0] + padded[1:-1, :-2] * sobel_x[1, 0] + padded[2:, :-2] * sobel_x[2, 0]
    ) / 8.0

    dy = (
        padded[2:, :-2] * sobel_y[2, 0] + padded[2:, 1:-1] * sobel_y[2, 1] + padded[2:, 2:] * sobel_y[2, 2]
        + padded[:-2, :-2] * sobel_y[0, 0] + padded[:-2, 1:-1] * sobel_y[0, 1] + padded[:-2, 2:] * sobel_y[0, 2]
    ) / 8.0

    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones((h, w), dtype=float) * 255.0

    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= norm
    ny /= norm
    nz /= norm

    r = ((nx * 0.5 + 0.5) * 255.0).astype(np.uint8)
    g = ((ny * 0.5 + 0.5) * 255.0).astype(np.uint8)
    b = ((nz * 0.5 + 0.5) * 255.0).astype(np.uint8)

    return Image.fromarray(np.stack([r, g, b], axis=-1))


def generate_skinny_professional_tops() -> Path:
    """Generate 1024x2048 tailored Midnight Slate blazer with ivory blouse and cute ribbon brooch.

    Returns:
        Path to the saved diffuse texture file.
    """
    w, h = 1024, 2048
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(101)

    # 1. Base Fabric: Fine diagonal twill weave in Midnight Slate/Navy (#1C2333)
    twill = (np.sin((x_g + y_g) * 0.8) + np.sin((x_g - y_g) * 0.8)) * 3.5
    noise = np.random.randn(h, w) * 2.0

    base_r = np.clip(28.0 + twill + noise, 18, 55)
    base_g = np.clip(35.0 + twill * 0.95 + noise, 24, 65)
    base_b = np.clip(52.0 + twill * 1.1 + noise, 38, 90)

    # Slenderizing lateral waist contouring
    dist_from_center = np.abs(x_g - 512) / 512.0
    slimming = 1.0 - (dist_from_center**1.8) * 0.22
    base_r *= slimming
    base_g *= slimming
    base_b *= slimming

    img = Image.fromarray(np.stack([base_r, base_g, base_b], axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    bump_img = Image.new("L", (w, h), 128)
    bump_draw = ImageDraw.Draw(bump_img)

    # 2. Inner Blouse: Soft Ivory Silk (#FBF8F3) in upper center V-opening (Y: 140 to 420)
    for y in range(140, 420):
        dy = (y - 140) / 280.0
        hw = int(math.sin(dy * math.pi) * 88.0 + (1.0 - dy) * 48.0)
        x_l, x_r = 512 - hw, 512 + hw
        for x in range(x_l, x_r + 1):
            silk_crease = math.sin((y + x * 0.2) * 0.3) * 3.0
            r_c = np.clip(250 + silk_crease, 235, 255)
            g_c = np.clip(246 + silk_crease * 0.95, 230, 252)
            b_c = np.clip(238 + silk_crease * 0.9, 220, 248)

            # Soft shading along inner border
            border_dist = min(x - x_l, x_r - x, 420 - y)
            if border_dist < 10:
                sh = border_dist / 10.0
                r_c = int(r_c * (0.85 + 0.15 * sh))
                g_c = int(g_c * (0.85 + 0.15 * sh))
                b_c = int(b_c * (0.85 + 0.15 * sh))
            img.putpixel((x, y), (int(r_c), int(g_c), int(b_c)))
            bump_img.putpixel((x, y), int(140 + silk_crease * 2))

    # Blouse collar folded wings (Y: 140 to 230)
    collar_col = (255, 252, 248)
    draw.polygon([(512, 210), (450, 140), (512, 160)], fill=collar_col, outline=(220, 215, 205))
    draw.polygon([(512, 210), (574, 140), (512, 160)], fill=collar_col, outline=(220, 215, 205))

    # 3. Cute Dusty-Rose Silk Ribbon Necktie (#D97D90) (Y: 220 to 390)
    ribbon_col = (217, 125, 144)
    ribbon_dark = (168, 88, 105)
    ribbon_hl = (238, 155, 172)
    # Knot / center
    draw.ellipse([502, 232, 522, 252], fill=ribbon_dark)
    draw.ellipse([504, 234, 520, 250], fill=ribbon_col)
    bump_draw.ellipse([502, 232, 522, 252], fill=210)

    # Ribbon wings
    draw.polygon([(504, 242), (440, 226), (436, 254)], fill=ribbon_col, outline=ribbon_dark)
    draw.polygon([(520, 242), (584, 226), (588, 254)], fill=ribbon_col, outline=ribbon_dark)
    bump_draw.polygon([(504, 242), (440, 226), (436, 254)], fill=185)
    bump_draw.polygon([(520, 242), (584, 226), (588, 254)], fill=185)

    # Ribbon tails draping down
    draw.polygon([(506, 248), (488, 380), (506, 372)], fill=ribbon_col, outline=ribbon_dark)
    draw.polygon([(518, 248), (536, 380), (518, 372)], fill=ribbon_col, outline=ribbon_dark)
    draw.line([(506, 248), (488, 380)], fill=ribbon_hl, width=2)
    draw.line([(518, 248), (536, 380)], fill=ribbon_hl, width=2)
    bump_draw.polygon([(506, 248), (488, 380), (506, 372)], fill=175)
    bump_draw.polygon([(518, 248), (536, 380), (518, 372)], fill=175)

    # 4. Crystalline Pearl & Gold Brooch at center knot (X: 512, Y: 242)
    gold_outline = (185, 142, 38)
    gold_fill = (226, 185, 68)
    draw.ellipse([503, 233, 521, 251], fill=gold_fill, outline=gold_outline)
    draw.ellipse([506, 236, 518, 248], fill=(255, 255, 255))
    draw.ellipse([507, 237, 512, 242], fill=(255, 255, 255))
    bump_draw.ellipse([503, 233, 521, 251], fill=240)

    # 5. Tailored Satin Peak Lapels with Gold Edge Piping (Y: 140 to 760)
    gold_piping = (220, 180, 60)
    lapel_satin = (22, 28, 42)
    # Left lapel
    draw.polygon([(512, 420), (380, 270), (420, 140), (452, 140), (512, 420)], fill=lapel_satin)
    draw.line([(452, 140), (420, 140), (380, 270), (512, 420)], fill=gold_piping, width=3)
    # Right lapel
    draw.polygon([(512, 420), (644, 270), (604, 140), (572, 140), (512, 420)], fill=lapel_satin)
    draw.line([(572, 140), (604, 140), (644, 270), (512, 420)], fill=gold_piping, width=3)
    bump_draw.line([(452, 140), (420, 140), (380, 270), (512, 420)], fill=210, width=4)
    bump_draw.line([(572, 140), (604, 140), (644, 270), (512, 420)], fill=210, width=4)

    # 6. Tailored Breast Welt Pocket & Silk Pocket Square (Left Chest, X: 335 to 415, Y: 430 to 470)
    draw.polygon([(345, 430), (375, 395), (400, 428)], fill=(255, 255, 255), outline=gold_piping)
    draw.rectangle([335, 430, 415, 442], fill=lapel_satin, outline=gold_piping)
    bump_draw.rectangle([335, 430, 415, 442], fill=190)

    # 7. Embossed Metallic Gold Crest Buttons down center closure
    gold_light = (255, 235, 145)
    gold_mid = (218, 172, 52)
    gold_shadow = (135, 98, 22)
    for by in [460, 560, 660, 760]:
        bx = 512
        r_b = 13
        draw.ellipse([bx - r_b - 2, by - r_b + 2, bx + r_b + 2, by + r_b + 4], fill=(12, 15, 22))
        draw.ellipse([bx - r_b, by - r_b, bx + r_b, by + r_b], fill=gold_shadow)
        draw.ellipse([bx - r_b + 2, by - r_b + 2, bx + r_b - 2, by + r_b - 2], fill=gold_mid)
        # Specular shine
        draw.ellipse([bx - 6, by - 6, bx - 1, by - 1], fill=gold_light)
        # Royal crest monogram cross
        draw.line([(bx, by - 5), (bx, by + 5)], fill=gold_shadow, width=2)
        draw.line([(bx - 5, by), (bx + 5, by)], fill=gold_shadow, width=2)
        bump_draw.ellipse([bx - r_b, by - r_b, bx + r_b, by + r_b], fill=245)

    # 8. Slender Tailored Princess Seams (dart lines from shoulders to waist)
    seam_col = (18, 22, 32)
    draw.line([(370, 240), (410, 520), (425, 850)], fill=seam_col, width=2)
    draw.line([(654, 240), (614, 520), (599, 850)], fill=seam_col, width=2)
    bump_draw.line([(370, 240), (410, 520), (425, 850)], fill=90, width=2)
    bump_draw.line([(654, 240), (614, 520), (599, 850)], fill=90, width=2)

    # 9. Clean Tapered Blazer Hem (Y: 980 to 1020) with fine gold edge stitch
    for x in range(0, w, 8):
        draw.line([(x, 1000), (x + 4, 1000)], fill=gold_piping, width=2)

    diff_path = TEXTURE_DIR / "realistic_tops.png"
    nml_path = TEXTURE_DIR / "realistic_tops_nml.png"

    img.save(diff_path, quality=95)
    normal_map = generate_normal_map_from_height(np.array(bump_img, dtype=float), strength=2.8)
    normal_map.save(nml_path, quality=95)

    print(f"Skinny Professional Tops saved: {diff_path}")
    print(f"Skinny Professional Tops Normal Map saved: {nml_path}")
    return diff_path


def generate_skinny_professional_bottoms() -> Path:
    """Generate 1024x1024 tailored pencil trousers with Italian cognac leather belt and gold buckle.

    Returns:
        Path to the saved diffuse texture file.
    """
    w, h = 1024, 1024
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(202)

    # 1. Base Fabric: Matching Midnight Slate Twill (#1C2333)
    twill = (np.sin((x_g + y_g) * 0.8) + np.sin((x_g - y_g) * 0.8)) * 3.2
    noise = np.random.randn(h, w) * 1.8

    base_r = np.clip(28.0 + twill + noise, 18, 55)
    base_g = np.clip(35.0 + twill * 0.95 + noise, 24, 65)
    base_b = np.clip(52.0 + twill * 1.1 + noise, 38, 90)

    img = Image.fromarray(np.stack([base_r, base_g, base_b], axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    bump_img = Image.new("L", (w, h), 128)
    bump_draw = ImageDraw.Draw(bump_img)

    # 2. Italian Cognac / Saddle Leather Belt (Y: 10 to 65)
    for y in range(10, 65):
        edge_dist = min(y - 10, 65 - y)
        shade = 0.82 + (edge_dist / 27.5) * 0.18
        for x in range(w):
            leather_grain = math.sin(x * 0.5) * 4.0
            r_val = np.clip((118.0 + leather_grain) * shade, 60, 160)
            g_val = np.clip((68.0 + leather_grain * 0.7) * shade, 35, 105)
            b_val = np.clip((42.0 + leather_grain * 0.5) * shade, 20, 75)
            img.putpixel((x, y), (int(r_val), int(g_val), int(b_val)))
            bump_img.putpixel((x, y), int(160 + edge_dist * 1.5))

    # Top & bottom gold contrast saddle-stitching on belt
    for x in range(0, w, 8):
        draw.line([(x, 14), (x + 4, 14)], fill=(215, 175, 80), width=1)
        draw.line([(x, 61), (x + 4, 61)], fill=(215, 175, 80), width=1)

    # 3. Brushed Gold Rectangular Buckle & Leather Loop (X: 512, Y: 37)
    gold_outer = (185, 142, 38)
    gold_inner = (235, 195, 82)
    # Buckle frame
    draw.rectangle([482, 18, 542, 56], fill=gold_outer, outline=(125, 90, 20), width=2)
    draw.rectangle([492, 26, 532, 48], fill=(90, 52, 32))  # Inner aperture revealing belt
    draw.line([(484, 20), (540, 20)], fill=(255, 235, 145), width=2)  # Highlight top bar
    # Buckle prong
    draw.line([(512, 18), (512, 56)], fill=(255, 240, 160), width=3)
    # Leather keeper loop
    draw.rectangle([552, 14, 568, 61], fill=(102, 56, 32), outline=(72, 38, 20))
    bump_draw.rectangle([482, 18, 542, 56], fill=230)
    bump_draw.rectangle([492, 26, 532, 48], fill=160)

    # 4. Crisp Pressed Front Vertical Creases (X: 320, 704 from Y: 65 to 550)
    for py in range(65, 550):
        for cx in [320, 704]:
            # Highlight on left side of crease, shadow on right
            c_hl = img.getpixel((cx - 1, py))
            c_sh = img.getpixel((cx + 1, py))
            img.putpixel((cx - 1, py), (min(255, c_hl[0] + 16), min(255, c_hl[1] + 16), min(255, c_hl[2] + 20)))
            img.putpixel((cx + 1, py), (max(0, c_sh[0] - 14), max(0, c_sh[1] - 14), max(0, c_sh[2] - 14)))
            bump_img.putpixel((cx, py), 180)

    # 5. Tailored Slant Pocket Welts with Gold Bar Tacks (Y: 90 to 220)
    for side, sx in [(-1, 210), (1, 814)]:
        draw.line([(sx, 90), (sx + side * 45, 210)], fill=(15, 18, 26), width=3)
        draw.line([(sx - 2, 90), (sx + 2, 90)], fill=(215, 175, 80), width=3)
        draw.line([(sx + side * 45 - 2, 210), (sx + side * 45 + 2, 210)], fill=(215, 175, 80), width=3)
        bump_draw.line([(sx, 90), (sx + side * 45, 210)], fill=160, width=3)

    diff_path = TEXTURE_DIR / "realistic_bottoms.png"
    nml_path = TEXTURE_DIR / "realistic_bottoms_nml.png"

    img.save(diff_path, quality=95)
    normal_map = generate_normal_map_from_height(np.array(bump_img, dtype=float), strength=2.8)
    normal_map.save(nml_path, quality=95)

    print(f"Skinny Professional Bottoms saved: {diff_path}")
    print(f"Skinny Professional Bottoms Normal Map saved: {nml_path}")
    return diff_path


def generate_professional_shoes() -> Path:
    """Generate 512x512 burnished espresso-black Italian leather pumps/shoes with gold accents.

    Returns:
        Path to the saved shoes texture.
    """
    w, h = 512, 512
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(303)

    noise = np.random.randn(h, w) * 1.5
    base_r = np.clip(22.0 + noise, 12, 45)
    base_g = np.clip(22.0 + noise, 12, 45)
    base_b = np.clip(26.0 + noise, 16, 55)

    # Burnished leather sheen gradient
    vert_sheen = np.sin(y_g / 512.0 * math.pi) * 18.0
    base_r += vert_sheen * 0.9
    base_g += vert_sheen * 0.9
    base_b += vert_sheen * 1.1

    img = Image.fromarray(np.stack([base_r, base_g, base_b], axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Contrast stacked leather welt sole
    draw.rectangle([0, 470, w, 512], fill=(62, 38, 24), outline=(42, 24, 14))
    for x in range(0, w, 12):
        draw.line([(x, 475), (x + 6, 475)], fill=(185, 145, 95), width=1)

    # Polished gold heel tap & horsebit buckle accent
    gold_fill = (215, 175, 80)
    for bx in [128, 384]:
        draw.rectangle([bx - 30, 240, bx + 30, 260], fill=gold_fill, outline=(145, 110, 30))
        draw.ellipse([bx - 12, 244, bx + 12, 256], fill=(22, 22, 26))

    out_path = TEXTURE_DIR / "realistic_shoes.png"
    img.save(out_path, quality=95)
    print(f"Professional Shoes saved: {out_path}")
    return out_path


def generate_flawless_body_skin() -> Path:
    """Ensure 2048x2048 body texture has pure radiant skin and seamless trousers/belt on hips.

    Returns:
        Path to the saved body texture.
    """
    body_path = TEXTURE_DIR / "realistic_body.png"
    if not body_path.exists():
        im = Image.new("RGB", (2048, 2048), (250, 222, 208))
    else:
        im = Image.open(body_path).convert("RGB")

    arr = np.array(im)
    np.random.seed(42)

    # 1. Cleanse ALL dark bra/underwear artifact pixels across the entire body texture
    dark_mask = (arr[:, :, 0] < 180) | (arr[:, :, 1] < 155) | (arr[:, :, 2] < 135)
    noise = (np.random.randn(2048, 2048) * 1.5).astype(np.int16)
    arr[dark_mask, 0] = np.clip(250 + noise[dark_mask], 230, 255).astype(np.uint8)
    arr[dark_mask, 1] = np.clip(222 + (noise[dark_mask] * 0.9).astype(np.int16), 200, 245).astype(np.uint8)
    arr[dark_mask, 2] = np.clip(208 + (noise[dark_mask] * 0.8).astype(np.int16), 185, 235).astype(np.uint8)

    # 2. Seamlessly paint the Cognac Leather Belt across hips (Y: 900 to 950)
    for y in range(900, 950):
        edge_dist = min(y - 900, 950 - y)
        shade = 0.82 + (edge_dist / 25.0) * 0.18
        arr[y, :, 0] = int(118 * shade)
        arr[y, :, 1] = int(68 * shade)
        arr[y, :, 2] = int(42 * shade)

    # Belt gold top/bottom stitching
    arr[904, ::8, 0] = 215
    arr[904, ::8, 1] = 175
    arr[904, ::8, 2] = 80
    arr[946, ::8, 0] = 215
    arr[946, ::8, 1] = 175
    arr[946, ::8, 2] = 80

    # 3. Seamlessly paint the Midnight Slate tailored trousers across pelvic & hip zone (Y: 950 to 1320)
    for y in range(950, 1320):
        dy_frac = (y - 950) / 370.0
        sh = 0.90 + dy_frac * 0.10
        arr[y, :, 0] = int(28 * sh)
        arr[y, :, 1] = int(35 * sh)
        arr[y, :, 2] = int(52 * sh)

    fixed_img = Image.fromarray(arr)
    fixed_img.save(body_path, quality=95)
    print(f"Body skin & seamless trousers saved: {body_path}")
    return body_path


def generate_cute_professional_face() -> Path:
    """Generate 1024x1024 radiant face texture with soft blush, winged eyeliner, and glossy coral lips.

    Returns:
        Path to the saved face texture.
    """
    w, h = 1024, 1024
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(123)

    pore_noise = np.random.randn(h, w) * 1.2
    base_r = np.clip(252.0 + pore_noise, 232, 255)
    base_g = np.clip(224.0 + pore_noise * 0.92, 198, 248)
    base_b = np.clip(212.0 + pore_noise * 0.82, 180, 238)
    arr = np.stack([base_r, base_g, base_b], axis=-1).astype(float)

    # 1. Soft Rosy-Coral Cheekbone Blush (#F09088)
    for cx in [350, 674]:
        dist_sq = ((x_g - cx) / 78.0)**2 + ((y_g - 665) / 44.0)**2
        blush = np.exp(-0.5 * dist_sq)
        arr[:, :, 0] += blush * 30.0
        arr[:, :, 1] += blush * 12.0
        arr[:, :, 2] += blush * 14.0

    # 2. Nose-Tip Cute Glow
    nose_dist = ((x_g - 512) / 20.0)**2 + ((y_g - 730) / 14.0)**2
    nose_blush = np.exp(-0.5 * nose_dist)
    arr[:, :, 0] += nose_blush * 20.0
    arr[:, :, 1] += nose_blush * 8.0
    arr[:, :, 2] += nose_blush * 10.0

    # Nose ridge specular highlight
    nose_hl = np.exp(-0.5 * (((x_g - 512) / 8.0)**2 + ((y_g - 722) / 9.0)**2))
    arr[:, :, 0] += nose_hl * 20.0
    arr[:, :, 1] += nose_hl * 18.0
    arr[:, :, 2] += nose_hl * 16.0

    # 3. Chin Radiant Glow
    chin_dist = ((x_g - 512) / 85.0)**2 + ((y_g - 880) / 55.0)**2
    chin_glow = np.exp(-0.5 * chin_dist)
    arr[:, :, 0] += chin_glow * 14.0
    arr[:, :, 1] += chin_glow * 12.0
    arr[:, :, 2] += chin_glow * 12.0

    # 4. Sweet Glossy Hydrated Strawberry-Rose Lips (#DF5874) (Y: 760 to 845, X: 445 to 579)
    for y in range(760, 845):
        for x in range(445, 579):
            dx = (x - 512) / 54.0
            dy = (y - 802) / 30.0
            r_dist = dx**2 + dy**2
            if r_dist < 1.0:
                lip_mask = math.cos(r_dist * (math.pi / 2))
                crease = math.sin(x * 1.8) * 2.5
                lip_r = 230.0 + crease
                lip_g = 115.0 + crease * 0.6
                lip_b = 135.0 + crease * 0.7

                # Center hydration gloss reflection
                if 814 < y < 832 and 500 < x < 524:
                    gloss = math.sin((y - 814) / 18.0 * math.pi) * math.sin((x - 500) / 24.0 * math.pi)
                    lip_r += gloss * 38.0
                    lip_g += gloss * 36.0
                    lip_b += gloss * 36.0

                arr[y, x, 0] = arr[y, x, 0] * (1.0 - lip_mask) + lip_r * lip_mask
                arr[y, x, 1] = arr[y, x, 1] * (1.0 - lip_mask) + lip_g * lip_mask
                arr[y, x, 2] = arr[y, x, 2] * (1.0 - lip_mask) + lip_b * lip_mask

    # Mouth seam crease line
    for x in range(468, 557):
        dx_val = (x - 512) / 44.0
        y_crease = int(801 - abs(dx_val)**2 * 2.5)
        fade = max(0.0, 1.0 - abs(dx_val))
        if 0 <= y_crease < h:
            arr[y_crease - 1:y_crease + 2, x, 0] -= fade * 22.0
            arr[y_crease - 1:y_crease + 2, x, 1] -= fade * 38.0
            arr[y_crease - 1:y_crease + 2, x, 2] -= fade * 38.0

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 5. Professional Winged Eyeliner & Feathered Individual Lashes
    for ex, sign in [(340, -1), (684, 1)]:
        eyelash_col = (36, 22, 18)
        points = []
        for t in np.linspace(0, 1, 30):
            px = ex + sign * (48 - t * 105)
            py = 502 - math.sin(t * math.pi) * 22.0
            points.append((px, py))
        for p1, p2 in zip(points[:-1], points[1:]):
            draw.line([p1, p2], fill=eyelash_col, width=4)

        # Feline wing flick
        wing_start = points[-1]
        wing_end = (wing_start[0] - sign * 16, wing_start[1] - 10)
        draw.line([wing_start, wing_end], fill=eyelash_col, width=3)

        # Individual delicate upper lashes
        for lash_idx in range(5, 26, 4):
            lp = points[lash_idx]
            angle = (lash_idx / 30.0 - 0.5) * 0.8
            lx = lp[0] + math.sin(angle) * 13.0 * sign
            ly = lp[1] - math.cos(angle) * 11.0
            draw.line([lp, (lx, ly)], fill=eyelash_col, width=2)

    out_path = TEXTURE_DIR / "realistic_face.png"
    img.save(out_path, quality=95)
    print(f"Cute Professional Face saved: {out_path}")
    return out_path


def generate_crystalline_gemstone_eyes() -> Path:
    """Generate 1024x512 stunning anime/doll crystalline iris texture with jewel depth.

    Returns:
        Path to the saved iris texture.
    """
    w, h = 1024, 512
    iris_centers = [(286, 256), (736, 256)]
    radius = 152

    arr = np.zeros((h, w, 3), dtype=float)
    y_g, x_g = np.ogrid[:h, :w]

    for cx, cy in iris_centers:
        dx = x_g - cx
        dy = y_g - cy
        dist = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        limbal_mask = dist <= radius
        norm_dist = np.clip(dist / radius, 0, 1)

        # Starburst radiating micro-fibers
        fibers = (
            np.sin(theta * 32.0) * 16.0
            + np.sin(theta * 64.0) * 12.0
            + np.sin(theta * 128.0) * 8.0
            + np.cos(theta * 16.0 + norm_dist * 8.0) * 14.0
        )
        rings = np.sin(norm_dist * 22.0) * 10.0

        vert_fade = np.clip((dy / radius) * 0.5 + 0.5, 0, 1)

        # Royal Sapphire upper iris (#2E4A96)
        top_r = 46.0 + fibers * 0.4
        top_g = 74.0 + fibers * 0.5
        top_b = 150.0 + fibers * 0.8

        # Luminous Warm Golden-Amber Honey lower iris (#E29632)
        bot_r = 226.0 + fibers + rings
        bot_g = 150.0 + fibers * 0.8 + rings * 0.8
        bot_b = 50.0 + fibers * 0.35 + rings * 0.3

        iris_r = top_r * (1.0 - vert_fade) + bot_r * vert_fade
        iris_g = top_g * (1.0 - vert_fade) + bot_g * vert_fade
        iris_b = top_b * (1.0 - vert_fade) + bot_b * vert_fade

        # Velvet limbal ring
        limbal_factor = np.clip((1.0 - norm_dist) / 0.12, 0, 1)
        iris_r *= 0.16 + 0.84 * limbal_factor
        iris_g *= 0.15 + 0.85 * limbal_factor
        iris_b *= 0.22 + 0.78 * limbal_factor

        # Velvet black pupil
        pupil_factor = np.clip((norm_dist - 0.20) / 0.06, 0, 1)
        iris_r = iris_r * pupil_factor + 4.0 * (1.0 - pupil_factor)
        iris_g = iris_g * pupil_factor + 4.0 * (1.0 - pupil_factor)
        iris_b = iris_b * pupil_factor + 6.0 * (1.0 - pupil_factor)

        # Golden starburst collarette halo
        halo_dist = np.abs(norm_dist - 0.32) / 0.10
        halo = np.exp(-0.5 * (halo_dist**2))
        iris_r += halo * 45.0
        iris_g += halo * 34.0
        iris_b += halo * 8.0

        arr[limbal_mask, 0] = np.clip(iris_r[limbal_mask], 0, 255)
        arr[limbal_mask, 1] = np.clip(iris_g[limbal_mask], 0, 255)
        arr[limbal_mask, 2] = np.clip(iris_b[limbal_mask], 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Sparkling diamond star catchlights
    for cx, cy in iris_centers:
        # 1. Main soft crescent catchlight (upper-right)
        draw.ellipse([cx + 22, cy - 68, cx + 64, cy - 26], fill=(255, 255, 255))
        draw.ellipse([cx + 28, cy - 62, cx + 58, cy - 32], fill=(255, 255, 255))

        # 2. Secondary delicate diamond star glint (lower-left)
        gx, gy = cx - 44, cy + 46
        draw.line([(gx - 10, gy), (gx + 10, gy)], fill=(255, 255, 255), width=2)
        draw.line([(gx, gy - 10), (gx, gy + 10)], fill=(255, 255, 255), width=2)
        draw.ellipse([gx - 5, gy - 5, gx + 5, gy + 5], fill=(255, 255, 255))

        # 3. Tertiary micro-glint
        draw.ellipse([cx - 24, cy - 50, cx - 14, cy - 40], fill=(255, 252, 254))

    out_path = TEXTURE_DIR / "realistic_iris.png"
    img.save(out_path, quality=95)
    print(f"Crystalline Gemstone Eyes saved: {out_path}")
    return out_path


def tighten_and_pack_glb() -> None:
    """Tighten clothing mesh vertices for slender tailored fit and pack all textures into avatar.glb."""
    src_glb = Path("src/robot/api/static/models/avatar.glb.bak")
    out_glb = Path("src/robot/api/static/models/avatar.glb")

    with open(src_glb, "rb") as f:
        magic = f.read(4)
        assert magic == b"glTF", "Not a valid glTF file"
        version = int.from_bytes(f.read(4), "little")
        total_len = int.from_bytes(f.read(4), "little")
        chunk_len = int.from_bytes(f.read(4), "little")
        chunk_type = f.read(4)
        assert chunk_type == b"JSON", "First chunk not JSON"
        json_data = json.loads(f.read(chunk_len).decode("utf-8"))
        bin_len = int.from_bytes(f.read(4), "little")
        bin_type = f.read(4)
        assert bin_type == b"BIN\x00", "Second chunk not BIN"
        bin_data = bytearray(f.read(bin_len))

    # 1. TIGHTEN TOPS_01_CLOTH VERTICES (bufferView 19, 717 vertices)
    bv19 = json_data["bufferViews"][19]
    top_off = bv19.get("byteOffset", 0)
    top_count = json_data["accessors"][19]["count"]

    for i in range(top_count):
        vx, vy, vz = struct.unpack_from("<3f", bin_data, top_off + i * 12)

        # Torso depth slimming (form-fitting tailored blazer)
        vz = 0.01 + (vz - 0.01) * 0.72

        # Taper sleeves inward towards arm bone axis for sharp tailoring
        if abs(vx) > 0.13:
            sign = 1.0 if vx > 0 else -1.0
            arm_center_x = sign * 0.19
            vx = arm_center_x + (vx - arm_center_x) * 0.60
            vz *= 0.70

        # TUCK IN LOWER HEM & FLAPS (Y <= 1.05) - eliminate flared flaps
        if vy <= 1.05:
            h_frac = max(0.0, (1.05 - vy) / 0.22)
            pull_in = 1.0 - (h_frac * 0.28)
            vx *= pull_in
            vz *= pull_in

        struct.pack_into("<3f", bin_data, top_off + i * 12, vx, vy, vz)

    print(f"Tightened {top_count} Tops vertices for tailored blazer fit!")

    # 2. TIGHTEN BOTTOMS_01_CLOTH VERTICES (bufferView 7, 470 vertices)
    bv7 = json_data["bufferViews"][7]
    bot_off = bv7.get("byteOffset", 0)
    bot_count = json_data["accessors"][7]["count"]

    for i in range(bot_count):
        vx, vy, vz = struct.unpack_from("<3f", bin_data, bot_off + i * 12)

        # Slender high-waisted pencil silhouette
        vx *= 0.82
        vz *= 0.78

        if vy < 0.85:
            vx *= 0.86
            vz *= 0.86

        struct.pack_into("<3f", bin_data, bot_off + i * 12, vx, vy, vz)

    print(f"Tightened {bot_count} Bottoms vertices for tailored pencil fit!")

    # 3. PACK UPDATED TEXTURES INTO GLB
    bv879_offset = json_data["bufferViews"][879]["byteOffset"]
    new_bin = bytearray(bin_data[:bv879_offset])

    trans_img_path = Path("/tmp/avatar_textures/transparent.png")
    trans_img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(trans_img_path)

    replace_files = {
        0: TEXTURE_DIR / "realistic_body.png",
        2: TEXTURE_DIR / "realistic_bottoms.png",
        3: TEXTURE_DIR / "realistic_shoes.png",
        4: TEXTURE_DIR / "realistic_tops.png",
        5: TEXTURE_DIR / "realistic_hairback.png",
        8: TEXTURE_DIR / "realistic_iris.png",
        9: TEXTURE_DIR / "realistic_face.png",
        10: trans_img_path,
        11: TEXTURE_DIR / "realistic_eyewhite.png",
        14: trans_img_path,
        15: TEXTURE_DIR / "realistic_hair.png",
    }

    current_offset = bv879_offset
    for i, img_meta in enumerate(json_data["images"]):
        bv_idx = img_meta["bufferView"]
        target_path = replace_files.get(i)
        if target_path and target_path.exists():
            with open(target_path, "rb") as rf:
                img_bytes = rf.read()
        else:
            old_bv = json_data["bufferViews"][bv_idx]
            o_off = old_bv["byteOffset"]
            o_len = old_bv["byteLength"]
            img_bytes = bin_data[o_off : o_off + o_len]

        pad = (4 - (len(img_bytes) % 4)) % 4
        padded_bytes = img_bytes + b"\x00" * pad

        json_data["bufferViews"][bv_idx]["byteOffset"] = current_offset
        json_data["bufferViews"][bv_idx]["byteLength"] = len(img_bytes)

        new_bin.extend(padded_bytes)
        current_offset += len(padded_bytes)

    json_data["buffers"][0]["byteLength"] = len(new_bin)

    json_bytes = json.dumps(json_data, separators=(",", ":")).encode("utf-8")
    json_pad = (4 - (len(json_bytes) % 4)) % 4
    json_bytes_padded = json_bytes + b" " * json_pad

    out_total_len = 12 + 8 + len(json_bytes_padded) + 8 + len(new_bin)

    with open(out_glb, "wb") as out:
        out.write(b"glTF")
        out.write((2).to_bytes(4, "little"))
        out.write(out_total_len.to_bytes(4, "little"))
        out.write(len(json_bytes_padded).to_bytes(4, "little"))
        out.write(b"JSON")
        out.write(json_bytes_padded)
        out.write(len(new_bin).to_bytes(4, "little"))
        out.write(b"BIN\x00")
        out.write(new_bin)

    print(f"Successfully packed tightened Skinny Cute Professional avatar into {out_glb} ({out_total_len} bytes)!")


if __name__ == "__main__":
    generate_flawless_body_skin()
    generate_skinny_professional_tops()
    generate_skinny_professional_bottoms()
    generate_professional_shoes()
    generate_cute_professional_face()
    generate_crystalline_gemstone_eyes()
    tighten_and_pack_glb()
