"""
Generate perfected skinny, cute avatar assets:
1. Tops: Skinny, form-fitting blush-pink ribbed sweetheart crop sweater with rich pastel contrast,
         scalloped lace trim, pearl buttons, and cute ribbon bow (not blown out).
2. Bottoms: Skinny pleated skater/tennis mini-skirt with vertical knife pleats, pastel pink belt,
            silver heart buckle, and scalloped lace hem.
3. Body: Clean porcelain-peach body skin with ZERO dark bra band or underwear artifacts!
         Seamlessly paints the cute pastel pleated skirt across the hips and thighs!
4. Face: Radiant, soft peach skin with ZERO chin/lower face shadow, sweet strawberry glossy lips,
         refined winged eyeliner and delicate feathered eyelashes.
5. Eyes: Breathtakingly beautiful anime/doll crystalline iris with deep velvet limbal ring,
         luminous sapphire-to-amber jewel gradient, radiating starburst fibers, and sparkling diamond catchlights.
6. Geometry: Tighten Tops and Bottoms mesh vertices so they tightly hug the slender body curves.
"""

import os
import math
import json
import struct
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TEXTURE_DIR = "src/robot/api/static/models/textures"
os.makedirs(TEXTURE_DIR, exist_ok=True)


def fix_body_skin():
    """Ensure realistic_body.png has pure radiant skin and seamless cute pleated skirt on hips/thighs."""
    body_path = os.path.join(TEXTURE_DIR, "realistic_body.png")
    if not os.path.exists(body_path):
        w, h = 2048, 2048
        im = Image.new("RGB", (w, h), (250, 222, 208))
    else:
        im = Image.open(body_path).convert("RGB")

    arr = np.array(im)
    np.random.seed(42)

    # 1. Eliminate ALL dark bra/underwear artifact pixels across the entire body texture
    dark_mask = (arr[:, :, 0] < 180) | (arr[:, :, 1] < 155) | (arr[:, :, 2] < 135)
    noise = (np.random.randn(2048, 2048) * 1.5).astype(np.int16)
    arr[dark_mask, 0] = np.clip(250 + noise[dark_mask], 230, 255).astype(np.uint8)
    arr[dark_mask, 1] = np.clip(222 + (noise[dark_mask] * 0.9).astype(np.int16), 200, 245).astype(np.uint8)
    arr[dark_mask, 2] = np.clip(208 + (noise[dark_mask] * 0.8).astype(np.int16), 185, 235).astype(np.uint8)

    # 2. Seamlessly paint the cute pastel lavender pleated mini-skirt & belt across hips/thighs
    # In VRoid body UV, Y 900 to 1320 corresponds to hips, pelvis, and upper thighs
    # Slender high-waisted pastel pink belt (Y: 900 to 950)
    for y in range(900, 950):
        edge_dist = min(y - 900, 950 - y)
        shade = 0.88 + (edge_dist / 25.0) * 0.12
        arr[y, :, 0] = int(242 * shade)
        arr[y, :, 1] = int(178 * shade)
        arr[y, :, 2] = int(202 * shade)

    # Pleated lavender skirt (Y: 950 to 1300)
    pleat_w = 60
    for x_s in range(0, 2048, pleat_w):
        x_m = min(2048, x_s + pleat_w // 2)
        x_e = min(2048, x_s + pleat_w)
        # Highlight half
        for px in range(x_s, x_m):
            frac = (px - x_s) / float(max(1, x_m - x_s))
            hl = math.sin(frac * math.pi) * 18.0
            arr[950:1300, px, 0] = np.clip(218.0 + hl, 180, 255)
            arr[950:1300, px, 1] = np.clip(188.0 + hl * 0.9, 150, 240)
            arr[950:1300, px, 2] = np.clip(232.0 + hl * 1.1, 195, 255)
        # Shadow half
        for px in range(x_m, x_e):
            frac = (px - x_m) / float(max(1, x_e - x_m))
            sh = math.sin(frac * math.pi) * 26.0
            arr[950:1300, px, 0] = np.clip(218.0 - sh, 0, 255)
            arr[950:1300, px, 1] = np.clip(188.0 - sh, 0, 255)
            arr[950:1300, px, 2] = np.clip(232.0 - sh * 0.9, 0, 255)

    fixed_img = Image.fromarray(arr)
    fixed_img.save(body_path, quality=95)
    print(f"Body skin & seamless skirt saved: {body_path}")
    return body_path


def generate_skinny_tops():
    """Generate 1024x2048 skinny, form-fitting cute ribbed sweetheart knit crop top."""
    w, h = 1024, 2048
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(42)

    rib = np.sin(x_g * 2.5) * 8.0 + np.sin(x_g * 0.8) * 4.0
    knit_h = np.cos(y_g * 1.2) * 2.5
    noise = np.random.randn(h, w) * 2.0

    base_r = np.clip(236.0 + rib + knit_h + noise, 195, 255)
    base_g = np.clip(176.0 + rib * 0.85 + knit_h * 0.8 + noise, 135, 220)
    base_b = np.clip(195.0 + rib * 0.9 + knit_h * 0.8 + noise, 150, 235)

    dist_from_center = np.abs(x_g - 512) / 512.0
    slimming = 1.0 - (dist_from_center ** 1.6) * 0.12
    base_r *= slimming
    base_g *= slimming
    base_b *= slimming

    arr = np.stack([base_r, base_g, base_b], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)

    # Sweetheart neckline (Y: 150 to 390, X: 415 to 609)
    for y in range(150, 390):
        dy = (y - 150) / 240.0
        half_w = int(math.sin(dy * math.pi) * 82.0 + (1.0 - dy) * 42.0)
        x_l = 512 - half_w
        x_r = 512 + half_w
        for x in range(x_l, x_r + 1):
            skin_noise = (np.random.rand() - 0.5) * 2.0
            sr = min(255, int(250 + skin_noise))
            sg = min(255, int(220 + skin_noise * 0.9))
            sb = min(255, int(208 + skin_noise * 0.8))
            dist_border = min(x - x_l, x_r - x, 390 - y)
            if dist_border < 8:
                sh = dist_border / 8.0
                sr = int(sr * (0.88 + 0.12 * sh))
                sg = int(sg * (0.88 + 0.12 * sh))
                sb = int(sb * (0.88 + 0.12 * sh))
            img.putpixel((x, y), (sr, sg, sb))

    # Scalloped French lace trim
    for dy_step in range(0, 120):
        frac = dy_step / 120.0
        cur_y = int(150 + frac * 240.0)
        hw = int(math.sin(frac * math.pi) * 82.0 + (1.0 - frac) * 42.0)
        for side in [-1, 1]:
            cur_x = 512 + side * hw
            for sc in range(-4, 5):
                sc_y = cur_y + int(abs(sc) * 0.8)
                sc_x = cur_x + sc
                if 0 <= sc_x < w and 0 <= sc_y < h:
                    img.putpixel((sc_x, sc_y), (255, 252, 248))

    # Pearl buttons
    pearl_main = (255, 255, 255)
    pearl_luster = (255, 220, 235)
    pearl_shadow = (170, 115, 135)
    for by in [420, 500, 580, 660]:
        bx = 512
        r_btn = 12
        draw.ellipse([bx - r_btn - 2, by - r_btn + 2, bx + r_btn + 2, by + r_btn + 4], fill=(150, 100, 120))
        draw.ellipse([bx - r_btn, by - r_btn, bx + r_btn, by + r_btn], fill=pearl_shadow)
        draw.ellipse([bx - r_btn + 2, by - r_btn + 2, bx + r_btn - 2, by + r_btn - 2], fill=pearl_luster)
        draw.ellipse([bx - 4, by - 4, bx - 1, by - 1], fill=pearl_main)
        draw.ellipse([bx - 3, by - 1, bx - 1, by + 1], fill=(120, 80, 95))
        draw.ellipse([bx + 1, by - 1, bx + 3, by + 1], fill=(120, 80, 95))

    # Cute rose ribbon bow
    bow_color = (235, 95, 135)
    bow_dark = (185, 65, 105)
    bx_c, by_c = 512, 385
    draw.ellipse([bx_c - 10, by_c - 8, bx_c + 10, by_c + 8], fill=bow_color, outline=bow_dark)
    draw.polygon([(bx_c - 9, by_c), (bx_c - 32, by_c - 16), (bx_c - 36, by_c + 12)], fill=bow_color, outline=bow_dark)
    draw.polygon([(bx_c + 9, by_c), (bx_c + 32, by_c - 16), (bx_c + 34, by_c + 12)], fill=bow_color, outline=bow_dark)
    draw.line([(bx_c - 6, by_c + 6), (bx_c - 22, by_c + 36)], fill=bow_color, width=4)
    draw.line([(bx_c + 6, by_c + 6), (bx_c + 22, by_c + 36)], fill=bow_color, width=4)

    # Scalloped lace trim on cropped sweater bottom hem
    for x in range(0, w, 16):
        draw.arc([x, 990, x + 16, 1018], start=0, end=180, fill=(255, 252, 248), width=3)

    out_path = os.path.join(TEXTURE_DIR, "realistic_tops.png")
    img.save(out_path, quality=95)
    print(f"Skinny Tops saved: {out_path}")
    return out_path


def generate_skinny_bottoms():
    """Generate 1024x1024 skinny, cute high-waisted pleated skater mini-skirt."""
    w, h = 1024, 1024
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(77)

    noise = np.random.randn(h, w) * 1.8
    base_r = np.clip(218.0 + noise, 180, 245)
    base_g = np.clip(188.0 + noise * 0.9, 150, 230)
    base_b = np.clip(232.0 + noise * 0.95, 195, 255)
    img = Image.fromarray(np.stack([base_r, base_g, base_b], axis=-1).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # 1. Slender High-Waisted Pastel Pink Belt (Y: 10 to 60)
    for y in range(10, 60):
        edge_dist = min(y - 10, 60 - y)
        shade = 0.88 + (edge_dist / 25.0) * 0.12
        for x in range(w):
            img.putpixel((x, y), (int(242 * shade), int(178 * shade), int(202 * shade)))

    for x in range(0, w, 6):
        draw.line([(x, 14), (x + 3, 14)], fill=(255, 252, 250), width=1)
        draw.line([(x, 56), (x + 3, 56)], fill=(255, 252, 250), width=1)

    # Silver Heart Buckle
    def draw_heart(draw_ctx, cx, cy, size, fill_col, outline_col=None):
        points = []
        for t in np.linspace(0, 2 * math.pi, 60):
            hx = 16 * (math.sin(t)**3)
            hy = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t))
            points.append((cx + hx * size, cy + hy * size))
        draw_ctx.polygon(points, fill=fill_col, outline=outline_col)

    draw_heart(draw, 512, 35, 1.25, (235, 238, 248), (185, 190, 205))
    draw_heart(draw, 512, 35, 0.75, (242, 178, 202), (185, 190, 205))
    draw.rectangle([510, 18, 514, 52], fill=(245, 248, 255), outline=(190, 195, 205))
    draw.arc([495, 20, 515, 34], start=180, end=360, fill=(255, 255, 255), width=2)

    # 2. Crisp VERTICAL Knife Pleats in the VISIBLE ZONE (Y: 60 to 360)
    pleat_width = 30
    for x_start in range(0, w, pleat_width):
        x_mid = min(w, x_start + pleat_width // 2)
        x_end = min(w, x_start + pleat_width)

        for px in range(x_start, x_mid):
            frac = (px - x_start) / float(max(1, x_mid - x_start))
            hl = math.sin(frac * math.pi) * 18.0
            for py in range(65, 355):
                c = img.getpixel((px, py))
                img.putpixel((px, py), (
                    min(255, int(c[0] + hl)),
                    min(255, int(c[1] + hl * 0.9)),
                    min(255, int(c[2] + hl * 1.1))
                ))

        for px in range(x_mid, x_end):
            frac = (px - x_mid) / float(max(1, x_end - x_mid))
            sh = math.sin(frac * math.pi) * 26.0
            for py in range(65, 355):
                c = img.getpixel((px, py))
                img.putpixel((px, py), (
                    max(0, int(c[0] - sh)),
                    max(0, int(c[1] - sh)),
                    max(0, int(c[2] - sh * 0.9))
                ))

        draw.line([(x_mid, 65), (x_mid, 355)], fill=(150, 125, 165), width=1)

    # 3. Scalloped White Lace Hem at Skirt Bottom (Y: 345 to 365)
    for x in range(0, w, 16):
        draw.arc([x, 345, x + 16, 365], start=0, end=180, fill=(255, 252, 250), width=3)
        draw.ellipse([x + 6, 352, x + 10, 356], fill=(255, 255, 255))

    out_path = os.path.join(TEXTURE_DIR, "realistic_bottoms.png")
    img.save(out_path, quality=95)
    print(f"Skinny Bottoms saved: {out_path}")
    return out_path


def generate_breathtaking_eyes():
    """Generate 1024x512 stunning anime/doll crystalline iris texture with jewel depth."""
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

        limbal_mask = (dist <= radius)
        norm_dist = np.clip(dist / radius, 0, 1)

        # Starburst fibers
        fibers = (
            np.sin(theta * 32.0) * 16.0 +
            np.sin(theta * 64.0) * 12.0 +
            np.sin(theta * 128.0) * 8.0 +
            np.cos(theta * 16.0 + norm_dist * 8.0) * 14.0
        )
        rings = np.sin(norm_dist * 22.0) * 10.0

        # Gemstone multi-tone coloration:
        # Upper/outer: deep sapphire-indigo (45, 65, 140)
        # Lower/mid: luminous warm golden honey (220, 150, 50)
        vert_fade = np.clip((dy / radius) * 0.5 + 0.5, 0, 1) # 0 at top, 1 at bottom

        top_r = 55.0 + fibers * 0.4
        top_g = 75.0 + fibers * 0.5
        top_b = 150.0 + fibers * 0.8

        bot_r = 225.0 + fibers + rings
        bot_g = 145.0 + fibers * 0.8 + rings * 0.8
        bot_b = 60.0 + fibers * 0.35 + rings * 0.3

        iris_r = top_r * (1.0 - vert_fade) + bot_r * vert_fade
        iris_g = top_g * (1.0 - vert_fade) + bot_g * vert_fade
        iris_b = top_b * (1.0 - vert_fade) + bot_b * vert_fade

        # Velvet limbal ring
        limbal_factor = np.clip((1.0 - norm_dist) / 0.12, 0, 1)
        iris_r *= (0.16 + 0.84 * limbal_factor)
        iris_g *= (0.15 + 0.85 * limbal_factor)
        iris_b *= (0.22 + 0.78 * limbal_factor)

        # Velvet black pupil
        pupil_factor = np.clip((norm_dist - 0.20) / 0.06, 0, 1)
        iris_r = iris_r * pupil_factor + 4.0 * (1.0 - pupil_factor)
        iris_g = iris_g * pupil_factor + 4.0 * (1.0 - pupil_factor)
        iris_b = iris_b * pupil_factor + 6.0 * (1.0 - pupil_factor)

        # Golden starburst collarette halo
        halo_dist = np.abs(norm_dist - 0.32) / 0.10
        halo = np.exp(-0.5 * (halo_dist ** 2))
        iris_r += halo * 45.0
        iris_g += halo * 34.0
        iris_b += halo * 8.0

        arr[limbal_mask, 0] = np.clip(iris_r[limbal_mask], 0, 255)
        arr[limbal_mask, 1] = np.clip(iris_g[limbal_mask], 0, 255)
        arr[limbal_mask, 2] = np.clip(iris_b[limbal_mask], 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # Sparkling diamond catchlights
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

    out_path = os.path.join(TEXTURE_DIR, "realistic_iris.png")
    img.save(out_path, quality=95)
    print(f"Breathtaking Eyes saved: {out_path}")
    return out_path


def generate_radiant_face():
    """Generate 1024x1024 radiant face texture with ZERO lower face shadow, beautiful eyeliner and lashes."""
    w, h = 1024, 1024
    y_g, x_g = np.ogrid[:h, :w]
    np.random.seed(123)

    pore_noise = np.random.randn(h, w) * 1.2
    base_r = np.clip(252.0 + pore_noise, 232, 255)
    base_g = np.clip(224.0 + pore_noise * 0.92, 198, 248)
    base_b = np.clip(212.0 + pore_noise * 0.82, 180, 238)
    arr = np.stack([base_r, base_g, base_b], axis=-1).astype(float)

    # Rosy Cheek Blush
    for cx in [350, 674]:
        dist_sq = ((x_g - cx) / 75.0)**2 + ((y_g - 665) / 45.0)**2
        blush = np.exp(-0.5 * dist_sq)
        arr[:, :, 0] += blush * 28.0
        arr[:, :, 1] += blush * 10.0
        arr[:, :, 2] += blush * 14.0

    # Nose-Tip Blush & Highlight
    nose_dist = ((x_g - 512) / 20.0)**2 + ((y_g - 730) / 14.0)**2
    nose_blush = np.exp(-0.5 * nose_dist)
    arr[:, :, 0] += nose_blush * 22.0
    arr[:, :, 1] += nose_blush * 8.0
    arr[:, :, 2] += nose_blush * 10.0

    nose_hl = np.exp(-0.5 * (((x_g - 512) / 9.0)**2 + ((y_g - 724) / 9.0)**2))
    arr[:, :, 0] += nose_hl * 18.0
    arr[:, :, 1] += nose_hl * 16.0
    arr[:, :, 2] += nose_hl * 14.0

    # Chin & Lower Face Radiant Glow
    chin_dist = ((x_g - 512) / 90.0)**2 + ((y_g - 880) / 60.0)**2
    chin_glow = np.exp(-0.5 * chin_dist)
    arr[:, :, 0] += chin_glow * 14.0
    arr[:, :, 1] += chin_glow * 12.0
    arr[:, :, 2] += chin_glow * 12.0

    # Sweet Glossy Strawberry Lips (Y: 760 to 845, X: 445 to 579)
    for y in range(760, 845):
        for x in range(445, 579):
            dx = (x - 512) / 54.0
            dy = (y - 802) / 30.0
            r_dist = dx**2 + dy**2
            if r_dist < 1.0:
                lip_mask = math.cos(r_dist * (math.pi / 2))
                crease = math.sin(x * 1.8) * 2.5
                lip_r = 232.0 + crease
                lip_g = 118.0 + crease * 0.6
                lip_b = 138.0 + crease * 0.7
                if 814 < y < 832 and 500 < x < 524:
                    gloss = math.sin((y - 814) / 18.0 * math.pi) * math.sin((x - 500) / 24.0 * math.pi)
                    lip_r += gloss * 36.0
                    lip_g += gloss * 34.0
                    lip_b += gloss * 34.0

                arr[y, x, 0] = arr[y, x, 0] * (1.0 - lip_mask) + lip_r * lip_mask
                arr[y, x, 1] = arr[y, x, 1] * (1.0 - lip_mask) + lip_g * lip_mask
                arr[y, x, 2] = arr[y, x, 2] * (1.0 - lip_mask) + lip_b * lip_mask

    for x in range(468, 557):
        dx_val = (x - 512) / 44.0
        y_crease = int(801 - abs(dx_val)**2 * 2.5)
        fade = max(0.0, 1.0 - abs(dx_val))
        if 0 <= y_crease < h:
            arr[y_crease-1:y_crease+2, x, 0] -= fade * 22.0
            arr[y_crease-1:y_crease+2, x, 1] -= fade * 38.0
            arr[y_crease-1:y_crease+2, x, 2] -= fade * 38.0

    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    for ex, sign in [(340, -1), (684, 1)]:
        td_x = ex + sign * 50
        draw.ellipse([td_x - 5, 535, td_x + 5, 545], fill=(255, 248, 242))

        eyelash_col = (42, 26, 22)
        points = []
        for t in np.linspace(0, 1, 30):
            px = ex + sign * (48 - t * 105)
            py = 502 - math.sin(t * math.pi) * 22.0
            points.append((px, py))
        for p1, p2 in zip(points[:-1], points[1:]):
            draw.line([p1, p2], fill=eyelash_col, width=4)

        wing_start = points[-1]
        wing_end = (wing_start[0] - sign * 14, wing_start[1] - 8)
        draw.line([wing_start, wing_end], fill=eyelash_col, width=3)

        for lash_idx in range(5, 26, 4):
            lp = points[lash_idx]
            angle = (lash_idx / 30.0 - 0.5) * 0.8
            lx = lp[0] + math.sin(angle) * 12.0 * sign
            ly = lp[1] - math.cos(angle) * 10.0
            draw.line([lp, (lx, ly)], fill=eyelash_col, width=2)

    out_path = os.path.join(TEXTURE_DIR, "realistic_face.png")
    img.save(out_path, quality=95)
    print(f"Radiant Face saved: {out_path}")
    return out_path


def tighten_and_pack_glb():
    """Tighten clothing mesh vertices (eliminate flared flaps) and pack all textures into avatar.glb."""
    src_glb = "src/robot/api/static/models/avatar.glb.bak"
    out_glb = "src/robot/api/static/models/avatar.glb"

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

        # Torso depth slimming
        vz = 0.01 + (vz - 0.01) * 0.74

        # Taper sleeves inward towards arm bone axis
        if abs(vx) > 0.13:
            sign = 1.0 if vx > 0 else -1.0
            arm_center_x = sign * 0.19
            vx = arm_center_x + (vx - arm_center_x) * 0.62
            vz *= 0.72

        # TUCK IN LOWER HEM & FLAPS (Y <= 1.05)
        if vy <= 1.05:
            h_frac = max(0.0, (1.05 - vy) / 0.22)
            pull_in = 1.0 - (h_frac * 0.26)
            vx *= pull_in
            vz *= pull_in

        struct.pack_into("<3f", bin_data, top_off + i * 12, vx, vy, vz)

    print(f"Tightened {top_count} Tops vertices (eliminated flared flaps)!")

    # 2. TIGHTEN BOTTOMS_01_CLOTH VERTICES (bufferView 7, 470 vertices)
    bv7 = json_data["bufferViews"][7]
    bot_off = bv7.get("byteOffset", 0)
    bot_count = json_data["accessors"][7]["count"]

    for i in range(bot_count):
        vx, vy, vz = struct.unpack_from("<3f", bin_data, bot_off + i * 12)

        vx *= 0.84
        vz *= 0.80

        if vy < 0.85:
            vx *= 0.88
            vz *= 0.88

        struct.pack_into("<3f", bin_data, bot_off + i * 12, vx, vy, vz)

    print(f"Tightened {bot_count} Bottoms vertices for skinny mini-skirt fit!")

    # 3. PACK UPDATED TEXTURES INTO GLB
    bv879_offset = json_data["bufferViews"][879]["byteOffset"]
    new_bin = bytearray(bin_data[:bv879_offset])

    trans_img_path = "/tmp/avatar_textures/transparent.png"
    os.makedirs("/tmp/avatar_textures", exist_ok=True)
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(trans_img_path)

    replace_files = {
        0: os.path.join(TEXTURE_DIR, "realistic_body.png"),
        2: os.path.join(TEXTURE_DIR, "realistic_bottoms.png"),
        3: os.path.join(TEXTURE_DIR, "realistic_shoes.png"),
        4: os.path.join(TEXTURE_DIR, "realistic_tops.png"),
        5: os.path.join(TEXTURE_DIR, "realistic_hairback.png"),
        8: os.path.join(TEXTURE_DIR, "realistic_iris.png"),
        9: os.path.join(TEXTURE_DIR, "realistic_face.png"),
        10: trans_img_path,
        11: os.path.join(TEXTURE_DIR, "realistic_eyewhite.png"),
        14: trans_img_path,
        15: os.path.join(TEXTURE_DIR, "realistic_hair.png"),
    }

    current_offset = bv879_offset
    for i, img_meta in enumerate(json_data["images"]):
        bv_idx = img_meta["bufferView"]
        if i in replace_files and os.path.exists(replace_files[i]):
            with open(replace_files[i], "rb") as rf:
                img_bytes = rf.read()
        else:
            old_bv = json_data["bufferViews"][bv_idx]
            o_off = old_bv["byteOffset"]
            o_len = old_bv["byteLength"]
            img_bytes = bin_data[o_off:o_off + o_len]

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

    print(f"Successfully packed tightened skinny cute avatar into {out_glb} ({out_total_len} bytes)!")


if __name__ == "__main__":
    fix_body_skin()
    generate_skinny_tops()
    generate_skinny_bottoms()
    generate_breathtaking_eyes()
    generate_radiant_face()
    tighten_and_pack_glb()
