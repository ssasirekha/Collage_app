import io
import os
import math
import json
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageColor
from openai import OpenAI

# --- 1. Setup & State ---
st.set_page_config(page_title="AI Industrial Studio Pro", page_icon="🏭", layout="wide")
st.title("🏭 AI Industrial Designer: Pro Grid & Reorder")
st.caption("Precision labeling, categorization, and custom positioning for technical equipment.")

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    category: str = "Hardware"
    display_name: str = "Equipment"

def ensure_state():
    if "images_meta" not in st.session_state:
        st.session_state["images_meta"] = []
    if "images_bytes" not in st.session_state:
        st.session_state["images_bytes"] = {}
    if "generated_collage" not in st.session_state:
        st.session_state["generated_collage"] = None

ensure_state()

# --- 2. AI & Sync Logic ---

def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key.strip()) if api_key else None

def classify_image_with_openai(raw_bytes: bytes):
    client = get_openai_client()
    if not client: return None
    base_img = base64.b64encode(raw_bytes).decode("utf-8")
    
    prompt_instruction = """
    Analyze this industrial equipment image. 
    1. Identify the specific type (e.g., 'SLA Resin Printer', 'CNC Mill').
    2. Assign a broad Category (e.g., '3D Printing', 'Machining').
    3. Keep display_name under 20 characters.
    Return ONLY a JSON object: {"category": "...", "display_name": "..."}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt_instruction},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base_img}", "detail": "high"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception: return None

def sync_edits():
    """Explicitly pulls data from widget keys back into the main metadata list."""
    for item in st.session_state["images_meta"]:
        iid = item['id']
        # Fetch from the dynamic keys generated in the UI
        if f"dn_{iid}" in st.session_state:
            item['display_name'] = st.session_state[f"dn_{iid}"]
        if f"cat_{iid}" in st.session_state:
            item['category'] = st.session_state[f"cat_{iid}"]

# --- 3. Rendering Engine ---

def get_shape_mask(size: Tuple[int, int], shape: str, radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if shape == "Circle": draw.ellipse((0, 0, size[0], size[1]), fill=255)
    elif shape == "Hexagon":
        w, h = size
        draw.polygon([(w/2,0), (w,h*0.25), (w,h*0.75), (w/2,h), (0,h*0.75), (0,h*0.25)], fill=255)
    else: draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask

def create_pro_tile(img_bytes: bytes, size: Tuple[int, int], shape: str, radius: int, name: str, category: str) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    fitted = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    
    mask = get_shape_mask(size, shape, radius)
    content = Image.new("RGBA", size, (0, 0, 0, 0))
    content.paste(fitted.convert("RGBA"), (0, 0), mask)
    
    border_s = 4
    canvas_s = (size[0] + border_s*2, size[1] + border_s*2)
    bordered = Image.new("RGBA", canvas_s, (255, 255, 255, 255))
    border_mask = get_shape_mask(canvas_s, shape, radius + border_s)
    bordered.putalpha(border_mask)
    bordered.paste(content, (border_s, border_s), content)
    
    draw = ImageDraw.Draw(bordered)
    try: 
        font_main = ImageFont.truetype("arial.ttf", max(14, size[0] // 18))
        font_sub = ImageFont.truetype("arial.ttf", max(10, size[0] // 28))
    except: 
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()
    
    txt_n, txt_c = name.upper(), category.upper()
    box_n = draw.textbbox((0, 0), txt_n, font=font_main)
    box_c = draw.textbbox((0, 0), txt_c, font=font_sub)
    tw = max(box_n[2]-box_n[0], box_c[2]-box_c[0])
    th = (box_n[3]-box_n[1]) + (box_c[3]-box_c[1]) + 8

    bx, by = (bordered.width - tw) // 2, bordered.height - th - 25
    draw.rounded_rectangle([bx - 12, by - 6, bx + tw + 12, by + th + 6], radius=6, fill=(40, 44, 52, 220))
    draw.text((bx + (tw - (box_c[2]-box_c[0]))//2, by), txt_c, fill=(180, 180, 180), font=font_sub)
    draw.text((bx + (tw - (box_n[2]-box_n[0]))//2, by + (box_c[3]-box_c[1]) + 4), txt_n, fill=(255, 255, 255), font=font_main)
    
    shadow_cv = Image.new("RGBA", (canvas_s[0] + 30, canvas_s[1] + 30), (0,0,0,0))
    shadow_cv.paste((0,0,0,60), (8, 10), border_mask)
    shadow_cv = shadow_cv.filter(ImageFilter.GaussianBlur(12))
    shadow_cv.paste(bordered, (10, 10), bordered)
    return shadow_cv

def build_dense_grid(items, width, bg_col, gap, radius, shape, cols):
    if not items: return None
    sync_edits() # CRITICAL: Update labels before drawing
    
    rows = math.ceil(len(items) / cols)
    margin = 50
    tile_w = (width - (2 * margin) - (cols - 1) * gap) // cols
    tile_h = tile_w
    height = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin)
    canvas = Image.new("RGBA", (width, int(height)), ImageColor.getrgb(bg_col) + (255,))
    
    for idx, item in enumerate(items):
        r, c = divmod(idx, cols)
        rem = len(items) - (r * cols)
        row_w = (min(rem, cols) * tile_w) + ((min(rem, cols) - 1) * gap)
        x_pos = ((width - row_w) // 2) + c * (tile_w + gap)
        y_pos = margin + r * (tile_h + gap)
        
        card = create_pro_tile(st.session_state["images_bytes"][item['id']], (tile_w, tile_h), shape, radius, item['display_name'], item['category'])
        canvas.alpha_composite(card, (int(x_pos) - 10, int(y_pos) - 10))
    return canvas.convert("RGB")

# --- 4. Main App UI ---

uploaded = st.file_uploader("Upload technical images", type=["jpg", "png", "webp"], accept_multiple_files=True)

if uploaded:
    if len(uploaded) != len(st.session_state["images_meta"]):
        new_meta, new_bytes = [], {}
        for idx, f in enumerate(uploaded):
            uid = f"img_{idx}"
            new_bytes[uid] = f.getvalue()
            new_meta.append(asdict(ImageItem(id=uid, original_file_name=f.name)))
        st.session_state["images_bytes"], st.session_state["images_meta"] = new_bytes, new_meta

if st.session_state["images_meta"]:
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("🛠️ Label & Order Studio")
        
        if st.button("✨ Run AI Auto-Label", use_container_width=True):
            with st.spinner("AI analyzing hardware..."):
                for m in st.session_state["images_meta"]:
                    res = classify_image_with_openai(st.session_state["images_bytes"][m['id']])
                    if res:
                        m['category'], m['display_name'] = res['category'], res['display_name']
                        # Sync to session state keys to ensure widgets update
                        st.session_state[f"dn_{m['id']}"] = res['display_name']
                        st.session_state[f"cat_{m['id']}"] = res['category']
            st.rerun()

        # POSITION REORDERING
        st.write("**Arrangement Order**")
        current_meta = st.session_state["images_meta"]
        order_options = [m['id'] for m in current_meta]
        
        # Mapping for display in selectbox
        def get_name(iid):
            match = next((x for x in current_meta if x['id'] == iid), None)
            return f"{match['display_name']} [{iid}]" if match else iid

        new_order = st.multiselect(
            "Select items in the desired order:",
            options=order_options,
            default=order_options,
            format_func=get_name
        )

        if len(new_order) == len(order_options):
            st.session_state["images_meta"] = [next(m for m in current_meta if m['id'] == iid) for iid in new_order]

        # MANUAL OVERRIDES (Fixing the persistent update issue)
        with st.expander("📝 Manual Label Overrides", expanded=True):
            for m in st.session_state["images_meta"]:
                row_c1, row_c2 = st.columns([1, 3])
                row_c1.image(st.session_state["images_bytes"][m['id']], width=60)
                # We use the key as the source of truth
                st.text_input(f"Category", value=m['category'], key=f"cat_{m['id']}", on_change=sync_edits)
                st.text_input(f"Name", value=m['display_name'], key=f"dn_{m['id']}", on_change=sync_edits)
                st.divider()

        st.subheader("🎨 Canvas Controls")
        cols = st.select_slider("Columns", options=[1, 2, 3, 4, 5], value=3)
        gap = st.slider("Grid Gap", 0, 100, 30)
        radius = st.slider("Corner Radius", 0, 100, 45)
        shape = st.selectbox("Shape", ["Rounded Rect", "Circle", "Hexagon"])
        bg_col = st.color_picker("Background", "#FFFFFF")

        if st.button("🚀 Render Canvas", use_container_width=True):
            st.session_state["generated_collage"] = build_dense_grid(
                st.session_state["images_meta"], 2000, bg_col, gap, radius, shape, cols
            )

    with c2:
        st.subheader("🖼️ Result")
        if st.session_state["generated_collage"]:
            st.image(st.session_state["generated_collage"], use_container_width=True)
            
            # Download Logic
            buf = io.BytesIO()
            st.session_state["generated_collage"].save(buf, format="PNG")
            st.download_button("📥 Download Final Grid", buf.getvalue(), file_name="equipment_grid.png", mime="image/png")
        else:
            st.info("Upload images and click 'Render' to generate the grid.")
