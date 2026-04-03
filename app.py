import io
import os
import math
import json
import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageColor
from openai import OpenAI

# --- 1. Setup & State ---
st.set_page_config(page_title="AI Industrial Studio Pro", page_icon="🏭", layout="wide")

# Custom CSS for a cleaner UI
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .main-header { font-size: 2.5rem; font-weight: 800; color: #1e293b; margin-bottom: 0.5rem; }
    .status-box { padding: 1rem; border-radius: 0.5rem; background-color: #ffffff; border: 1px solid #e2e8f0; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">🏭 Industrial Asset Designer</p>', unsafe_allow_html=True)

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    category: str = "HARDWARE"
    display_name: str = "EQUIPMENT"

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
    
    prompt = """
    Identify this industrial tool. 
    Return JSON: {"category": "SHORT_CATEGORY", "display_name": "Product Name"}
    Keep display_name < 18 chars. Use Uppercase for category.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base_img}", "detail": "high"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception: return None

def sync_edits():
    for item in st.session_state["images_meta"]:
        iid = item['id']
        if f"dn_{iid}" in st.session_state:
            item['display_name'] = st.session_state[f"dn_{iid}"].upper()
        if f"cat_{iid}" in st.session_state:
            item['category'] = st.session_state[f"cat_{iid}"].upper()

# --- 3. High-Definition Rendering Engine ---

def get_shape_mask(size: Tuple[int, int], shape: str, radius: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if shape == "Circle": draw.ellipse((0, 0, size[0], size[1]), fill=255)
    elif shape == "Hexagon":
        w, h = size
        draw.polygon([(w/2,0), (w,h*0.25), (w,h*0.75), (w/2,h), (0,h*0.75), (0,h*0.25)], fill=255)
    else: draw.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, fill=255)
    return mask

def create_pro_tile(img_bytes: bytes, size: Tuple[int, int], shape: str, radius: int, name: str, category: str, border_color: str) -> Image.Image:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    fitted = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    
    # 1. Image with internal 2px white stroke
    mask = get_shape_mask(size, shape, radius)
    content = Image.new("RGBA", size, (0, 0, 0, 0))
    content.paste(fitted.convert("RGBA"), (0, 0), mask)
    draw_internal = ImageDraw.Draw(content)
    draw_internal.rounded_rectangle((0, 0, size[0], size[1]), radius=radius, outline="white", width=2)
    
    # 2. Main Border Frame
    b_width = 6
    canvas_s = (size[0] + b_width*2, size[1] + b_width*2)
    bordered = Image.new("RGBA", canvas_s, (0,0,0,0))
    b_mask = get_shape_mask(canvas_s, shape, radius + b_width)
    
    # Draw the main frame
    frame_draw = ImageDraw.Draw(bordered)
    frame_draw.rounded_rectangle((0, 0, canvas_s[0], canvas_s[1]), radius=radius+b_width, fill=border_color)
    bordered.paste(content, (b_width, b_width), content)
    
    # 3. High-Contrast Label Pill
    draw = ImageDraw.Draw(bordered)
    try: 
        font_main = ImageFont.truetype("arialbd.ttf", max(16, size[0] // 16))
        font_sub = ImageFont.truetype("arial.ttf", max(11, size[0] // 26))
    except: 
        font_main = font_sub = ImageFont.load_default()
    
    txt_n, txt_c = name.upper(), category.upper()
    bn, bc = draw.textbbox((0, 0), txt_n, font=font_main), draw.textbbox((0, 0), txt_c, font=font_sub)
    tw = max(bn[2]-bn[0], bc[2]-bc[0]) + 30
    th = (bn[3]-bn[1]) + (bc[3]-bc[1]) + 15

    bx, by = (bordered.width - tw) // 2, bordered.height - th - 35
    # Dark Glass Pill with White Outline
    draw.rounded_rectangle([bx, by, bx + tw, by + th], radius=8, fill=(15, 23, 42, 230), outline="white", width=1)
    
    draw.text((bx + (tw - (bc[2]-bc[0]))//2, by + 5), txt_c, fill="#94a3b8", font=font_sub)
    draw.text((bx + (tw - (bn[2]-bn[0]))//2, by + 18), txt_n, fill="white", font=font_main)
    
    # 4. Drop Shadow
    shadow_cv = Image.new("RGBA", (canvas_s[0] + 40, canvas_s[1] + 40), (0,0,0,0))
    shadow_cv.paste((0,0,0,70), (12, 14), b_mask)
    shadow_cv = shadow_cv.filter(ImageFilter.GaussianBlur(15))
    shadow_cv.paste(bordered, (10, 10), bordered)
    return shadow_cv

def build_dense_grid(items, width, bg_col, gap, radius, shape, cols, b_color):
    if not items: return None
    sync_edits()
    rows = math.ceil(len(items) / cols)
    margin = 60
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
        
        card = create_pro_tile(st.session_state["images_bytes"][item['id']], (tile_w, tile_h), shape, radius, item['display_name'], item['category'], b_color)
        canvas.alpha_composite(card, (int(x_pos) - 10, int(y_pos) - 10))
    return canvas.convert("RGB")

# --- 4. Streamlit UI Layout ---

with st.sidebar:
    st.header("📤 Media Upload")
    uploaded = st.file_uploader("Drop technical images here", type=["jpg", "png", "webp"], accept_multiple_files=True)
    if uploaded:
        if len(uploaded) != len(st.session_state["images_meta"]):
            new_meta, new_bytes = [], {}
            for idx, f in enumerate(uploaded):
                uid = f"img_{idx}"
                new_bytes[uid] = f.getvalue()
                new_meta.append(asdict(ImageItem(id=uid, original_file_name=f.name)))
            st.session_state["images_bytes"], st.session_state["images_meta"] = new_bytes, new_meta

if st.session_state["images_meta"]:
    tab1, tab2 = st.tabs(["📝 Data & Labeling", "🎨 Canvas Styling"])
    
    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("✨ Auto-Label with AI", use_container_width=True):
                with st.status("Analyzing...") as s:
                    for m in st.session_state["images_meta"]:
                        res = classify_image_with_openai(st.session_state["images_bytes"][m['id']])
                        if res:
                            m['category'], m['display_name'] = res['category'], res['display_name']
                            st.session_state[f"dn_{m['id']}"], st.session_state[f"cat_{m['id']}"] = res['display_name'], res['category']
                    s.update(label="Analysis Done!", state="complete")
                st.rerun()
        
        with c2:
            current_order = [m['id'] for m in st.session_state["images_meta"]]
            new_order = st.multiselect("Drag/Remove to reorder grid sequence", options=current_order, default=current_order)
            if len(new_order) == len(current_order):
                st.session_state["images_meta"] = [next(m for m in st.session_state["images_meta"] if m['id'] == iid) for iid in new_order]

        with st.expander("Edit Labels Manually", expanded=False):
            for m in st.session_state["images_meta"]:
                col_i, col_cat, col_nm = st.columns([1, 2, 2])
                col_i.image(st.session_state["images_bytes"][m['id']], width=80)
                st.text_input(f"Category", value=m['category'], key=f"cat_{m['id']}", on_change=sync_edits)
                st.text_input(f"Display Name", value=m['display_name'], key=f"dn_{m['id']}", on_change=sync_edits)
                st.divider()

    with tab2:
        col_a, col_b, col_c = st.columns(3)
        cols = col_a.select_slider("Columns", options=[1, 2, 3, 4, 5], value=3)
        gap = col_b.slider("Tile Spacing", 20, 150, 60)
        radius = col_c.slider("Rounding", 0, 100, 45)
        
        col_d, col_e, col_f = st.columns(3)
        shape = col_d.selectbox("Tile Shape", ["Rounded Rect", "Circle", "Hexagon"])
        bg_col = col_e.color_picker("Background", "#F1F5F9")
        b_color = col_f.color_picker("Tile Border", "#FFFFFF")

    st.divider()
    if st.button("🚀 GENERATE HIGH-RES COLLAGE", use_container_width=True, type="primary"):
        st.session_state["generated_collage"] = build_dense_grid(
            st.session_state["images_meta"], 2400, bg_col, gap, radius, shape, cols, b_color
        )

    if st.session_state["generated_collage"]:
        st.subheader("🖼️ Final Output")
        st.image(st.session_state["generated_collage"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button("📥 Download 2400px Grid", buf.getvalue(), file_name="technical_grid_pro.png", mime="image/png")
else:
    st.info("Please upload images in the sidebar to begin.")
