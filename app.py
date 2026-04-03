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
st.set_page_config(page_title="AI Asset Studio Pro", page_icon="🏭", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 1rem; }
    /* Highlight AI Button */
    div.stButton > button:first-child[kind="primary"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    display_name: str = "IMAGE"

if "images_meta" not in st.session_state: st.session_state["images_meta"] = []
if "images_bytes" not in st.session_state: st.session_state["images_bytes"] = {}
if "generated_collage" not in st.session_state: st.session_state["generated_collage"] = None

# --- 2. Enhanced AI Logic ---
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key.strip()) if api_key else None

def classify_image(raw_bytes: bytes):
    client = get_openai_client()
    if not client: return None
    base_img = base64.b64encode(raw_bytes).decode("utf-8")
    try:
        # Improved Prompt for better Industrial Labelling
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Identify this industrial asset. Provide a very short, 2-3 word professional name (e.g., 'CNC Milling Machine', '3D Resin Printer'). Return ONLY JSON: {'name': 'Short Name'}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base_img}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

# --- 3. Rendering Engine ---
def render_collage(items, mode, cols, gap, margin, radius, b_weight, b_color, bg_color, font_size, sizing_option):
    if not items: return None
    
    pil_images = [Image.open(io.BytesIO(st.session_state["images_bytes"][i['id']])).convert("RGB") for i in items]
    widths, heights = zip(*(i.size for i in pil_images))
    
    # Sizing Logic
    if sizing_option == "Enlarge to Largest": ref_w, ref_h = max(widths), max(heights)
    elif sizing_option == "Shrink to Smallest": ref_w, ref_h = min(widths), min(heights)
    elif sizing_option == "Match Width": ref_w = max(widths); ref_h = ref_w
    elif sizing_option == "Match Height": ref_h = max(heights); ref_w = ref_h
    else: ref_w, ref_h = widths[0], heights[0]

    canvas_w = 2000
    count = len(items)
    cols = count if mode == "Horizontal" else (1 if mode == "Vertical" else cols)
    rows = math.ceil(count / cols)

    tile_w = (canvas_w - (2 * margin) - (cols - 1) * gap) // cols
    tile_h = int(tile_w * (ref_h / ref_w))
    canvas_h = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin)
    
    canvas = Image.new("RGBA", (canvas_w, int(canvas_h)), ImageColor.getrgb(bg_color) + (255,))
    
    # CLOUD FONT FIX: Looking for a bundled file first
    font = None
    # Add your uploaded font filename to this list
    font_paths = ["boldfont.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except: continue
    if not font: font = ImageFont.load_default()

    for idx, (item, raw_img) in enumerate(zip(items, pil_images)):
        r, c = divmod(idx, cols)
        # Center alignment for odd-numbered last rows
        current_row_count = min(count - (r * cols), cols)
        row_offset = (canvas_w - (current_row_count * tile_w + (current_row_count - 1) * gap)) // 2
        
        x = row_offset + c * (tile_w + gap)
        y = margin + r * (tile_h + gap)

        img = ImageOps.fit(raw_img, (tile_w, tile_h), Image.LANCZOS)
        mask = Image.new("L", (tile_w, tile_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,tile_w,tile_h), radius=radius, fill=255)
        
        tile_cv = Image.new("RGBA", (tile_w, tile_h), (0,0,0,0))
        tile_cv.paste(img, (0,0), mask)
        
        draw = ImageDraw.Draw(tile_cv)
        if b_weight > 0:
            draw.rounded_rectangle((0,0,tile_w,tile_h), radius=radius, outline=b_color, width=b_weight)
        
        # Label with dynamic background
        name_txt = st.session_state.get(f"dn_{item['id']}", item['display_name']).upper()
        bbox = draw.textbbox((0,0), name_txt, font=font)
        tw, th = bbox[2]-bbox[0]+40, bbox[3]-bbox[1]+20
        px, py = (tile_w-tw)//2, tile_h-th-30
        
        draw.rounded_rectangle([px, py, px+tw, py+th], radius=10, fill=(0,0,0,200), outline="white", width=2)
        draw.text((px+20, py+10), name_txt, fill="white", font=font)
        canvas.alpha_composite(tile_cv, (int(x), int(y)))
    
    return canvas.convert("RGB")

# --- 4. Main App ---
with st.sidebar:
    st.header("📤 Upload Section")
    files = st.file_uploader("Upload Images", accept_multiple_files=True)
    if files:
        if len(files) != len(st.session_state["images_meta"]):
            st.session_state["images_bytes"] = {f"img_{i}": f.getvalue() for i, f in enumerate(files)}
            st.session_state["images_meta"] = [asdict(ImageItem(id=f"img_{i}", original_file_name=f.name)) for i in range(len(files))]

if st.session_state["images_meta"]:
    st.markdown('<p class="main-header">🖼️ AI Asset Studio Pro</p>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📝 AI Labeling", "🎨 Style & Scale"])
    
    with t1:
        if st.button("✨ ✨ RUN AI AUTO-LABEL ✨ ✨", use_container_width=True, type="primary"):
            with st.spinner("AI analyzing hardware..."):
                for m in st.session_state["images_meta"]:
                    res = classify_image(st.session_state["images_bytes"][m['id']])
                    if res: st.session_state[f"dn_{m['id']}"] = res['name']
            st.rerun()
        
        for m in st.session_state["images_meta"]:
            col_a, col_b = st.columns([1, 4])
            col_a.image(st.session_state["images_bytes"][m['id']], width=100)
            st.session_state[f"dn_{m['id']}"] = col_b.text_input(f"Edit Label for {m['id']}", value=st.session_state.get(f"dn_{m['id']}", "IMAGE"))

    with t2:
        sizing_option = st.radio("Auto-Scale Mode", ["Enlarge to Largest", "Shrink to Smallest", "Match Width", "Match Height"], horizontal=True)
        col1, col2, col3 = st.columns(3)
        mode = col1.selectbox("Layout", ["Grid", "Horizontal", "Vertical"])
        cols = col2.slider("Cols", 1, 6, 3) if mode == "Grid" else 1
        font_size = col3.slider("Font Size", 20, 120, 45)
        
        col4, col5, col6 = st.columns(3)
        gap = col4.slider("Gap", 0, 100, 40)
        radius = col5.slider("Corner", 0, 100, 25)
        bg = col6.color_picker("Background", "#111827")

    if st.button("🚀 GENERATE COLLAGE", use_container_width=True, type="primary"):
        st.session_state["generated_collage"] = render_collage(st.session_state["images_meta"], mode, cols, gap, 50, radius, 4, "#ffffff", bg, font_size, sizing_option)

    if st.session_state["generated_collage"]:
        st.image(st.session_state["generated_collage"], use_container_width=True)
        buf = io.BytesIO(); st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button("📥 Download PNG", buf.getvalue(), "collage.png", "image/png")
