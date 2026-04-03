import io
import os
import math
import json
import base64
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageColor
from openai import OpenAI

# --- 1. Setup & State ---
st.set_page_config(page_title="AI Asset Studio Pro", page_icon="🖼️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f1f5f9; border-radius: 4px 4px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #e2e8f0; font-weight: bold; }
    
    div.stButton > button:first-child[kind="primary"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    category: str = "GENERAL"
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Identify this industrial hardware. Provide a 2-3 word professional name. Return ONLY JSON: {'name':'Title Case Name'}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base_img}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except: return None

# --- 3. Rendering Engine ---
def render_collage(items, mode, cols, gap, margin, radius, b_weight, b_color, bg_color, font_size, sizing_option):
    if not items: return None
    
    pil_images = [Image.open(io.BytesIO(st.session_state["images_bytes"][m['id']])).convert("RGB") for m in items]
    widths, heights = zip(*(i.size for i in pil_images))
    
    # Image Sizing Logic
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
    
    # Cloud Font Fix
    font_path = "Roboto-Bold.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get("https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf")
            with open(font_path, "wb") as f_file: f_file.write(r.content)
        except: pass

    try: font = ImageFont.truetype(font_path, font_size)
    except: font = ImageFont.load_default()

    for idx, (item, raw_img) in enumerate(zip(items, pil_images)):
        r, c = divmod(idx, cols)
        rem = count - (r * cols)
        row_cols = min(rem, cols)
        row_w = (row_cols * tile_w) + ((row_cols - 1) * gap)
        
        x = ((canvas_w - row_w) // 2) + c * (tile_w + gap)
        y = margin + r * (tile_h + gap)

        # Auto-Scale Logic
        img = ImageOps.fit(raw_img, (tile_w, tile_h), Image.LANCZOS)
        mask = Image.new("L", (tile_w, tile_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,tile_w,tile_h), radius=radius, fill=255)
        
        tile_cv = Image.new("RGBA", (tile_w, tile_h), (0,0,0,0))
        tile_cv.paste(img, (0,0), mask)
        
        draw = ImageDraw.Draw(tile_cv)
        if b_weight > 0:
            draw.rounded_rectangle((0,0,tile_w,tile_h), radius=radius, outline=b_color, width=b_weight)
        
        # Label Rendering
        name_txt = st.session_state.get(f"dn_{item['id']}", item['display_name']).upper()
        bbox = draw.textbbox((0,0), name_txt, font=font)
        tw, th = bbox[2]-bbox[0]+60, bbox[3]-bbox[1]+30
        px, py = (tile_w-tw)//2, tile_h-th-40
        
        draw.rounded_rectangle([px, py, px+tw, py+th], radius=12, fill=(15, 23, 42, 220), outline="white", width=2)
        draw.text((tile_w//2, py + (th//2)), name_txt, fill="white", font=font, anchor="mm")

        canvas.alpha_composite(tile_cv, (int(x), int(y)))
    
    return canvas.convert("RGB")

# --- 4. Sidebar & UI ---
with st.sidebar:
    st.header("📤 Media Input")
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) != len(st.session_state["images_meta"]):
            new_meta, new_bytes = [], {}
            # FIX: Explicit loop to define 'f' and prevent NameError
            for i, f in enumerate(uploaded_files):
                uid = f"img_{i}"
                new_bytes[uid] = f.getvalue()
                new_meta.append(asdict(ImageItem(id=uid, original_file_name=f.name)))
            st.session_state["images_bytes"], st.session_state["images_meta"] = new_bytes, new_meta

if st.session_state["images_meta"]:
    st.markdown('<p class="main-header">🖼️ AI Image Studio</p>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📝 Content & AI", "🎨 Layout & Style"])
    
    with t1:
        if st.button("✨ Run AI Auto-Label", use_container_width=True, type="primary"):
            with st.spinner("AI Identifying..."):
                for m in st.session_state["images_meta"]:
                    res = classify_image(st.session_state["images_bytes"][m['id']])
                    if res: st.session_state[f"dn_{m['id']}"] = res['name']
            st.rerun()
            
        for m in st.session_state["images_meta"]:
            c_i, c_e = st.columns([1, 5])
            c_i.image(st.session_state["images_bytes"][m['id']], width=100)
            st.session_state[f"dn_{m['id']}"] = c_e.text_input(f"Label", value=st.session_state.get(f"dn_{m['id']}", "IMAGE"), key=f"inp_{m['id']}")

    with t2:
        st.subheader("📏 Image Sizing (Auto-Scale)")
        sizing_option = st.radio("Normalize Dimensions", ["Keep Original", "Enlarge to Largest", "Shrink to Smallest", "Match Width", "Match Height"], horizontal=True, index=1)
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        mode = col1.selectbox("Layout", ["Grid", "Horizontal", "Vertical"])
        cols = col2.slider("Cols", 1, 6, 3) if mode == "Grid" else 1
        font_size = col3.slider("Font Size", 20, 100, 45)
        
        col4, col5, col6 = st.columns(3)
        gap = col4.slider("Gap", 0, 150, 40)
        radius = col5.slider("Corner", 0, 100, 30)
        bg = col6.color_picker("Background", "#0F172A")

    if st.button("🚀 GENERATE FINAL IMAGE", use_container_width=True, type="primary"):
        st.session_state["generated_collage"] = render_collage(st.session_state["images_meta"], mode, cols, gap, 60, radius, 4, "#FFFFFF", bg, font_size, sizing_option)

    if st.session_state["generated_collage"]:
        st.image(st.session_state["generated_collage"], use_container_width=True)
        buf = io.BytesIO(); st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button("📥 Download", buf.getvalue(), file_name="output.png")
else:
    st.info("Upload images to start.")
