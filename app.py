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
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f1f5f9; border-radius: 4px 4px 0 0; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { background-color: #e2e8f0; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    category: str = "INDUSTRIAL"
    display_name: str = "ASSET"

if "images_meta" not in st.session_state: st.session_state["images_meta"] = []
if "images_bytes" not in st.session_state: st.session_state["images_bytes"] = {}
if "generated_collage" not in st.session_state: st.session_state["generated_collage"] = None

# --- 2. Logic ---
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
                {"type": "text", "text": "Identify this industrial machine. Return JSON: {'category':'uppercase_type','name':'Title Case Name'}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base_img}"}}
            ]}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except: return None

def sync_data():
    for m in st.session_state["images_meta"]:
        m['display_name'] = st.session_state.get(f"dn_{m['id']}", m['display_name'])
        m['category'] = st.session_state.get(f"cat_{m['id']}", m['category'])

# --- 3. Rendering Engine ---
def render_collage(items, mode, cols, gap, margin, radius, b_weight, b_color, bg_color, font_size):
    if not items: return None
    sync_data()
    
    canvas_w = 2000
    count = len(items)
    
    if mode == "Horizontal": cols, rows = count, 1
    elif mode == "Vertical": cols, rows = 1, count
    else: rows = math.ceil(count / cols)

    tile_w = (canvas_w - (2 * margin) - (cols - 1) * gap) // cols
    tile_h = tile_w
    canvas_h = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin)
    
    canvas = Image.new("RGBA", (canvas_w, int(canvas_h)), ImageColor.getrgb(bg_color) + (255,))
    
    for idx, item in enumerate(items):
        r, c = divmod(idx, cols)
        rem = count - (r * cols)
        row_cols = min(rem, cols)
        row_w = (row_cols * tile_w) + ((row_cols - 1) * gap)
        x = ((canvas_w - row_w) // 2) + c * (tile_w + gap)
        y = margin + r * (tile_h + gap)

        raw = Image.open(io.BytesIO(st.session_state["images_bytes"][item['id']])).convert("RGB")
        img = ImageOps.fit(raw, (tile_w, tile_h), Image.LANCZOS)
        
        mask = Image.new("L", (tile_w, tile_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,tile_w,tile_h), radius=radius, fill=255)
        
        tile_cv = Image.new("RGBA", (tile_w, tile_h), (0,0,0,0))
        tile_cv.paste(img, (0,0), mask)
        
        # Dual Border
        draw = ImageDraw.Draw(tile_cv)
        if b_weight > 0:
            draw.rounded_rectangle((0,0,tile_w,tile_h), radius=radius, outline=b_color, width=b_weight)
        
        # High-Contrast Labels
        try: font = ImageFont.truetype("arialbd.ttf", font_size)
        except: font = ImageFont.load_default()
        
        name_txt, cat_txt = item['display_name'].upper(), item['category'].upper()
        
        # Label Pill
        tw = draw.textbbox((0,0), name_txt, font=font)[2] + 20
        th = font_size + 15
        px, py = (tile_w - tw)//2, tile_h - th - 20
        
        draw.rounded_rectangle([px, py, px+tw, py+th], radius=5, fill=(15, 23, 42, 220), outline="white", width=1)
        draw.text((px + 10, py + 5), name_txt, fill="white", font=font)

        canvas.alpha_composite(tile_cv, (int(x), int(y)))
    
    return canvas.convert("RGB")

# --- 4. Sidebar & UI ---
with st.sidebar:
    st.header("📤 Input Assets")
    uploaded_files = st.file_uploader("Upload Hardware Images", accept_multiple_files=True)
    
    # FIXED: Proper initialization of meta to avoid NameError 'f'
    if uploaded_files:
        if len(uploaded_files) != len(st.session_state["images_meta"]):
            new_meta = []
            new_bytes = {}
            for i, uploaded_file in enumerate(uploaded_files):
                uid = f"img_{i}"
                new_bytes[uid] = uploaded_file.getvalue()
                new_meta.append(asdict(ImageItem(id=uid, original_file_name=uploaded_file.name)))
            st.session_state["images_bytes"] = new_bytes
            st.session_state["images_meta"] = new_meta

if st.session_state["images_meta"]:
    st.markdown('<p class="main-header">🏭 Industrial Asset Combiner</p>', unsafe_allow_html=True)
    
    t1, t2 = st.tabs(["📝 Labeling & AI", "🎨 Layout & Styling"])
    
    with t1:
        if st.button("✨ Run AI Auto-Label", use_container_width=True):
            with st.spinner("Analyzing machines..."):
                for m in st.session_state["images_meta"]:
                    res = classify_image(st.session_state["images_bytes"][m['id']])
                    if res:
                        m['display_name'], m['category'] = res['name'], res['category']
                        st.session_state[f"dn_{m['id']}"] = res['name']
            st.rerun()
            
        for m in st.session_state["images_meta"]:
            c_i, c_e = st.columns([1, 5])
            c_i.image(st.session_state["images_bytes"][m['id']], width=100)
            c_e.text_input(f"Label Name ({m['id']})", value=m['display_name'], key=f"dn_{m['id']}")
            st.divider()

    with t2:
        col1, col2 = st.columns(2)
        mode = col1.selectbox("Combine Mode", ["Grid", "Horizontal", "Vertical"])
        cols = col2.slider("Grid Columns", 1, 6, 3) if mode == "Grid" else 1
        
        col3, col4, col5 = st.columns(3)
        gap = col3.slider("Spacing", 0, 150, 40)
        margin = col4.slider("Outer Margin", 0, 200, 60)
        radius = col5.slider("Rounding", 0, 100, 30)
        
        col6, col7, col8 = st.columns(3)
        b_weight = col6.slider("Border", 0, 20, 5)
        b_color = col7.color_picker("Border Color", "#FFFFFF")
        bg_color = col8.color_picker("Background", "#0F172A")
        
        font_size = st.slider("Label Size", 10, 60, 24)

    if st.button("🚀 GENERATE COLLAGE", use_container_width=True, type="primary"):
        st.session_state["generated_collage"] = render_collage(
            st.session_state["images_meta"], mode, cols, gap, margin, radius, b_weight, b_color, bg_color, font_size
        )

    if st.session_state["generated_collage"]:
        st.image(st.session_state["generated_collage"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button("📥 Download Asset Grid", buf.getvalue(), file_name="asset_collage.png", mime="image/png")
else:
    st.info("Upload industrial images in the sidebar to start.")
