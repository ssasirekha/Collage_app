import io
import os
import math
import json
import base64
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageColor
from openai import OpenAI

# --- 1. Setup & State ---
st.set_page_config(page_title="AI Asset Studio Pro", page_icon="🖼️", layout="wide")

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    display_name: str = "IMAGE"

if "images_meta" not in st.session_state: st.session_state["images_meta"] = []
if "images_bytes" not in st.session_state: st.session_state["images_bytes"] = {}
if "generated_collage" not in st.session_state: st.session_state["generated_collage"] = None

# --- 2. MODIFIED: Image Type & Specific Naming Logic ---
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key: return None
    return OpenAI(api_key=api_key.strip())

def classify_image(raw_bytes: bytes):
    client = get_openai_client()
    if not client: return None
    base_img = base64.b64encode(raw_bytes).decode("utf-8")
    try:
        # Strict prompt to identify TYPE and specific NAME
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a technical asset auditor. Your goal is to categorize the image type and provide a specific functional name."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text", 
                            "text": """Analyze this image. 
                            1. Identify Image Type: (e.g., Software UI, Mechanical Part, Architectural Site).
                            2. Provide a Specific Name: Based on visual evidence (text on screen, component shape).
                            Return ONLY a JSON object: {"name": "Specific Functional Name"} 
                            Example: 'SCADA Fleet Dashboard' or 'Vertical Centrifugal Pump'."""
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base_img}", "detail": "high"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"name": "ANALYSIS ERROR"}

# --- 3. Rendering Engine ---
def render_collage(items, mode, cols, gap, margin, radius, b_weight, b_color, bg_color, font_size, sizing_option):
    if not items: return None
    
    pil_images = [Image.open(io.BytesIO(st.session_state["images_bytes"][m['id']])).convert("RGB") for m in items]
    widths, heights = zip(*(i.size for i in pil_images))
    
    # Defaults: Enlarge to Largest
    if sizing_option == "Enlarge to Largest": ref_w, ref_h = max(widths), max(heights)
    elif sizing_option == "Increase to Tallest":
        ref_h = max(heights)
        avg_aspect = sum(w/h for w, h in zip(widths, heights)) / len(items)
        ref_w = int(ref_h * avg_aspect)
    else: ref_w, ref_h = widths[0], heights[0]

    canvas_w = 2000
    count = len(items)
    cols = count if mode == "Horizontal" else (1 if mode == "Vertical" else cols)
    rows = math.ceil(count / cols)

    tile_w = (canvas_w - (2 * margin) - (cols - 1) * gap) // cols
    tile_h = int(tile_w * (ref_h / ref_w))
    canvas_h = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin)
    
    canvas = Image.new("RGBA", (canvas_w, int(canvas_h)), ImageColor.getrgb(bg_color) + (255,))
    
    font_path = "Roboto-Bold.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get("https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf")
            with open(font_path, "wb") as f_f: f_f.write(r.content)
        except: pass
    try: font = ImageFont.truetype(font_path, font_size)
    except: font = ImageFont.load_default()

    for idx, (item, raw_img) in enumerate(zip(items, pil_images)):
        r, c = divmod(idx, cols)
        x = margin + c * (tile_w + gap)
        y = margin + r * (tile_h + gap)

        img = ImageOps.fit(raw_img, (tile_w, tile_h), Image.LANCZOS)
        mask = Image.new("L", (tile_w, tile_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0,0,tile_w,tile_h), radius=radius, fill=255)
        
        tile_cv = Image.new("RGBA", (tile_w, tile_h), (0,0,0,0))
        tile_cv.paste(img, (0,0), mask)
        
        draw = ImageDraw.Draw(tile_cv)
        if b_weight > 0:
            draw.rounded_rectangle((0,0,tile_w,tile_h), radius=radius, outline=b_color, width=b_weight)
        
        # Labeling
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
            for i, f in enumerate(uploaded_files):
                uid = f"img_{i}"
                new_bytes[uid] = f.getvalue()
                new_meta.append(asdict(ImageItem(id=uid, original_file_name=f.name)))
            st.session_state["images_bytes"], st.session_state["images_meta"] = new_bytes, new_meta

if st.session_state["images_meta"]:
    st.title("🖼️ Smart Asset Labeller")
    t1, t2 = st.tabs(["📝 Image Analysis", "🎨 Layout & Style"])
    
    with t1:
        if st.button("✨ RUN SMART ANALYSIS", use_container_width=True, type="primary"):
            with st.spinner("Determining image types and names..."):
                for m in st.session_state["images_meta"]:
                    res = classify_image(st.session_state["images_bytes"][m['id']])
                    if res: st.session_state[f"dn_{m['id']}"] = res['name']
            st.rerun()
            
        for m in st.session_state["images_meta"]:
            col_a, col_b = st.columns([1, 5])
            col_a.image(st.session_state["images_bytes"][m['id']], width=100)
            st.session_state[f"dn_{m['id']}"] = col_b.text_input(f"Resulting Label", value=st.session_state.get(f"dn_{m['id']}", "ANALYZING..."), key=f"inp_{m['id']}")

    with t2:
        st.subheader("📏 Grid Configuration")
        sizing_option = st.radio("Sizing:", ["Keep Original", "Enlarge to Largest", "Increase to Tallest"], index=1)
        
        col1, col2, col3 = st.columns(3)
        mode = col1.selectbox("Layout", ["Grid", "Horizontal", "Vertical"])
        cols = col2.slider("Columns", 1, 6, 3)
        gap = col3.slider("Inner Spacing", 0, 100, 40)
        
        col4, col5, col6 = st.columns(3)
        b_weight = col4.slider("Border", 0, 20, 5)
        b_color = col5.color_picker("Border Color", "#0000FF") # Default: Blue
        bg_color = col6.color_picker("Background", "#FFFFFF") # Default: White

    if st.button("🚀 GENERATE COLLAGE", use_container_width=True, type="primary"):
        st.session_state["generated_collage"] = render_collage(st.session_state["images_meta"], mode, cols, gap, 60, 30, b_weight, b_color, bg_color, 40, sizing_option)

    if st.session_state["generated_collage"]:
        st.image(st.session_state["generated_collage"], use_container_width=True)
        buf = io.BytesIO(); st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button("📥 Save Image", buf.getvalue(), file_name="asset_grid.png")
