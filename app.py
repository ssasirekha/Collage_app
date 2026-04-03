import io
import os
import math
import json
import base64
import hashlib
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageColor
from openai import OpenAI

# --- 1. Setup & State ---
st.set_page_config(page_title="AI Asset Studio Pro", page_icon="🖼️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 1rem; }
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
    display_name: str = "IMAGE"

if "images_meta" not in st.session_state:
    st.session_state["images_meta"] = []

if "images_bytes" not in st.session_state:
    st.session_state["images_bytes"] = {}

if "generated_collage" not in st.session_state:
    st.session_state["generated_collage"] = None


# --- 2. AI Logic ---
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key.strip()) if api_key else None


def classify_image(raw_bytes: bytes):
    client = get_openai_client()
    if not client:
        return {"name": "ASSET"}

    base_img = base64.b64encode(raw_bytes).decode("utf-8")

    prompt = """
You are an image labelling assistant.

Task:
Look at the uploaded image and identify the main object clearly.

Rules:
- Return a short display label of 1 to 3 words only.
- Use title case.
- Do not include extra explanation.
- Do not guess wildly. If unclear, return "Unknown Asset".
- Output only valid JSON in this format:
{"name":"Label Here"}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base_img}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        label = parsed.get("name", "Unknown Asset").strip()
        if not label:
            label = "Unknown Asset"

        return {"name": label}

    except Exception:
        return {"name": "Unknown Asset"}


# --- 3. Rendering Engine ---
def render_collage(items, mode, cols, gap, margin, radius, b_weight, b_color, bg_color, font_size, sizing_option):
    if not items:
        return None

    pil_images = []
    valid_items = []

    for m in items:
        img_bytes = st.session_state["images_bytes"].get(m["id"])
        if img_bytes:
            pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            valid_items.append(m)

    if not pil_images:
        return None

    widths, heights = zip(*(i.size for i in pil_images))

    # Logic for Image Sizing
    if sizing_option == "Enlarge to Largest":
        ref_w, ref_h = max(widths), max(heights)
    elif sizing_option == "Increase to Tallest":
        ref_h = max(heights)
        avg_aspect = sum(w / h for w, h in zip(widths, heights)) / len(valid_items)
        ref_w = int(ref_h * avg_aspect)
    elif sizing_option == "Shrink to Smallest":
        ref_w, ref_h = min(widths), min(heights)
    elif sizing_option == "Match Width":
        ref_w = max(widths)
        ref_h = ref_w
    elif sizing_option == "Match Height":
        ref_h = max(heights)
        ref_w = ref_h
    else:
        ref_w, ref_h = widths[0], heights[0]

    canvas_w = 2000
    count = len(valid_items)
    cols = count if mode == "Horizontal" else (1 if mode == "Vertical" else cols)
    rows = math.ceil(count / cols)

    tile_w = (canvas_w - (2 * margin) - (cols - 1) * gap) // cols
    tile_h = int(tile_w * (ref_h / ref_w))
    canvas_h = (rows * tile_h) + ((rows - 1) * gap) + (2 * margin)

    canvas = Image.new("RGBA", (canvas_w, int(canvas_h)), ImageColor.getrgb(bg_color) + (255,))

    # Font Support
    font_path = "Roboto-Bold.ttf"
    if not os.path.exists(font_path):
        try:
            r = requests.get("https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf", timeout=10)
            with open(font_path, "wb") as f_f:
                f_f.write(r.content)
        except Exception:
            pass

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    for idx, (item, raw_img) in enumerate(zip(valid_items, pil_images)):
        r, c = divmod(idx, cols)
        x = margin + c * (tile_w + gap)
        y = margin + r * (tile_h + gap)

        img = ImageOps.fit(raw_img, (tile_w, tile_h), Image.LANCZOS)

        mask = Image.new("L", (tile_w, tile_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile_w, tile_h), radius=radius, fill=255)

        tile_cv = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 0))
        tile_cv.paste(img, (0, 0), mask)

        draw = ImageDraw.Draw(tile_cv)
        if b_weight > 0:
            draw.rounded_rectangle((0, 0, tile_w, tile_h), radius=radius, outline=b_color, width=b_weight)

        name_txt = item.get("display_name", "ASSET").upper()

        bbox = draw.textbbox((0, 0), name_txt, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x = 30
        pad_y = 15
        box_w = text_w + (2 * pad_x)
        box_h = text_h + (2 * pad_y)

        px = (tile_w - box_w) // 2
        py = tile_h - box_h - 40

        draw.rounded_rectangle(
            [px, py, px + box_w, py + box_h],
            radius=12,
            fill=(15, 23, 42, 220),
            outline="white",
            width=2
        )
        draw.text(
            (tile_w // 2, py + box_h // 2),
            name_txt,
            fill="white",
            font=font,
            anchor="mm"
        )

        canvas.alpha_composite(tile_cv, (int(x), int(y)))

    return canvas.convert("RGB")


# --- 4. Sidebar & UI ---
with st.sidebar:
    st.header("📤 Media Input")
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg", "webp"])

    if uploaded_files:
        new_meta = []
        new_bytes = {}

        for f in uploaded_files:
            raw = f.getvalue()
            file_hash = hashlib.md5(raw).hexdigest()[:10]
            uid = f"img_{file_hash}"

            new_bytes[uid] = raw
            new_meta.append(
                asdict(
                    ImageItem(
                        id=uid,
                        original_file_name=f.name,
                        display_name=os.path.splitext(f.name)[0]
                    )
                )
            )

        old_ids = {m["id"] for m in st.session_state["images_meta"]}
        new_ids = {m["id"] for m in new_meta}

        if old_ids != new_ids:
            st.session_state["images_bytes"] = new_bytes
            st.session_state["images_meta"] = new_meta
            st.session_state["generated_collage"] = None

            keys_to_delete = [
                k for k in list(st.session_state.keys())
                if k.startswith("dn_") or k.startswith("inp_")
            ]
            for k in keys_to_delete:
                del st.session_state[k]

            for m in new_meta:
                st.session_state[f"dn_{m['id']}"] = m["display_name"]


if st.session_state["images_meta"]:
    st.markdown('<p class="main-header">🖼️ AI Image Studio</p>', unsafe_allow_html=True)
    t1, t2 = st.tabs(["📝 AI & Labels", "🎨 Style & Layout"])

    with t1:
        if st.button("✨ RUN AI AUTO-LABEL", use_container_width=True, type="primary"):
            with st.spinner("Analyzing..."):
                for m in st.session_state["images_meta"]:
                    res = classify_image(st.session_state["images_bytes"][m["id"]])
                    label = res.get("name", "Unknown Asset")

                    st.session_state[f"dn_{m['id']}"] = label
                    m["display_name"] = label

            st.rerun()

        for m in st.session_state["images_meta"]:
            col_a, col_b = st.columns([1, 5])
            col_a.image(st.session_state["images_bytes"][m["id"]], width=100)

            current_label = st.session_state.get(f"dn_{m['id']}", m["display_name"])
            new_label = col_b.text_input(
                "Label",
                value=current_label,
                key=f"inp_{m['id']}"
            )

            st.session_state[f"dn_{m['id']}"] = new_label
            m["display_name"] = new_label

    with t2:
        st.subheader("📏 Image Sizing")
        sizing_option = st.radio(
            "Scaling Method:",
            [
                "Keep Original",
                "Enlarge to Largest",
                "Increase to Tallest",
                "Shrink to Smallest",
                "Match Width",
                "Match Height"
            ],
            horizontal=True,
            index=1
        )

        st.divider()

        col1, col2 = st.columns(2)
        mode = col1.selectbox("Layout Mode", ["Grid", "Horizontal", "Vertical"], index=0)
        cols = col2.slider("Columns", 1, 6, 3)

        col3, col4, col5 = st.columns(3)
        gap = col3.slider("Inner Gap (Grid Spacing)", 0, 150, 40)
        margin = col4.slider("Outer Margin", 0, 200, 60)
        radius = col5.slider("Corner Rounding", 0, 100, 30)

        col6, col7, col8 = st.columns(3)
        b_weight = col6.slider("Border Thickness", 0, 20, 5)
        b_color = col7.color_picker("Border Color", "#0000FF")
        bg_color = col8.color_picker("Background Color", "#FFFFFF")

        font_size = st.slider("Label Font Size", 20, 120, 40)

    if st.button("🚀 GENERATE FINAL COLLAGE", use_container_width=True, type="primary"):
        st.session_state["generated_collage"] = render_collage(
            st.session_state["images_meta"],
            mode,
            cols,
            gap,
            margin,
            radius,
            b_weight,
            b_color,
            bg_color,
            font_size,
            sizing_option
        )

    if st.session_state["generated_collage"]:
        st.image(st.session_state["generated_collage"], use_container_width=True)
        buf = io.BytesIO()
        st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button("📥 Download Collage", buf.getvalue(), file_name="collage.png")
else:
    st.info("Please upload images in the sidebar to start.")
