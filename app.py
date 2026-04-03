import io
import os
import math
import json
import base64
import hashlib
import requests
from dataclasses import dataclass, asdict

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageColor
from openai import OpenAI

# -----------------------------
# 1. Setup
# -----------------------------
st.set_page_config(page_title="AI Asset Studio Pro", page_icon="🖼️", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1e293b; margin-bottom: 1rem; }
    div.stButton > button:first-child[kind="primary"] {
        background-color: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


@dataclass
class ImageItem:
    id: str
    original_file_name: str
    display_name: str = "Processing..."
    category: str = "Uncategorized"


if "images_meta" not in st.session_state:
    st.session_state["images_meta"] = []

if "images_bytes" not in st.session_state:
    st.session_state["images_bytes"] = {}

if "generated_collage" not in st.session_state:
    st.session_state["generated_collage"] = None

if "smart_collage_generated" not in st.session_state:
    st.session_state["smart_collage_generated"] = False


# -----------------------------
# 2. OpenAI client
# -----------------------------
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key.strip()) if api_key else None


# -----------------------------
# 3. Fonts
# -----------------------------
def load_font(font_size=40, bold=True):
    font_path = "Roboto-Bold.ttf" if bold else "Roboto-Regular.ttf"
    font_url = (
        "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Bold.ttf"
        if bold else
        "https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf"
    )

    if not os.path.exists(font_path):
        try:
            r = requests.get(font_url, timeout=10)
            with open(font_path, "wb") as f:
                f.write(r.content)
        except Exception:
            pass

    try:
        return ImageFont.truetype(font_path, font_size)
    except Exception:
        return ImageFont.load_default()


# -----------------------------
# 4. AI Label + Category
# -----------------------------
def classify_image(raw_bytes: bytes):
    client = get_openai_client()
    if not client:
        return {"name": "Unknown Asset", "category": "General"}

    base_img = base64.b64encode(raw_bytes).decode("utf-8")

    prompt = """
You are an image classification assistant.

Look at the image and return:
1. a short object name
2. a broader category

Rules:
- "name" must be 1 to 3 words only
- "category" must be 1 or 2 words only
- use title case
- do not use the uploaded file name
- identify what is visually present
- if not clear, return:
  {"name":"Unknown Asset","category":"General"}

Return ONLY valid JSON in this exact format:
{"name":"Object Name","category":"Category Name"}
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
                            "image_url": {"url": f"data:image/png;base64,{base_img}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )

        parsed = json.loads(response.choices[0].message.content)
        name = parsed.get("name", "Unknown Asset").strip()
        category = parsed.get("category", "General").strip()

        if not name:
            name = "Unknown Asset"
        if not category:
            category = "General"

        return {"name": name, "category": category}

    except Exception:
        return {"name": "Unknown Asset", "category": "General"}


# -----------------------------
# 5. Common helpers
# -----------------------------
def wrap_text(draw, text, font, max_width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        test_line = current + " " + word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test_line
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def sort_items_by_category(items):
    grouped = {}
    for item in items:
        cat = item.get("category", "General")
        grouped.setdefault(cat, []).append(item)

    sorted_grouped = dict(sorted(grouped.items(), key=lambda x: x[0].lower()))

    final_items = []
    for cat, cat_items in sorted_grouped.items():
        cat_items = sorted(cat_items, key=lambda x: x.get("display_name", "").lower())
        final_items.extend(cat_items)

    return final_items, sorted_grouped


# -----------------------------
# 6. Standard Collage
# -----------------------------
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
    font = load_font(font_size=font_size, bold=True)

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

        name_txt = item.get("display_name", "UNKNOWN ASSET").upper()

        bbox = draw.textbbox((0, 0), name_txt, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad_x = 30
        pad_y = 15
        box_w = text_w + (2 * pad_x)
        box_h = text_h + (2 * pad_y)

        px = (tile_w - box_w) // 2
        py = tile_h - box_h - 35

        draw.rounded_rectangle(
            [px, py, px + box_w, py + box_h],
            radius=16,
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


# -----------------------------
# 7. AI Smart Grouped Collage
# -----------------------------
def render_smart_grouped_collage(items, show_category_headers=True):
    if not items:
        return None

    items, grouped = sort_items_by_category(items)

    canvas_w = 2200
    outer_margin = 60
    group_gap = 55
    tile_gap = 28
    group_header_h = 80
    group_section_pad = 28
    tile_w = 300
    tile_h = 240
    radius = 24
    label_font = load_font(30, bold=True)
    cat_font = load_font(40, bold=True)
    small_font = load_font(24, bold=False)

    # Estimate height
    total_h = outer_margin
    for cat, cat_items in grouped.items():
        n = len(cat_items)
        cols = min(4, max(1, n))
        rows = math.ceil(n / cols)
        group_h = (
            (group_header_h if show_category_headers else 0)
            + (rows * tile_h)
            + ((rows - 1) * tile_gap)
            + (2 * group_section_pad)
        )
        total_h += group_h + group_gap
    total_h += outer_margin

    canvas = Image.new("RGBA", (canvas_w, total_h), (246, 248, 252, 255))
    draw_canvas = ImageDraw.Draw(canvas)

    y_cursor = outer_margin

    category_colors = [
        (233, 243, 255, 255),
        (240, 248, 240, 255),
        (255, 245, 235, 255),
        (248, 240, 255, 255),
        (255, 242, 246, 255),
        (242, 250, 250, 255)
    ]

    for idx_cat, (cat, cat_items) in enumerate(grouped.items()):
        n = len(cat_items)
        cols = min(4, max(1, n))
        rows = math.ceil(n / cols)

        section_h = (
            (group_header_h if show_category_headers else 0)
            + (rows * tile_h)
            + ((rows - 1) * tile_gap)
            + (2 * group_section_pad)
        )

        section_x1 = outer_margin
        section_y1 = y_cursor
        section_x2 = canvas_w - outer_margin
        section_y2 = y_cursor + section_h

        section_fill = category_colors[idx_cat % len(category_colors)]
        draw_canvas.rounded_rectangle(
            [section_x1, section_y1, section_x2, section_y2],
            radius=32,
            fill=section_fill,
            outline=(220, 226, 235, 255),
            width=2
        )

        inner_y = y_cursor + group_section_pad

        if show_category_headers:
            header_x = section_x1 + 25
            header_y = inner_y

            header_text = cat.upper()
            draw_canvas.text((header_x, header_y), header_text, font=cat_font, fill=(30, 41, 59, 255))

            count_text = f"{len(cat_items)} image{'s' if len(cat_items) > 1 else ''}"
            draw_canvas.text((header_x, header_y + 42), count_text, font=small_font, fill=(71, 85, 105, 255))

            inner_y += group_header_h

        available_w = (section_x2 - section_x1) - (2 * group_section_pad)
        tile_w_eff = (available_w - ((cols - 1) * tile_gap)) // cols

        for idx_item, item in enumerate(cat_items):
            r = idx_item // cols
            c = idx_item % cols

            x = section_x1 + group_section_pad + c * (tile_w_eff + tile_gap)
            y = inner_y + r * (tile_h + tile_gap)

            raw_bytes = st.session_state["images_bytes"].get(item["id"])
            if not raw_bytes:
                continue

            raw_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            fitted = ImageOps.fit(raw_img, (tile_w_eff, tile_h), Image.LANCZOS)

            tile = Image.new("RGBA", (tile_w_eff, tile_h), (255, 255, 255, 0))
            mask = Image.new("L", (tile_w_eff, tile_h), 0)
            ImageDraw.Draw(mask).rounded_rectangle((0, 0, tile_w_eff, tile_h), radius=radius, fill=255)
            tile.paste(fitted, (0, 0), mask)

            tile_draw = ImageDraw.Draw(tile)
            tile_draw.rounded_rectangle(
                (0, 0, tile_w_eff, tile_h),
                radius=radius,
                outline=(255, 255, 255, 255),
                width=4
            )

            name_txt = item.get("display_name", "Unknown Asset")
            lines = wrap_text(tile_draw, name_txt.upper(), label_font, tile_w_eff - 50)

            line_heights = []
            for line in lines:
                bb = tile_draw.textbbox((0, 0), line, font=label_font)
                line_heights.append(bb[3] - bb[1])

            total_text_h = sum(line_heights) + ((len(lines) - 1) * 6)
            text_box_h = total_text_h + 24
            text_box_y = tile_h - text_box_h - 16

            tile_draw.rounded_rectangle(
                [16, text_box_y, tile_w_eff - 16, tile_h - 16],
                radius=16,
                fill=(15, 23, 42, 215)
            )

            cy = text_box_y + 12
            for i, line in enumerate(lines):
                h = line_heights[i]
                tile_draw.text(
                    (tile_w_eff // 2, cy + h // 2),
                    line,
                    font=label_font,
                    fill="white",
                    anchor="mm"
                )
                cy += h + 6

            canvas.alpha_composite(tile, (x, y))

        y_cursor = section_y2 + group_gap

    return canvas.convert("RGB")


# -----------------------------
# 8. Sidebar Upload
# -----------------------------
with st.sidebar:
    st.header("📤 Media Input")
    uploaded_files = st.file_uploader(
        "Upload Images",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "webp"]
    )

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
                        display_name="Processing...",
                        category="General"
                    )
                )
            )

        old_ids = {m["id"] for m in st.session_state["images_meta"]}
        new_ids = {m["id"] for m in new_meta}

        if old_ids != new_ids:
            st.session_state["images_bytes"] = new_bytes
            st.session_state["images_meta"] = new_meta
            st.session_state["generated_collage"] = None
            st.session_state["smart_collage_generated"] = False

            keys_to_delete = [
                k for k in list(st.session_state.keys())
                if k.startswith("dn_") or k.startswith("cat_") or k.startswith("inp_")
            ]
            for k in keys_to_delete:
                del st.session_state[k]

            with st.spinner("AI is naming and categorizing images..."):
                for m in st.session_state["images_meta"]:
                    result = classify_image(st.session_state["images_bytes"][m["id"]])

                    label = result.get("name", "Unknown Asset").strip() or "Unknown Asset"
                    category = result.get("category", "General").strip() or "General"

                    st.session_state[f"dn_{m['id']}"] = label
                    st.session_state[f"cat_{m['id']}"] = category
                    m["display_name"] = label
                    m["category"] = category

            st.rerun()


# -----------------------------
# 9. Main UI
# -----------------------------
if st.session_state["images_meta"]:
    st.markdown('<p class="main-header">🖼️ AI Image Studio</p>', unsafe_allow_html=True)

    t1, t2 = st.tabs(["📝 AI & Labels", "🎨 Style & Layout"])

    with t1:
        c1, c2 = st.columns(2)

        with c1:
            if st.button("✨ RUN AI AUTO-LABEL", use_container_width=True, type="primary"):
                with st.spinner("Analyzing images..."):
                    for m in st.session_state["images_meta"]:
                        result = classify_image(st.session_state["images_bytes"][m["id"]])

                        label = result.get("name", "Unknown Asset").strip() or "Unknown Asset"
                        category = result.get("category", "General").strip() or "General"

                        st.session_state[f"dn_{m['id']}"] = label
                        st.session_state[f"cat_{m['id']}"] = category
                        m["display_name"] = label
                        m["category"] = category
                st.rerun()

        with c2:
            if st.button("🧠 AI SMART GROUP & COLLAGE", use_container_width=True, type="primary"):
                with st.spinner("AI is grouping images and creating an attractive collage..."):
                    for m in st.session_state["images_meta"]:
                        result = classify_image(st.session_state["images_bytes"][m["id"]])

                        label = result.get("name", "Unknown Asset").strip() or "Unknown Asset"
                        category = result.get("category", "General").strip() or "General"

                        st.session_state[f"dn_{m['id']}"] = label
                        st.session_state[f"cat_{m['id']}"] = category
                        m["display_name"] = label
                        m["category"] = category

                    st.session_state["generated_collage"] = render_smart_grouped_collage(
                        st.session_state["images_meta"],
                        show_category_headers=True
                    )
                    st.session_state["smart_collage_generated"] = True

                st.rerun()

        st.divider()

        for m in st.session_state["images_meta"]:
            col_a, col_b, col_c = st.columns([1, 4, 3])
            col_a.image(st.session_state["images_bytes"][m["id"]], width=100)

            new_label = col_b.text_input(
                "Label",
                value=st.session_state.get(f"dn_{m['id']}", m["display_name"]),
                key=f"inp_{m['id']}"
            )
            new_cat = col_c.text_input(
                "Category",
                value=st.session_state.get(f"cat_{m['id']}", m.get("category", "General")),
                key=f"catinp_{m['id']}"
            )

            st.session_state[f"dn_{m['id']}"] = new_label.strip() if new_label.strip() else "Unknown Asset"
            st.session_state[f"cat_{m['id']}"] = new_cat.strip() if new_cat.strip() else "General"
            m["display_name"] = st.session_state[f"dn_{m['id']}"]
            m["category"] = st.session_state[f"cat_{m['id']}"]

    with t2:
        st.subheader("📏 Standard Collage Controls")

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
        b_color = col7.color_picker("Border Color", "#3B82F6")
        bg_color = col8.color_picker("Background Color", "#FFFFFF")

        font_size = st.slider("Label Font Size", 20, 120, 40)

        if st.button("🚀 GENERATE STANDARD COLLAGE", use_container_width=True, type="primary"):
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
            st.session_state["smart_collage_generated"] = False

    if st.session_state["generated_collage"]:
        st.divider()

        if st.session_state["smart_collage_generated"]:
            st.subheader("🧠 AI Smart Grouped Collage")
        else:
            st.subheader("🖼️ Generated Collage")

        st.image(st.session_state["generated_collage"], use_container_width=True)

        buf = io.BytesIO()
        st.session_state["generated_collage"].save(buf, format="PNG")
        st.download_button(
            "📥 Download Collage",
            buf.getvalue(),
            file_name="ai_smart_collage.png" if st.session_state["smart_collage_generated"] else "collage.png",
            use_container_width=True
        )
else:
    st.info("Please upload images in the sidebar to start.")
