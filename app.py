import io
import os
import math
import json
import base64
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from collections import defaultdict

import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageColor

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from streamlit_elements import elements, dashboard, mui, sync
    ELEMENTS_AVAILABLE = True
except Exception:
    ELEMENTS_AVAILABLE = False


st.set_page_config(page_title="AI Collage Studio", page_icon="🖼️", layout="wide")
st.title("🖼️ AI Collage Studio")
st.caption("Upload images, classify them, edit names, drag and place images, and generate a polished collage.")


@dataclass
class ImageItem:
    id: str
    original_file_name: str
    category: str = "General"
    subcategory: str = "General"
    display_name: str = "Image Highlight"
    confidence: int = 0
    display_order: int = 0
    highlight: bool = False
    grid_x: int = 0
    grid_y: int = 0
    grid_w: int = 3
    grid_h: int = 4


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def get_openai_client():
    api_key = ""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "").strip()
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key or OpenAI is None:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates = [
            "DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "arialbd.ttf",
        ]
    else:
        candidates = [
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "arial.ttf",
        ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def safe_open_image(file_bytes: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    return img.convert("RGB")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buffer = io.BytesIO()
    save_kwargs = {}

    if fmt.upper() == "JPEG":
        img = img.convert("RGB")
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True

    if fmt.upper() == "PDF":
        img = img.convert("RGB")

    img.save(buffer, format=fmt.upper(), **save_kwargs)
    return buffer.getvalue()


def hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
    return ImageColor.getrgb(color_hex)


def image_to_data_uri(raw_bytes: bytes, mime_type: str = "image/png") -> str:
    encoded = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def parse_json_from_text(text: str) -> Dict:
    if not text:
        return {}

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    return {}


def create_mask(size: Tuple[int, int], radius: int, shape_style: str = "Rounded Rectangle") -> Image.Image:
    w, h = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    if shape_style == "Rectangle":
        draw.rectangle((0, 0, w, h), fill=255)
    elif shape_style == "Rounded Rectangle":
        draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)
    elif shape_style == "Ellipse":
        draw.ellipse((0, 0, w, h), fill=255)
    elif shape_style == "Circle":
        side = min(w, h)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        draw.ellipse((x0, y0, x0 + side, y0 + side), fill=255)
    else:
        draw.rounded_rectangle((0, 0, w, h), radius=radius, fill=255)

    return mask


def fit_to_tile(img: Image.Image, size: Tuple[int, int], radius: int, shape_style: str) -> Image.Image:
    fitted = ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)
    fitted = fitted.convert("RGBA")
    mask = create_mask(size, radius, shape_style)
    canvas = Image.new("RGBA", size, (255, 255, 255, 0))
    canvas.paste(fitted, (0, 0), mask)
    return canvas


def add_shadow(card: Image.Image, radius: int = 18, blur: int = 12, offset=(6, 8), opacity: int = 65, shape_style: str = "Rounded Rectangle") -> Image.Image:
    shadow_canvas = Image.new("RGBA", (card.width + 40, card.height + 40), (0, 0, 0, 0))
    shadow_shape = Image.new("RGBA", card.size, (0, 0, 0, opacity))
    mask = create_mask(card.size, radius, shape_style)
    shadow_canvas.paste(shadow_shape, (14 + offset[0], 14 + offset[1]), mask)
    shadow_canvas = shadow_canvas.filter(ImageFilter.GaussianBlur(blur))
    shadow_canvas.paste(card, (14, 14), card)
    return shadow_canvas


def wrap_text(draw, text: str, font, max_width: int):
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def draw_wrapped_text(draw, x: int, y: int, text: str, font, fill, max_width: int, line_gap: int = 2, max_lines: int = 2):
    lines = wrap_text(draw, text, font, max_width)[:max_lines]
    cur_y = y

    for line in lines:
        draw.text((x, cur_y), line, font=font, fill=fill)
        bbox = draw.textbbox((0, 0), line, font=font)
        cur_y += (bbox[3] - bbox[1]) + line_gap


def upload_signature(files) -> str:
    parts = []
    for f in files:
        content = f.getvalue()
        digest = hashlib.md5(content).hexdigest()
        parts.append(f"{f.name}:{len(content)}:{digest}")
    joined = "|".join(parts)
    return hashlib.md5(joined.encode()).hexdigest()


def tile_ratio(shape_format: str) -> float:
    if shape_format == "Landscape":
        return 0.72
    if shape_format == "Portrait":
        return 1.18
    if shape_format == "Square":
        return 1.0
    if shape_format == "Tall Portrait":
        return 1.35
    return 0.92


# --------------------------------------------------
# AI classification
# --------------------------------------------------
def classify_image_with_openai(raw_bytes: bytes, model_name: str) -> Dict:
    client = get_openai_client()
    if client is None:
        return {
            "category": "General",
            "subcategory": "General",
            "display_name": "Image Highlight",
            "confidence": 0,
        }

    image_data_uri = image_to_data_uri(raw_bytes, "image/png")

    instruction = (
        "Analyze this image for a professional presentation collage. "
        "Return STRICT JSON only in this format: "
        '{"category":"short category","subcategory":"short subcategory","display_name":"short specific presentation label","confidence":0}. '
        "The display_name must be specific, concise, and presentation-friendly, such as "
        "'3D Printer Setup', 'Lab Equipment Unit', 'Testing Station', 'Technical Device', 'Research Instrument', or 'Machine Interface'. "
        "Avoid generic labels like 'Image Highlight' or 'Visual Highlight'. "
        "Choose category from: People, Event, Classroom, Laboratory, Technology, Building, Nature, Workshop, Research, Industrial Visit, Wellness, Culture, Other."
    )

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": image_data_uri, "detail": "low"},
                    ],
                }
            ],
        )

        output_text = getattr(response, "output_text", "") or ""
        parsed = parse_json_from_text(output_text)

        return {
            "category": parsed.get("category", "General"),
            "subcategory": parsed.get("subcategory", "General"),
            "display_name": parsed.get("display_name", parsed.get("subcategory", "Image Highlight")),
            "confidence": int(parsed.get("confidence", 0) or 0),
        }
    except Exception:
        return {
            "category": "General",
            "subcategory": "General",
            "display_name": "Image Highlight",
            "confidence": 0,
        }


# --------------------------------------------------
# State helpers
# --------------------------------------------------
def ensure_state():
    if "images_meta" not in st.session_state:
        st.session_state["images_meta"] = []
    if "images_bytes" not in st.session_state:
        st.session_state["images_bytes"] = {}
    if "generated_collage" not in st.session_state:
        st.session_state["generated_collage"] = None
    if "upload_sig" not in st.session_state:
        st.session_state["upload_sig"] = None
    if "dashboard_layout" not in st.session_state:
        st.session_state["dashboard_layout"] = []


def get_meta_items() -> List[ImageItem]:
    return [ImageItem(**d) for d in st.session_state["images_meta"]]


def set_meta_items(items: List[ImageItem]):
    st.session_state["images_meta"] = [asdict(x) for x in items]


def push_meta_to_widget_state():
    for item in get_meta_items():
        st.session_state[f"display_name_{item.id}"] = item.display_name
        st.session_state[f"category_{item.id}"] = item.category
        st.session_state[f"order_{item.id}"] = int(item.display_order)
        st.session_state[f"highlight_{item.id}"] = bool(item.highlight)


def sync_edits_from_widgets():
    updated = get_meta_items()
    for item in updated:
        item.display_name = st.session_state.get(f"display_name_{item.id}", item.display_name)
        item.category = st.session_state.get(f"category_{item.id}", item.category)
        item.display_order = int(st.session_state.get(f"order_{item.id}", item.display_order))
        item.highlight = bool(st.session_state.get(f"highlight_{item.id}", item.highlight))
    set_meta_items(updated)
    return updated


def init_dashboard_layout(items: List[ImageItem]):
    st.session_state["dashboard_layout"] = [
        dashboard.Item(
            item.id,
            item.grid_x,
            item.grid_y,
            item.grid_w,
            item.grid_h,
            isDraggable=True,
            isResizable=True,
        )
        for item in items
    ]


def sync_dashboard_to_meta():
    items = get_meta_items()
    layout_map = {}

    raw_layout = st.session_state.get("dashboard", None)
    if raw_layout:
        for entry in raw_layout:
            layout_map[entry["i"]] = entry

    for item in items:
        if item.id in layout_map:
            entry = layout_map[item.id]
            item.grid_x = int(entry["x"])
            item.grid_y = int(entry["y"])
            item.grid_w = int(entry["w"])
            item.grid_h = int(entry["h"])

    set_meta_items(items)
    return items


# --------------------------------------------------
# Collage drawing
# --------------------------------------------------
def draw_card(canvas: Image.Image, img: Image.Image, item: ImageItem, box, radius: int, show_category_tag: bool, shape_style: str):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    tile = fit_to_tile(img, (w, h), radius, shape_style)
    card = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    card.paste(tile, (0, 0), tile)

    draw = ImageDraw.Draw(card)

    overlay_h = 54 if show_category_tag else 42
    overlay_y = h - overlay_h - 8

    draw.rounded_rectangle(
        (10, overlay_y, w - 10, h - 10),
        radius=14,
        fill=(255, 255, 255, 235),
    )

    text_x = 22
    cur_y = overlay_y + 8

    if show_category_tag:
        tag_font = load_font(11, bold=False)
        cat = item.category.strip() if item.category else "General"
        draw.text((text_x, cur_y), cat, font=tag_font, fill=(110, 110, 110))
        cur_y += 14

    title_font = load_font(15, bold=True)
    title = item.display_name.strip() if item.display_name else "Image Highlight"
    draw_wrapped_text(draw, text_x, cur_y, title[:70], title_font, (28, 28, 28), w - 40)

    shadowed = add_shadow(card, radius=radius, shape_style=shape_style)
    canvas.alpha_composite(shadowed, (x1 - 14, y1 - 14))


def build_grouped_collage(
    items: List[ImageItem],
    width: int,
    bg_rgb: Tuple[int, int, int],
    gap: int,
    radius: int,
    show_category_tag: bool,
    images_per_row: int = 3,
    show_group_headings: bool = False,
    section_title_align: str = "Left",
    highlight_scale: float = 1.12,
    shape_style: str = "Rounded Rectangle",
    shape_format: str = "Classic",
):
    groups = defaultdict(list)
    for item in items:
        key = item.category.strip() if item.category else "General"
        groups[key].append(item)

    for cat in groups:
        groups[cat] = sorted(groups[cat], key=lambda x: (x.display_order, x.original_file_name))

    top_margin = 28
    side_margin = 44
    section_gap = 26
    heading_h = 26 if show_group_headings else 0

    usable_w = width - 2 * side_margin
    base_tile_w = (usable_w - (images_per_row - 1) * gap) // images_per_row
    base_tile_h = int(base_tile_w * tile_ratio(shape_format))

    est_h = top_margin + 10
    for _, g in groups.items():
        rows = math.ceil(len(g) / images_per_row)
        est_h += heading_h + rows * base_tile_h + (rows - 1) * gap + section_gap

    height = max(900, est_h)
    canvas = Image.new("RGBA", (width, height), bg_rgb + (255,))
    draw = ImageDraw.Draw(canvas)
    y = top_margin
    heading_font = load_font(max(18, width // 56), bold=True)

    for category, group_items in groups.items():
        if show_group_headings:
            bbox = draw.textbbox((0, 0), category, font=heading_font)
            heading_w = bbox[2] - bbox[0]
            if section_title_align == "Center":
                heading_x = (width - heading_w) // 2
            else:
                heading_x = side_margin
            draw.text((heading_x, y), category, font=heading_font, fill=(42, 42, 42))
            y += heading_h

        idx = 0
        rows = math.ceil(len(group_items) / images_per_row)

        for r in range(rows):
            row_items = group_items[idx: idx + images_per_row]
            row_count = len(row_items)
            row_width = row_count * base_tile_w + (row_count - 1) * gap
            row_start_x = (width - row_width) // 2

            for c, item in enumerate(row_items):
                cur_w = base_tile_w
                cur_h = base_tile_h

                if item.highlight:
                    cur_w = int(base_tile_w * highlight_scale)
                    cur_h = int(base_tile_h * highlight_scale)

                x1 = row_start_x + c * (base_tile_w + gap)
                y1 = y + r * (base_tile_h + gap)
                x2 = x1 + cur_w
                y2 = y1 + cur_h

                img = safe_open_image(st.session_state["images_bytes"][item.id])
                draw_card(canvas, img, item, (x1, y1, x2, y2), radius, show_category_tag, shape_style)

            idx += images_per_row

        y += rows * (base_tile_h + gap) + section_gap

    return canvas.convert("RGB")


def build_grid_collage(
    items: List[ImageItem],
    width: int,
    height: int,
    bg_rgb: Tuple[int, int, int],
    gap: int,
    radius: int,
    show_category_tag: bool,
    shape_style: str,
    shape_format: str,
):
    canvas = Image.new("RGBA", (width, height), bg_rgb + (255,))
    margin_x = 44
    margin_y = 28

    cols = math.ceil(math.sqrt(len(items)))
    rows = math.ceil(len(items) / cols)

    usable_w = width - 2 * margin_x - (cols - 1) * gap
    tile_w = max(160, usable_w // cols)
    tile_h = max(170, int(tile_w * tile_ratio(shape_format)))

    idx = 0
    for r in range(rows):
        remaining = len(items) - idx
        row_count = min(cols, remaining)
        row_width = row_count * tile_w + (row_count - 1) * gap
        row_start_x = (width - row_width) // 2

        for c in range(row_count):
            item = items[idx]
            x1 = row_start_x + c * (tile_w + gap)
            y1 = margin_y + r * (tile_h + gap)
            x2 = x1 + tile_w
            y2 = y1 + tile_h
            img = safe_open_image(st.session_state["images_bytes"][item.id])
            draw_card(canvas, img, item, (x1, y1, x2, y2), radius, show_category_tag, shape_style)
            idx += 1

    return canvas.convert("RGB")


def build_manual_grid_collage(
    items: List[ImageItem],
    width: int,
    bg_rgb: Tuple[int, int, int],
    radius: int,
    show_category_tag: bool,
    shape_style: str,
):
    cols = 12
    gutter = 12
    canvas_margin = 20
    cell_px = (width - 2 * canvas_margin) / cols

    max_bottom = 0
    for item in items:
        bottom = (item.grid_y + item.grid_h) * cell_px
        if bottom > max_bottom:
            max_bottom = bottom

    height = max(900, int(max_bottom + 80))
    canvas = Image.new("RGBA", (width, height), bg_rgb + (255,))

    for item in items:
        x1 = int(canvas_margin + item.grid_x * cell_px + gutter / 2)
        y1 = int(canvas_margin + item.grid_y * cell_px + gutter / 2)
        x2 = int(canvas_margin + (item.grid_x + item.grid_w) * cell_px - gutter / 2)
        y2 = int(canvas_margin + (item.grid_y + item.grid_h) * cell_px - gutter / 2)

        img = safe_open_image(st.session_state["images_bytes"][item.id])
        draw_card(canvas, img, item, (x1, y1, x2, y2), radius, show_category_tag, shape_style)

    return canvas.convert("RGB")


# --------------------------------------------------
# App
# --------------------------------------------------
ensure_state()

with st.sidebar:
    st.header("Settings")

    base_layout_options = ["Grouped by AI Category", "Grid"]
    if ELEMENTS_AVAILABLE:
        base_layout_options.append("Flexible Grid")

    layout_style = st.selectbox("Layout Style", base_layout_options, index=0)
    aspect_ratio = st.selectbox("Canvas Ratio", ["16:9", "4:3", "1:1"], index=0)
    canvas_width = st.slider("Canvas Width", 1200, 4000, 2200, 100)
    gap = st.slider("Spacing", 8, 40, 14, 2)
    radius = st.slider("Corner Radius", 0, 40, 16, 2)
    bg_color = st.color_picker("Background Color", "#F4F4F6")

    st.subheader("Tile Options")
    shape_style = st.selectbox("Shape Style", ["Rounded Rectangle", "Rectangle", "Ellipse", "Circle"], index=0)
    shape_format = st.selectbox("Shape Format", ["Classic", "Landscape", "Portrait", "Square", "Tall Portrait"], index=0)

    st.subheader("Label Options")
    show_category_tag = st.toggle("Show category on each card", value=False)
    show_group_headings = st.toggle("Show category group headings", value=False)

    st.subheader("Layout Controls")
    section_title_align = st.selectbox("Group Heading Alignment", ["Left", "Center"], index=0)
    images_per_row = st.slider("Images Per Row", 2, 5, 3)
    highlight_scale = st.slider("Highlight Image Scale", 1.0, 1.5, 1.12, 0.05)

    st.subheader("AI")
    use_ai = st.toggle("Use OpenAI vision classification", True)
    model_name = st.text_input("Model name", "gpt-5.2")

uploaded_files = st.file_uploader(
    "Upload multiple images",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    new_sig = upload_signature(uploaded_files)
    if st.session_state["upload_sig"] != new_sig:
        meta_items = []
        image_bytes = {}

        for idx, file in enumerate(uploaded_files):
            try:
                file_bytes = file.getvalue()
                _ = safe_open_image(file_bytes)
                uid = f"img_{idx}_{hashlib.md5(file_bytes).hexdigest()[:10]}"
                image_bytes[uid] = file_bytes

                row = idx // 4
                col = idx % 4

                meta_items.append(
                    ImageItem(
                        id=uid,
                        original_file_name=file.name,
                        display_order=idx + 1,
                        grid_x=col * 3,
                        grid_y=row * 4,
                        grid_w=3,
                        grid_h=4,
                    )
                )
            except Exception as e:
                st.warning(f"Could not read {file.name}: {e}")

        st.session_state["images_bytes"] = image_bytes
        set_meta_items(meta_items)
        push_meta_to_widget_state()
        init_dashboard_layout(meta_items)
        st.session_state["generated_collage"] = None
        st.session_state["upload_sig"] = new_sig

items = get_meta_items()

if st.button("Classify Images", use_container_width=True, type="primary") and items:
    if use_ai:
        progress = st.progress(0)
        status = st.empty()

        for i, item in enumerate(items):
            status.write(f"Analyzing {item.original_file_name}...")
            result = classify_image_with_openai(st.session_state["images_bytes"][item.id], model_name)
            item.category = result.get("category", "General")
            item.subcategory = result.get("subcategory", "General")
            item.display_name = result.get("display_name", "Image Highlight")
            item.confidence = result.get("confidence", 0)
            progress.progress((i + 1) / len(items))

        set_meta_items(items)
        push_meta_to_widget_state()
        status.success("Classification completed.")
    else:
        for item in items:
            item.category = "General"
            item.display_name = "Image Highlight"
        set_meta_items(items)
        push_meta_to_widget_state()

items = get_meta_items()

if items:
    st.subheader("Images")
    cols = st.columns(4)
    for idx, item in enumerate(items):
        with cols[idx % 4]:
            st.image(safe_open_image(st.session_state["images_bytes"][item.id]), use_container_width=True)
            st.markdown(f"**{st.session_state.get(f'display_name_{item.id}', item.display_name)}**")
            st.caption(f"Category: {st.session_state.get(f'category_{item.id}', item.category)}")

if items:
    st.subheader("Edit Display Names and Grouping")
    for item in items:
        current_title = st.session_state.get(f"display_name_{item.id}", item.display_name)
        with st.expander(f"Edit: {current_title}", expanded=False):
            a, b = st.columns([1, 2])

            with a:
                st.image(safe_open_image(st.session_state["images_bytes"][item.id]), use_container_width=True)

            with b:
                st.text_input("Display Name", key=f"display_name_{item.id}")
                st.text_input("Category", key=f"category_{item.id}")
                st.number_input("Display Order", min_value=1, step=1, key=f"order_{item.id}")
                st.checkbox("Highlight", key=f"highlight_{item.id}")

items = sync_edits_from_widgets()

if layout_style == "Flexible Grid":
    if not ELEMENTS_AVAILABLE:
        st.warning("Flexible Grid needs `streamlit-elements` installed.")
    else:
        st.subheader("Flexible Grid Editor")
        st.caption("Drag and resize the cards below. Then click Generate Collage.")

        if "dashboard_layout" not in st.session_state or not st.session_state["dashboard_layout"]:
            init_dashboard_layout(items)

        with elements("flex_grid_editor"):
            with dashboard.Grid(
                st.session_state["dashboard_layout"],
                cols=12,
                rowHeight=70,
                onLayoutChange=sync("dashboard"),
                draggableHandle=".drag-handle",
            ):
                for item in items:
                    img_b64 = base64.b64encode(st.session_state["images_bytes"][item.id]).decode()
                    card_title = st.session_state.get(f"display_name_{item.id}", item.display_name)

                    with mui.Paper(
                        key=item.id,
                        elevation=3,
                        sx={
                            "display": "flex",
                            "flexDirection": "column",
                            "height": "100%",
                            "borderRadius": "18px",
                            "overflow": "hidden",
                            "backgroundColor": "#ffffff",
                        },
                    ):
                        with mui.Box(
                            className="drag-handle",
                            sx={
                                "padding": "8px 12px",
                                "cursor": "move",
                                "fontSize": "13px",
                                "fontWeight": 600,
                                "backgroundColor": "#f7f7fa",
                                "borderBottom": "1px solid #ececf2",
                                "lineHeight": 1.3,
                            },
                        ):
                            mui.Typography(card_title)

                        mui.Box(
                            component="img",
                            src=f"data:image/png;base64,{img_b64}",
                            sx={
                                "width": "100%",
                                "height": "calc(100% - 44px)",
                                "objectFit": "cover",
                            },
                        )

if items and st.button("Generate Collage", type="primary", use_container_width=True):
    items = sync_edits_from_widgets()

    if layout_style == "Flexible Grid" and ELEMENTS_AVAILABLE:
        items = sync_dashboard_to_meta()

    ordered_items = sorted(items, key=lambda x: (x.category, x.display_order, x.original_file_name))
    width = canvas_width
    bg_rgb = hex_to_rgb(bg_color)

    if aspect_ratio == "16:9":
        height = int(width * 9 / 16)
    elif aspect_ratio == "4:3":
        height = int(width * 3 / 4)
    else:
        height = width

    if layout_style == "Grouped by AI Category":
        collage = build_grouped_collage(
            items=ordered_items,
            width=width,
            bg_rgb=bg_rgb,
            gap=gap,
            radius=radius,
            show_category_tag=show_category_tag,
            images_per_row=images_per_row,
            show_group_headings=show_group_headings,
            section_title_align=section_title_align,
            highlight_scale=highlight_scale,
            shape_style=shape_style,
            shape_format=shape_format,
        )
    elif layout_style == "Flexible Grid" and ELEMENTS_AVAILABLE:
        collage = build_manual_grid_collage(
            items=items,
            width=width,
            bg_rgb=bg_rgb,
            radius=radius,
            show_category_tag=show_category_tag,
            shape_style=shape_style,
        )
    else:
        collage = build_grid_collage(
            items=ordered_items,
            width=width,
            height=height,
            bg_rgb=bg_rgb,
            gap=gap,
            radius=radius,
            show_category_tag=show_category_tag,
            shape_style=shape_style,
            shape_format=shape_format,
        )

    st.session_state["generated_collage"] = collage

if st.session_state["generated_collage"] is not None:
    st.subheader("Collage Preview")
    st.image(st.session_state["generated_collage"], use_container_width=True)

    png_data = pil_to_bytes(st.session_state["generated_collage"], "PNG")
    jpg_data = pil_to_bytes(st.session_state["generated_collage"], "JPEG", quality=96)
    pdf_data = pil_to_bytes(st.session_state["generated_collage"], "PDF")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Download PNG", data=png_data, file_name="ai_collage.png", mime="image/png", use_container_width=True)
    with d2:
        st.download_button("Download JPEG", data=jpg_data, file_name="ai_collage.jpg", mime="image/jpeg", use_container_width=True)
    with d3:
        st.download_button("Download PDF", data=pdf_data, file_name="ai_collage.pdf", mime="application/pdf", use_container_width=True)

with st.expander("Notes"):
    st.markdown(
        "- Flexible Grid uses a drag-and-resize editor and saves tile positions into the image metadata.\n"
        "- Edited names are preserved unless the uploaded file set changes.\n"
        "- Add `streamlit-elements==0.1.0` to requirements for drag-and-place support.\n"
    )
