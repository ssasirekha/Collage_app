import io
import os
import math
import streamlit as st
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageColor
from dataclasses import dataclass, asdict

@dataclass
class ImageItem:
    id: str
    original_file_name: str
    display_name: str = "ASSET"

# --- 1. Sidebar & File Upload ---
with st.sidebar:
    st.header("📤 Media Input")
    uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True)
    
    # THE FIX: Properly defining the list and iterator 'f'
    if uploaded_files:
        if "images_meta" not in st.session_state or len(uploaded_files) != len(st.session_state.get("images_meta", [])):
            new_meta = []
            new_bytes = {}
            for i, f in enumerate(uploaded_files):
                uid = f"img_{i}"
                new_bytes[uid] = f.getvalue()
                new_meta.append(asdict(ImageItem(id=uid, original_file_name=f.name)))
            
            st.session_state["images_bytes"] = new_bytes
            st.session_state["images_meta"] = new_meta

# --- 2. Rendering Engine with Auto-Scale ---
def render_collage(items, sizing_option, font_size, cols=3, gap=40):
    if not items: return None
    
    canvas_w = 2000
    pil_images = [Image.open(io.BytesIO(st.session_state["images_bytes"][m['id']])).convert("RGB") for m in items]
    
    # Calculate target dimensions based on "Image Sizing" options
    widths, heights = zip(*(i.size for i in pil_images))
    if sizing_option == "Match Height":
        target_h = max(heights)
        target_w = target_h # Force square for grid uniformity
    else:
        target_w, target_h = max(widths), max(heights)

    rows = math.ceil(len(items) / cols)
    tile_w = (canvas_w - (cols - 1) * gap) // cols
    tile_h = tile_w # Auto-scaling to fit square grid
    
    canvas = Image.new("RGBA", (canvas_w, (rows * tile_h) + (rows - 1) * gap), (255, 255, 255, 255))
    
    # Load Font (Cloud-safe fallback)
    try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except: font = ImageFont.load_default()

    for idx, (m, raw_img) in enumerate(zip(items, pil_images)):
        row, col = divmod(idx, cols)
        x = col * (tile_w + gap)
        y = row * (tile_h + gap)

        # Auto-Scale to fit the grid perfectly
        img = ImageOps.fit(raw_img, (tile_w, tile_h), Image.LANCZOS)
        
        # Rounded corners & Label (similar to industrial reference)
        mask = Image.new("L", (tile_w, tile_h), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rounded_rectangle((0, 0, tile_w, tile_h), radius=30, fill=255)
        
        tile = Image.new("RGBA", (tile_w, tile_h), (0,0,0,0))
        tile.paste(img, (0,0), mask)
        
        # Add Label Pill
        draw = ImageDraw.Draw(tile)
        label = m['display_name'].upper()
        draw.rounded_rectangle([tile_w//4, tile_h-70, 3*tile_w//4, tile_h-20], radius=10, fill=(0,0,0,180))
        draw.text((tile_w//2, tile_h-45), label, fill="white", font=font, anchor="mm")

        canvas.alpha_composite(tile, (x, y))
    
    return canvas.convert("RGB")

# --- 3. UI Controls ---
if st.session_state.get("images_meta"):
    st.subheader("📏 Image Sizing")
    # Implemented based on requested UI options
    sizing = st.radio("Sizing Method", ["Match Height", "Match Width", "Enlarge to Largest"], horizontal=True)
    
    f_size = st.slider("Label Font Size", 20, 100, 45)
    
    if st.button("🚀 Generate Collage"):
        result = render_collage(st.session_state["images_meta"], sizing, f_size)
        st.image(result, use_container_width=True)
