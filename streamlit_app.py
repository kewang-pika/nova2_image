"""
Nova2 Image Creator - Streamlit UI
Web interface for image editing using Gemini gemini-3-pro-image-preview
With Asset Closet and @mention support
"""

import os
import io
import base64
import streamlit as st
from PIL import Image
from datetime import datetime

from image_creator import (
    rewrite_prompt,
    generate_image,
    generate_variation_prompts,
    generate_variations,
    save_image,
    image_to_base64,
    parse_mentions,
    MODEL_IMAGE,
    OUTPUT_DIR,
)

from closet import (
    save_asset,
    get_asset,
    get_assets,
    get_all_assets,
    delete_asset,
    list_all_asset_names,
    CATEGORIES,
    init_closet,
)

# Initialize closet
init_closet()

# Page config
st.set_page_config(
    page_title="Nova2 Image Creator",
    page_icon="🎨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
    .main-generate-btn > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
        padding: 0.6em 1em !important;
    }
    .asset-pill {
        display: inline-block;
        background-color: #e0e0e0;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 2px 4px;
        font-size: 0.9em;
    }
    .asset-thumbnail {
        border-radius: 8px;
        cursor: pointer;
    }
    .section-header {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 0.5em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = None
if "image_description" not in st.session_state:
    st.session_state.image_description = None
if "attached_assets" not in st.session_state:
    st.session_state.attached_assets = []
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""
if "detected_style" not in st.session_state:
    st.session_state.detected_style = None
if "used_assets" not in st.session_state:
    st.session_state.used_assets = []
# Variation generation state
if "variations" not in st.session_state:
    st.session_state.variations = []
if "variation_prompts" not in st.session_state:
    st.session_state.variation_prompts = []
if "variation_full_prompts" not in st.session_state:
    st.session_state.variation_full_prompts = []
if "original_selfie_bytes" not in st.session_state:
    st.session_state.original_selfie_bytes = None
if "selfie_description" not in st.session_state:
    st.session_state.selfie_description = None
if "generation_assets" not in st.session_state:
    st.session_state.generation_assets = []


def attach_asset(asset_name):
    """Add asset to attached list."""
    if asset_name not in st.session_state.attached_assets:
        st.session_state.attached_assets.append(asset_name)


def detach_asset(asset_name):
    """Remove asset from attached list."""
    if asset_name in st.session_state.attached_assets:
        st.session_state.attached_assets.remove(asset_name)


def get_asset_thumbnail(asset):
    """Get base64 thumbnail for display."""
    if asset and "image_bytes" in asset:
        return f"data:image/jpeg;base64,{base64.b64encode(asset['image_bytes']).decode()}"
    return None


# ===== SIDEBAR =====
with st.sidebar:
    # Settings Section
    st.header("Settings")

    aspect_ratio = st.selectbox(
        "Aspect Ratio",
        options=["9:16", "1:1", "16:9", "4:3", "3:4"],
        index=0
    )

    image_size = st.selectbox(
        "Image Size",
        options=["1K", "2K", "4K"],
        index=0,
        help="Higher resolution = more detail but slower"
    )

    use_ai_rewrite = st.checkbox(
        "AI Enhance Prompt",
        value=True,
        help="Use AI to analyze image and improve your prompt"
    )

    st.divider()

    # Asset Closet Section
    with st.expander("Asset Closet", expanded=False):
        # Category tabs
        tab_people, tab_outfit, tab_location, tab_style = st.tabs([
            "👤 People", "👔 Outfit", "📍 Location", "🎨 Style"
        ])

        category_tabs = {
            "people": tab_people,
            "outfit": tab_outfit,
            "location": tab_location,
            "style": tab_style
        }

        for category, tab in category_tabs.items():
            with tab:
                assets = get_assets(category)

                if assets:
                    # Display assets in 3-column grid
                    cols = st.columns(3)
                    for i, asset in enumerate(assets):
                        with cols[i % 3]:
                            thumb = get_asset_thumbnail(asset)
                            if thumb:
                                st.image(thumb, caption=asset['name'], use_container_width=True)

                            col_add, col_del = st.columns(2)
                            with col_add:
                                if st.button("➕", key=f"add_{category}_{asset['name']}", help="Attach"):
                                    attach_asset(asset['name'])
                                    st.rerun()
                            with col_del:
                                if st.button("🗑️", key=f"del_{category}_{asset['name']}", help="Delete"):
                                    delete_asset(asset['name'], category)
                                    st.rerun()
                else:
                    st.caption("No assets yet")

        st.divider()

        # Add New Asset Form
        st.markdown("**Add New Asset**")

        new_asset_name = st.text_input("Name", key="new_asset_name", placeholder="e.g., redDress")
        new_asset_category = st.selectbox("Category", options=CATEGORIES, key="new_asset_category")
        new_asset_file = st.file_uploader(
            "Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="new_asset_file"
        )

        if st.button("Add to Closet", key="add_to_closet"):
            if new_asset_name and new_asset_file:
                with st.spinner("Adding asset..."):
                    try:
                        asset_bytes = new_asset_file.read()
                        result = save_asset(new_asset_name, new_asset_category, asset_bytes)
                        st.success(f"Added '{result['name']}' to {new_asset_category}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please provide name and image")


# ===== MAIN CONTENT =====
st.title("Nova2 Image Creator")
st.caption(f"Model: {MODEL_IMAGE}")

# Two-column layout
col_left, col_right = st.columns([1, 1])

with col_left:
    # Upload Section
    st.markdown("**📷 Your Photo**")
    uploaded_file = st.file_uploader(
        "Upload base image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload the image you want to edit",
        label_visibility="collapsed"
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        # Small display thumbnail
        st.image(image_bytes, caption="Base Image", width=200)
        with st.expander("🔍 View Full Size"):
            st.image(image_bytes)  # Full resolution, no constraints

with col_right:
    # Attached Assets Section (moved to right column)
    st.markdown("**🏷️ Attached Assets**")

    if st.session_state.attached_assets:
        # Display attached assets as removable pills
        cols = st.columns(min(len(st.session_state.attached_assets) + 1, 4))
        for i, asset_name in enumerate(st.session_state.attached_assets):
            with cols[i % 4]:
                if st.button(f"{asset_name} ✕", key=f"remove_{asset_name}"):
                    detach_asset(asset_name)
                    st.rerun()

        # Show thumbnails of attached assets
        asset_cols = st.columns(min(len(st.session_state.attached_assets), 4))
        for i, asset_name in enumerate(st.session_state.attached_assets):
            asset = get_asset(asset_name)
            if asset:
                with asset_cols[i % 4]:
                    thumb = get_asset_thumbnail(asset)
                    if thumb:
                        st.image(thumb, caption=asset_name, width=80)
    else:
        st.caption("No assets attached. Type @ in prompt to see suggestions.")

# Prompt Input Section
st.markdown("**✏️ What do you want to create?**")

# Initialize pending insert in session state
if "pending_insert" not in st.session_state:
    st.session_state.pending_insert = None

# Apply pending insert BEFORE widget renders
if st.session_state.pending_insert:
    st.session_state.prompt_input = st.session_state.pending_insert
    st.session_state.pending_insert = None

prompt = st.text_area(
    "Editing prompt",
    placeholder="e.g., 'I'm wearing @redDress at @beach', 'Make it look like a painting'",
    height=100,
    label_visibility="collapsed",
    key="prompt_input"
)

# Check if user typed @ - show asset picker
all_asset_names = list_all_asset_names()
show_picker = prompt and prompt.endswith('@')

if show_picker and all_asset_names:
    st.markdown("**📎 Quick Insert Asset:**")

    # Group assets by category for better UX
    all_assets_grouped = get_all_assets()

    # Create columns for each category with assets
    cols = st.columns(4)
    col_idx = 0

    for category in CATEGORIES:
        assets = all_assets_grouped.get(category, [])
        if assets:
            with cols[col_idx % 4]:
                st.caption(f"**{category.title()}**")
                for asset in assets[:5]:  # Limit to 5 per category
                    if st.button(
                        f"@{asset['name']}",
                        key=f"insert_{category}_{asset['name']}",
                        use_container_width=True
                    ):
                        # Set pending insert (replace trailing @)
                        st.session_state.pending_insert = prompt[:-1] + f"@{asset['name']} "
                        st.rerun()
            col_idx += 1

elif all_asset_names:
    # Show assets organized by category
    st.caption("💡 Type `@` to insert an asset:")
    all_assets_grouped = get_all_assets()
    category_icons = {"people": "👤", "outfit": "👔", "location": "📍", "style": "🎨"}

    for category in CATEGORIES:
        assets = all_assets_grouped.get(category, [])
        if assets:
            names = " ".join([f"`@{a['name']}`" for a in assets])
            st.caption(f"{category_icons.get(category, '')} {category.title()}: {names}")

# Auto-attach mentioned assets from prompt
if prompt:
    mentions = parse_mentions(prompt)
    for mention in mentions:
        if mention not in st.session_state.attached_assets:
            # Check if asset exists
            if get_asset(mention):
                attach_asset(mention)

# Enhanced Prompt Display (collapsible)
if st.session_state.image_description:
    # Show detected style as a badge
    if st.session_state.detected_style:
        st.markdown(f"**🎨 Style:** `{st.session_state.detected_style}`")
    with st.expander("📝 Enhanced Prompt", expanded=False):
        st.code(st.session_state.image_description, language=None)

# Generate Button
st.markdown('<div class="main-generate-btn">', unsafe_allow_html=True)
generate_clicked = st.button(
    "🎨 Generate Image",
    type="primary",
    use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

if generate_clicked:
    if not uploaded_file:
        st.error("Please upload an image first")
    elif not prompt:
        st.error("Please enter a prompt")
    else:
        with st.spinner("Processing..."):
            try:
                # Reset file pointer and read image
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()

                # Resolve attached assets
                mentioned_assets = []
                for asset_name in st.session_state.attached_assets:
                    asset = get_asset(asset_name)
                    if asset:
                        mentioned_assets.append({
                            "name": asset["name"],
                            "image_bytes": asset["image_bytes"],
                            "description": asset["description"]
                        })

                # Optionally enhance prompt
                final_prompt = prompt
                additional_images = []

                if use_ai_rewrite:
                    with st.status("Analyzing image and enhancing prompt..."):
                        rewrite_result = rewrite_prompt(prompt, image_bytes, mentioned_assets)
                        st.session_state.image_description = rewrite_result["structured_output"]
                        st.session_state.final_prompt = rewrite_result["rewritten_prompt"]
                        st.session_state.detected_style = rewrite_result.get("style_name", "Editorial")
                        final_prompt = rewrite_result["rewritten_prompt"]
                        # Get additional images (excluding base)
                        if len(rewrite_result.get('all_images', [])) > 1:
                            additional_images = rewrite_result['all_images'][1:]
                        st.code(rewrite_result["structured_output"], language=None)
                else:
                    st.session_state.image_description = None
                    st.session_state.final_prompt = None
                    st.session_state.detected_style = None
                    # Still pass additional images if not using rewrite
                    additional_images = [a['image_bytes'] for a in mentioned_assets]

                # Generate image
                with st.status("Generating image..."):
                    result_bytes = generate_image(
                        image_bytes=image_bytes,
                        prompt=final_prompt,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        additional_images=additional_images
                    )

                # Store result
                st.session_state.generated_image = result_bytes
                st.session_state.used_assets = list(st.session_state.attached_assets)  # Copy the list
                if not use_ai_rewrite:
                    st.session_state.final_prompt = final_prompt

                # Store data needed for variations
                st.session_state.original_selfie_bytes = image_bytes
                st.session_state.generation_assets = mentioned_assets
                # Extract selfie description from rewrite result
                if use_ai_rewrite and rewrite_result.get("description"):
                    # Parse first line: "1. I/myself: [description] (attached image 1)"
                    desc_lines = rewrite_result["description"].split("\n")
                    if desc_lines:
                        first_line = desc_lines[0]
                        # Remove "1. I/myself: " prefix and " (attached image 1)" suffix
                        selfie_desc = first_line.replace("1. I/myself: ", "")
                        if " (attached image" in selfie_desc:
                            selfie_desc = selfie_desc.split(" (attached image")[0]
                        st.session_state.selfie_description = selfie_desc
                    else:
                        st.session_state.selfie_description = None
                else:
                    st.session_state.selfie_description = None

                # Clear previous variations when generating new first image
                st.session_state.variations = []
                st.session_state.variation_prompts = []
                st.session_state.variation_full_prompts = []

                st.success("Image generated successfully!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.generated_image = None

# Result Section (below Generate button)
if st.session_state.generated_image:
    st.divider()
    st.markdown("**🖼️ Result**")

    # Show detected style
    if st.session_state.detected_style:
        st.markdown(f"**🎨 Style:** `{st.session_state.detected_style}`")

    # Show used assets
    if st.session_state.used_assets:
        st.caption(f"**Assets used:** {', '.join(st.session_state.used_assets)}")
        # Show thumbnails of used assets
        used_cols = st.columns(min(len(st.session_state.used_assets), 6))
        for i, asset_name in enumerate(st.session_state.used_assets):
            asset = get_asset(asset_name)
            if asset:
                with used_cols[i % 6]:
                    thumb = get_asset_thumbnail(asset)
                    if thumb:
                        st.image(thumb, caption=asset_name, width=60)

    # Small display thumbnail
    st.image(
        st.session_state.generated_image,
        caption="Generated Image",
        width=300
    )
    with st.expander("🔍 View Full Size"):
        st.image(st.session_state.generated_image)  # Full resolution, no constraints

# Download Section
if st.session_state.generated_image:
    col_dl1, col_dl2 = st.columns([2, 1])

    with col_dl1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="💾 Download Image",
            data=st.session_state.generated_image,
            file_name=f"nova2_generated_{timestamp}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    with col_dl2:
        if st.button("Save to Server", use_container_width=True):
            try:
                output_path = save_image(st.session_state.generated_image)
                st.success(f"Saved: {output_path}")
            except Exception as e:
                st.error(f"Save error: {e}")

# Variations Section (only show after first image generated with AI enhance)
if st.session_state.generated_image and st.session_state.image_description and st.session_state.selfie_description:
    st.divider()

    if st.button("🎲 Generate 4 Variations", use_container_width=True):
        with st.spinner("Generating variations..."):
            try:
                # Step 1: Get variation prompts using structured prompt + generated image
                with st.status("Creating variation ideas...") as status:
                    var_prompts = generate_variation_prompts(
                        structured_prompt=st.session_state.image_description,
                        image_bytes=st.session_state.generated_image
                    )
                    for i, p in enumerate(var_prompts, 1):
                        st.write(f"{i}. {p}")
                    status.update(label="Variation ideas created!", state="complete")

                # Step 2: Generate all 4 variations IN PARALLEL
                with st.status("Generating 4 images in parallel...") as status:
                    result = generate_variations(
                        original_selfie_bytes=st.session_state.original_selfie_bytes,
                        generated_image_bytes=st.session_state.generated_image,
                        selfie_description=st.session_state.selfie_description,
                        variation_prompts=var_prompts,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                        additional_assets=st.session_state.generation_assets
                    )
                    status.update(label="All 4 variations generated!", state="complete")

                # Store images and full prompts separately
                st.session_state.variations = result["images"]
                st.session_state.variation_prompts = var_prompts
                st.session_state.variation_full_prompts = result["full_prompts"]
                st.success("Generated 4 variations!")

            except Exception as e:
                st.error(f"Error generating variations: {str(e)}")

# Display variations in 2x2 grid
if st.session_state.get("variations") and len(st.session_state.variations) > 0:
    st.divider()
    st.markdown("**🎲 Variations**")

    # 2x2 grid
    row1_cols = st.columns(2)
    row2_cols = st.columns(2)
    all_cols = row1_cols + row2_cols

    for i, img_bytes in enumerate(st.session_state.variations):
        if i < len(all_cols) and img_bytes:
            with all_cols[i]:
                st.image(img_bytes, width=200)
                # Show full structured prompt in expander
                full_prompt = st.session_state.get("variation_full_prompts", [])[i] if i < len(st.session_state.get("variation_full_prompts", [])) else ""
                with st.expander(f"📝 V{i+1} Full Prompt"):
                    st.code(full_prompt, language=None)
                with st.expander(f"🔍 V{i+1} Full Size"):
                    st.image(img_bytes)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    f"💾 Download V{i+1}",
                    data=img_bytes,
                    file_name=f"nova2_variation_{i+1}_{timestamp}.jpg",
                    mime="image/jpeg",
                    key=f"dl_var_{i}",
                    use_container_width=True
                )

# Footer
st.divider()
st.caption("Nova2 Image Creator | Powered by Gemini gemini-3-pro-image-preview")
