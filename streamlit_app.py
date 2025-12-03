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
    get_style_system_prompt,
    save_style_system_prompt,
    reset_style_system_prompt,
    DEFAULT_STYLE_SYSTEM_PROMPT,
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

from history import (
    save_generation,
    update_generation_variations,
    get_generation,
    list_generations,
    delete_generation,
    verify_password,
    get_history_stats,
    HISTORY_PASSWORD,
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
# History state
if "current_gen_id" not in st.session_state:
    st.session_state.current_gen_id = None
if "history_authenticated" not in st.session_state:
    st.session_state.history_authenticated = False
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "selected_history_item" not in st.session_state:
    st.session_state.selected_history_item = None


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

    # Aesthetic Preferences Section
    with st.expander("🎨 Aesthetic Preferences (JSON)", expanded=False):
        st.caption("Paste JSON to influence style detection. Leave empty to use auto-detection.")

        # Example JSON
        example_json = '''{
  "aesthetic": {
    "visual_style": "Soft editorial, 35mm film grain, muted colors, natural light, minimalist",
    "avoids": ["neon", "high contrast", "heavy makeup", "3d render", "cartoon"]
  }
}'''

        # Initialize session state
        if "aesthetic_json" not in st.session_state:
            st.session_state.aesthetic_json = ""

        aesthetic_input = st.text_area(
            "Aesthetic JSON",
            value=st.session_state.aesthetic_json,
            height=150,
            key="aesthetic_json_input",
            placeholder=example_json,
            label_visibility="collapsed"
        )

        # Update session state
        st.session_state.aesthetic_json = aesthetic_input

        # Parse and validate
        aesthetic_dict = None
        if aesthetic_input.strip():
            try:
                import json
                parsed = json.loads(aesthetic_input)
                if "aesthetic" in parsed:
                    aesthetic_dict = parsed["aesthetic"]
                    st.success(f"✓ Valid: {aesthetic_dict.get('visual_style', '')[:40]}...")
                else:
                    st.warning("JSON should have 'aesthetic' key")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

        st.caption("**Example:**")
        st.code(example_json, language="json")

    st.divider()

    # StyleAgent System Prompt Editor
    with st.expander("⚙️ Style Detection Prompt", expanded=False):
        st.caption("Edit the system prompt used by StyleAgent to detect image styles.")

        # Initialize session state for the text area
        if "style_prompt_editor" not in st.session_state:
            st.session_state.style_prompt_editor = get_style_system_prompt()

        # Check if current prompt differs from default
        current_prompt = get_style_system_prompt()
        is_custom = current_prompt != DEFAULT_STYLE_SYSTEM_PROMPT

        if is_custom:
            st.info("📝 Using custom system prompt")

        # Text area for editing
        edited_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.style_prompt_editor,
            height=300,
            key="style_prompt_text",
            help="Use {prompt} as placeholder for the user's editing prompt",
            label_visibility="collapsed"
        )

        # Update session state when text changes
        st.session_state.style_prompt_editor = edited_prompt

        # Buttons row
        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button("💾 Save", key="save_style_prompt", use_container_width=True):
                if "{prompt}" not in edited_prompt:
                    st.error("Prompt must contain {prompt} placeholder")
                else:
                    save_style_system_prompt(edited_prompt)
                    st.success("Saved!")

        with col_reset:
            if st.button("🔄 Reset", key="reset_style_prompt", use_container_width=True):
                reset_style_system_prompt()
                st.session_state.style_prompt_editor = DEFAULT_STYLE_SYSTEM_PROMPT
                st.success("Reset to default!")
                st.rerun()

        st.caption("💡 Tip: The prompt must include `{prompt}` placeholder where the user's editing prompt will be inserted.")

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

                # Parse aesthetic JSON if provided
                aesthetic_prefs = None
                if st.session_state.aesthetic_json.strip():
                    try:
                        import json
                        parsed = json.loads(st.session_state.aesthetic_json)
                        if "aesthetic" in parsed:
                            aesthetic_prefs = parsed["aesthetic"]
                    except:
                        pass  # Ignore invalid JSON

                if use_ai_rewrite:
                    with st.status("Analyzing image and enhancing prompt..."):
                        rewrite_result = rewrite_prompt(prompt, image_bytes, mentioned_assets, aesthetic_prefs)
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

                # Save to history
                gen_id = save_generation(
                    original_prompt=prompt,
                    editing_prompt=st.session_state.final_prompt or prompt,
                    structured_prompt=st.session_state.image_description or prompt,
                    style_name=st.session_state.detected_style or "None",
                    main_image_bytes=result_bytes,
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                    aesthetic=aesthetic_prefs
                )
                st.session_state.current_gen_id = gen_id

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

                # Update history with variations
                if st.session_state.current_gen_id:
                    update_generation_variations(
                        gen_id=st.session_state.current_gen_id,
                        variation_images=result["images"],
                        variation_prompts=var_prompts,
                        variation_full_prompts=result["full_prompts"]
                    )

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

# ===== HISTORY GALLERY =====
st.divider()
st.markdown("### 📚 Generation History")

# History stats
stats = get_history_stats()
st.caption(f"Total: {stats['total']} generations | {stats['with_variations']} with variations")

# Password protection
if not st.session_state.history_authenticated:
    col_pwd, col_btn = st.columns([3, 1])
    with col_pwd:
        password_input = st.text_input(
            "Enter password to view history",
            type="password",
            key="history_password",
            placeholder="Password required"
        )
    with col_btn:
        if st.button("🔓 Unlock", key="unlock_history"):
            if verify_password(password_input):
                st.session_state.history_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")
else:
    # Show history gallery
    if st.button("🔒 Lock History", key="lock_history"):
        st.session_state.history_authenticated = False
        st.session_state.selected_history_item = None
        st.rerun()

    # Load history items
    history_items = list_generations(limit=30)

    if not history_items:
        st.info("No generations in history yet. Generate some images to see them here!")
    else:
        # View mode: Grid or Detail
        if st.session_state.selected_history_item:
            # Detail view
            item = get_generation(st.session_state.selected_history_item)
            if item:
                if st.button("← Back to Gallery", key="back_to_gallery"):
                    st.session_state.selected_history_item = None
                    st.rerun()

                st.markdown(f"**Generation:** `{item['id']}`")
                st.caption(f"Created: {item['timestamp'][:19].replace('T', ' ')}")

                # Main image and info
                col_img, col_info = st.columns([1, 1])

                with col_img:
                    st.markdown("**🖼️ Main Image**")
                    st.image(item["main_image_bytes"], use_container_width=True)

                with col_info:
                    st.markdown(f"**🎨 Style:** `{item.get('style_name', 'N/A')}`")
                    st.markdown(f"**📐 Aspect:** `{item.get('aspect_ratio', 'N/A')}` | **Size:** `{item.get('image_size', 'N/A')}`")

                    st.markdown("**💬 Original Prompt:**")
                    st.code(item.get("original_prompt", "N/A"), language=None)

                # Structured prompt
                with st.expander("📝 Full Structured Prompt", expanded=False):
                    st.code(item.get("structured_prompt", "N/A"), language=None)

                # Aesthetic preferences (if used)
                if item.get("aesthetic"):
                    with st.expander("🎨 Aesthetic Preferences", expanded=False):
                        import json
                        st.json(item.get("aesthetic"))

                # Variations
                if item.get("variation_image_bytes"):
                    st.markdown("**🎲 Variations**")

                    var_cols = st.columns(4)
                    for i, var_bytes in enumerate(item["variation_image_bytes"]):
                        with var_cols[i % 4]:
                            st.image(var_bytes, caption=f"V{i+1}", use_container_width=True)

                            # Show variation prompt
                            var_prompt = item.get("variation_prompts", [])[i] if i < len(item.get("variation_prompts", [])) else ""
                            if var_prompt:
                                st.caption(f"_{var_prompt[:60]}..._" if len(var_prompt) > 60 else f"_{var_prompt}_")

                    # Full variation prompts
                    with st.expander("📝 Variation Full Prompts", expanded=False):
                        for i, fp in enumerate(item.get("variation_full_prompts", [])):
                            st.markdown(f"**V{i+1}:**")
                            st.code(fp, language=None)

                # Delete button
                st.divider()
                if st.button("🗑️ Delete This Generation", key="delete_gen", type="secondary"):
                    delete_generation(item["id"])
                    st.session_state.selected_history_item = None
                    st.success("Deleted!")
                    st.rerun()

            else:
                st.error("Generation not found")
                st.session_state.selected_history_item = None

        else:
            # Grid view
            st.markdown("**Click on an image to view details**")

            # Display in 4-column grid
            cols_per_row = 4
            for row_start in range(0, len(history_items), cols_per_row):
                cols = st.columns(cols_per_row)
                for i, col in enumerate(cols):
                    idx = row_start + i
                    if idx < len(history_items):
                        item = history_items[idx]
                        with col:
                            # Thumbnail
                            if item.get("main_image_bytes"):
                                st.image(item["main_image_bytes"], use_container_width=True)

                            # Info
                            timestamp_short = item["timestamp"][:10]
                            has_vars = "🎲" if item.get("variations") else ""
                            st.caption(f"{timestamp_short} {has_vars}")
                            st.caption(f"_{item.get('original_prompt', '')[:30]}..._" if len(item.get('original_prompt', '')) > 30 else f"_{item.get('original_prompt', '')}_")

                            if st.button("View", key=f"view_{item['id']}", use_container_width=True):
                                st.session_state.selected_history_item = item["id"]
                                st.rerun()

# Footer
st.divider()
st.caption("Nova2 Image Creator | Powered by Gemini gemini-3-pro-image-preview")
