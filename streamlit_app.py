"""
Nova2 Image Creator - Streamlit UI
Web interface for image editing using Gemini gemini-3-pro-image-preview
"""

import os
import io
import streamlit as st
from PIL import Image
from datetime import datetime

from image_creator import (
    rewrite_prompt,
    generate_image,
    save_image,
    image_to_base64,
    MODEL_IMAGE,
    OUTPUT_DIR,
)

# Page config
st.set_page_config(
    page_title="Nova2 Image Creator",
    page_icon="🎨",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("Nova2 Image Creator")
st.caption(f"Model: {MODEL_IMAGE}")

# Initialize session state
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None
if "final_prompt" not in st.session_state:
    st.session_state.final_prompt = None
if "image_description" not in st.session_state:
    st.session_state.image_description = None

# Sidebar options
with st.sidebar:
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

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload the image you want to edit"
    )

    if uploaded_file:
        image_bytes = uploaded_file.read()
        st.image(image_bytes, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("Result")
    result_placeholder = st.empty()

    if st.session_state.generated_image:
        result_placeholder.image(
            st.session_state.generated_image,
            caption="Generated Image",
            use_container_width=True
        )
    else:
        result_placeholder.info("Generated image will appear here")

# Prompt input
st.subheader("Editing Prompt")
prompt = st.text_area(
    "Describe how to edit the image",
    placeholder="e.g., 'Make it look like a painting', 'Add a sunset background', 'Change the outfit to casual wear'",
    height=100
)

# Show structured output (description + editing prompt) if available
if st.session_state.image_description:
    st.code(st.session_state.image_description, language=None)

# Generate button
if st.button("Generate", type="primary", disabled=not (uploaded_file and prompt)):
    if not uploaded_file:
        st.error("Please upload an image first")
    elif not prompt:
        st.error("Please enter a prompt")
    else:
        with st.spinner("Processing..."):
            try:
                # Reset file pointer
                uploaded_file.seek(0)
                image_bytes = uploaded_file.read()

                # Optionally enhance prompt
                final_prompt = prompt
                if use_ai_rewrite:
                    with st.status("Analyzing image and enhancing prompt..."):
                        rewrite_result = rewrite_prompt(prompt, image_bytes)
                        st.session_state.image_description = rewrite_result["structured_output"]
                        st.session_state.final_prompt = rewrite_result["rewritten_prompt"]
                        final_prompt = rewrite_result["rewritten_prompt"]
                        st.code(rewrite_result["structured_output"], language=None)
                else:
                    st.session_state.image_description = None
                    st.session_state.final_prompt = None

                # Generate image
                with st.status("Generating image..."):
                    result_bytes = generate_image(
                        image_bytes=image_bytes,
                        prompt=final_prompt,
                        aspect_ratio=aspect_ratio,
                        image_size=image_size
                    )

                # Store result
                st.session_state.generated_image = result_bytes
                if not use_ai_rewrite:
                    st.session_state.final_prompt = final_prompt

                # Display result
                result_placeholder.image(
                    result_bytes,
                    caption="Generated Image",
                    use_container_width=True
                )

                st.success("Image generated successfully!")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.generated_image = None

# Download button (only show if we have a generated image)
if st.session_state.generated_image:
    st.divider()

    col_dl1, col_dl2 = st.columns([2, 1])

    with col_dl1:
        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="Download Image",
            data=st.session_state.generated_image,
            file_name=f"nova2_generated_{timestamp}.jpg",
            mime="image/jpeg"
        )

    with col_dl2:
        # Save to server button
        if st.button("Save to Server"):
            try:
                output_path = save_image(st.session_state.generated_image)
                st.success(f"Saved: {output_path}")
            except Exception as e:
                st.error(f"Save error: {e}")

# Footer
st.divider()
st.caption("Nova2 Image Creator | Powered by Gemini gemini-3-pro-image-preview")
