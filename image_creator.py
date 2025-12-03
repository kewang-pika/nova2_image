"""
Nova2 Image Creator
Standalone image editing module using Gemini gemini-3-pro-image-preview
"""

import os
import io
import json
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import from agents module
from agents import (
    StyleAgent,
    PromptAgent,
    VariationsAgent,
    STYLE_PRESETS,
    STYLE_CATEGORIES,
    DEFAULT_STYLE_SYSTEM_PROMPT,
    image_to_base64,
    base64_to_image,
)

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_IMAGE = "gemini-3-pro-image-preview"
MODEL_REWRITE = "gemini-2.5-flash"

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== CONFIG FILE ==============
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "agent_config.json")


def load_agent_config() -> dict:
    """Load agent configuration from JSON file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Config] Error loading config: {e}")
    return {}


def save_agent_config(config: dict):
    """Save agent configuration to JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"[Config] Saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"[Config] Error saving config: {e}")


# ============== AGENTS ==============
# Load saved config
_config = load_agent_config()
_saved_style_prompt = _config.get("style_agent_system_prompt")

# Initialize agents with saved config
style_agent = StyleAgent(system_prompt=_saved_style_prompt)
prompt_agent = PromptAgent(style_agent=style_agent)
variations_agent = VariationsAgent()


# ============== SYSTEM PROMPT MANAGEMENT ==============
def get_style_system_prompt() -> str:
    """Get the current StyleAgent system prompt."""
    return style_agent.get_system_prompt()


def set_style_system_prompt(prompt: str):
    """Set the StyleAgent system prompt (in memory only)."""
    style_agent.set_system_prompt(prompt)


def save_style_system_prompt(prompt: str):
    """Save the StyleAgent system prompt to config file and apply it."""
    style_agent.set_system_prompt(prompt)
    config = load_agent_config()
    config["style_agent_system_prompt"] = prompt
    save_agent_config(config)
    return True


def reset_style_system_prompt():
    """Reset StyleAgent system prompt to default."""
    style_agent.set_system_prompt(DEFAULT_STYLE_SYSTEM_PROMPT)
    config = load_agent_config()
    if "style_agent_system_prompt" in config:
        del config["style_agent_system_prompt"]
    save_agent_config(config)
    return DEFAULT_STYLE_SYSTEM_PROMPT


# ============== CORE GENERATION ==============
def generate_image(
    image_bytes: bytes,
    prompt: str,
    aspect_ratio: str = "9:16",
    image_size: str = "1K",
    additional_images: list = None
) -> bytes:
    """
    Generate edited image using Gemini gemini-3-pro-image-preview.

    Args:
        image_bytes: Input image as bytes (base image)
        prompt: Editing prompt
        aspect_ratio: Output aspect ratio (1:1, 9:16, 16:9, 4:3, 3:4)
        image_size: Resolution (1K, 2K, 4K) - only for gemini-3-pro-image-preview
        additional_images: Optional list of additional image bytes (from @mentions)

    Returns:
        Generated image as JPEG bytes
    """
    if additional_images is None:
        additional_images = []

    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Convert additional images
        additional_pil = []
        for img_bytes in additional_images:
            if img_bytes:
                additional_pil.append(Image.open(io.BytesIO(img_bytes)))

        total_images = 1 + len(additional_pil)

        print(f"\n[Generate] === IMAGE EDITING ===")
        print(f"[Generate] Model: {MODEL_IMAGE}")
        print(f"[Generate] Prompt: {prompt}")
        print(f"[Generate] Input size: {pil_image.size} ({pil_image.mode})")
        print(f"[Generate] Total images: {total_images}")
        print(f"[Generate] Aspect ratio: {aspect_ratio}")
        print(f"[Generate] Image size: {image_size}")
        print(f"[Generate] ====================\n")

        # Build contents: [prompt, base_image, additional_images...]
        contents = [prompt, pil_image] + additional_pil

        # Build ImageConfig with aspect ratio and image size
        img_config = types.ImageConfig(
            aspectRatio=aspect_ratio,
            imageSize=image_size
        )

        # Call Gemini API
        response = client.models.generate_content(
            model=MODEL_IMAGE,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=img_config
            )
        )

        # Extract image from response
        if not response.candidates or not response.candidates[0].content.parts:
            raise ValueError("No image generated in response")

        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Get raw image data
                image_data = part.inline_data.data
                result_image = Image.open(io.BytesIO(image_data))

                # Convert to RGB (JPEG doesn't support alpha)
                if result_image.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', result_image.size, (255, 255, 255))
                    if result_image.mode == 'P':
                        result_image = result_image.convert('RGBA')
                    rgb_img.paste(
                        result_image,
                        mask=result_image.split()[-1] if result_image.mode in ('RGBA', 'LA') else None
                    )
                else:
                    rgb_img = result_image.convert('RGB')

                # Save as high-quality JPEG
                jpg_buffer = io.BytesIO()
                rgb_img.save(jpg_buffer, format='JPEG', quality=95, optimize=True)
                jpg_bytes = jpg_buffer.getvalue()

                print(f"[Generate] Success! Output size: {rgb_img.size}")
                return jpg_bytes

        raise ValueError("No image data found in response")

    except Exception as e:
        print(f"[Generate] Error: {e}")
        raise


# Wire up variations agent with generate_image function
variations_agent.generate_fn = generate_image


# ============== FILE I/O ==============
def save_image(image_bytes: bytes, output_path: str = None) -> str:
    """
    Save image bytes to disk as JPEG.

    Args:
        image_bytes: Image data as bytes
        output_path: Optional output path. If None, generates timestamped filename.

    Returns:
        Path to saved file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"generated_{timestamp}.jpg")

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(image_bytes)

    print(f"[Save] Image saved to: {output_path}")
    return output_path


def load_image(image_path: str) -> bytes:
    """Load image from path and return as bytes."""
    with open(image_path, 'rb') as f:
        return f.read()


# ============== CONVENIENCE WRAPPERS ==============
# These functions wrap agent methods for backward compatibility with streamlit_app.py

def parse_mentions(prompt: str) -> list:
    """Parse @mentions from a prompt. Wrapper for PromptAgent.parse_mentions()."""
    return prompt_agent.parse_mentions(prompt)


def remove_mentions_from_prompt(prompt: str) -> str:
    """Remove @mentions from prompt. Wrapper for PromptAgent.remove_mentions()."""
    return prompt_agent.remove_mentions(prompt)


def detect_style(prompt: str) -> dict:
    """Detect style from prompt. Wrapper for StyleAgent.detect()."""
    return style_agent.detect(prompt)


def rewrite_prompt(original_prompt: str, image_bytes: bytes, mentioned_assets: list = None, aesthetic: dict = None) -> dict:
    """Rewrite and enhance prompt. Wrapper for PromptAgent.rewrite().

    Args:
        original_prompt: User's original prompt
        image_bytes: Base image bytes
        mentioned_assets: List of asset dicts
        aesthetic: Optional dict with visual_style and avoids keys
            Example: {"visual_style": "35mm film grain", "avoids": ["neon", "cartoon"]}
    """
    return prompt_agent.rewrite(original_prompt, image_bytes, mentioned_assets, aesthetic)


def generate_variation_prompts(structured_prompt: str, image_bytes: bytes) -> list:
    """Generate 4 variation prompts. Wrapper for VariationsAgent.create_prompts()."""
    return variations_agent.create_prompts(structured_prompt, image_bytes)


def generate_variations(
    original_selfie_bytes: bytes,
    generated_image_bytes: bytes,
    selfie_description: str,
    variation_prompts: list,
    aspect_ratio: str = "9:16",
    image_size: str = "1K",
    additional_assets: list = None
) -> dict:
    """Generate 4 variations in parallel. Returns dict with 'images' and 'full_prompts'."""
    return variations_agent.generate(
        original_selfie_bytes,
        generated_image_bytes,
        selfie_description,
        variation_prompts,
        aspect_ratio,
        image_size,
        additional_assets
    )


# ============== CLI CLOSET COMMANDS ==============
def cli_closet_add():
    """CLI: Add asset to closet."""
    from closet import save_asset, CATEGORIES

    name = input("Asset name: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return

    print(f"Categories: {', '.join(CATEGORIES)}")
    category = input("Category: ").strip().lower()
    if category not in CATEGORIES:
        print(f"Error: Invalid category. Must be one of: {', '.join(CATEGORIES)}")
        return

    image_path = input("Image path: ").strip()
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    try:
        image_bytes = load_image(image_path)
        print("[Closet] Adding asset...")
        result = save_asset(name, category, image_bytes)
        print(f"[Closet] Added '{result['name']}' to {category}/")
        print(f"[Closet] Description: {result['description']}")
    except Exception as e:
        print(f"Error: {e}")


def cli_closet_list(category: str = None):
    """CLI: List assets in closet."""
    from closet import get_assets, get_all_assets, CATEGORIES

    print("\n=== Asset Closet ===")

    if category:
        if category not in CATEGORIES:
            print(f"Error: Invalid category. Must be one of: {', '.join(CATEGORIES)}")
            return
        assets = get_assets(category)
        print(f"{category.title()} ({len(assets)}):")
        for asset in assets:
            print(f"  - {asset['name']}: {asset['description'][:50]}...")
    else:
        all_assets = get_all_assets()
        for cat in CATEGORIES:
            assets = all_assets.get(cat, [])
            names = [a['name'] for a in assets]
            print(f"{cat.title()} ({len(assets)}): {', '.join(names) if names else '-'}")
    print()


def cli_closet_delete():
    """CLI: Delete asset from closet."""
    from closet import delete_asset, CATEGORIES

    name = input("Asset name to delete: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        return

    category = input(f"Category ({', '.join(CATEGORIES)}) [search all]: ").strip().lower()
    if category and category not in CATEGORIES:
        print(f"Error: Invalid category")
        return

    if delete_asset(name, category if category else None):
        print(f"[Closet] Deleted '{name}'")
    else:
        print(f"[Closet] Asset '{name}' not found")


# ============== CLI ==============
def run_cli():
    """Interactive CLI mode for image editing with closet support."""
    from closet import get_asset, list_all_asset_names

    print("\n" + "="*50)
    print("Nova2 Image Creator - CLI Mode")
    print("="*50)
    print(f"Model: {MODEL_IMAGE}")
    print("\nCommands:")
    print("  closet add    - Add asset to closet")
    print("  closet list   - List all assets")
    print("  closet delete - Delete an asset")
    print("  generate      - Generate image (default)")
    print("  quit          - Exit")
    print()

    while True:
        cmd = input("> ").strip().lower()

        if cmd == 'quit' or cmd == 'q':
            break
        elif cmd == 'closet add':
            cli_closet_add()
            continue
        elif cmd.startswith('closet list'):
            parts = cmd.split()
            category = parts[2] if len(parts) > 2 else None
            cli_closet_list(category)
            continue
        elif cmd == 'closet delete':
            cli_closet_delete()
            continue
        elif cmd == 'closet':
            cli_closet_list()
            continue
        elif cmd != 'generate' and cmd != '':
            print(f"Unknown command: {cmd}")
            continue

        # Generate flow
        image_path = input("Image path (or 'back'): ").strip()
        if image_path.lower() == 'back':
            continue

        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            continue

        # Load image
        try:
            image_bytes = load_image(image_path)
            print(f"Loaded image: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        # Show available assets
        asset_names = list_all_asset_names()
        if asset_names:
            print(f"\nAvailable assets: {', '.join(['@' + n for n in asset_names])}")

        # Get prompt
        prompt = input("Editing prompt: ").strip()
        if not prompt:
            print("Error: Prompt cannot be empty")
            continue

        # Parse @mentions and resolve assets
        mentions = parse_mentions(prompt)
        mentioned_assets = []

        if mentions:
            print(f"\n[Mention] Found mentions: {', '.join(mentions)}")
            for mention in mentions:
                asset = get_asset(mention)
                if asset:
                    print(f"[Mention] Resolved: {mention} ({asset['category']})")
                    mentioned_assets.append({
                        "name": asset["name"],
                        "image_bytes": asset["image_bytes"],
                        "description": asset["description"]
                    })
                else:
                    print(f"[Mention] Warning: Asset '{mention}' not found in closet")

        # Ask about AI rewrite (default: yes)
        use_rewrite = input("Use AI to enhance prompt? (y/n) [y]: ").strip().lower()
        additional_images = []

        if use_rewrite != 'n':
            result = rewrite_prompt(prompt, image_bytes, mentioned_assets)
            print(f"\n{result['structured_output']}\n")
            prompt = result['rewritten_prompt']
            # Get additional images from rewrite result (excluding base image)
            if len(result.get('all_images', [])) > 1:
                additional_images = result['all_images'][1:]
        else:
            # Still need to pass additional images if not using rewrite
            additional_images = [a['image_bytes'] for a in mentioned_assets]

        # Get aspect ratio
        aspect = input("Aspect ratio (9:16, 1:1, 16:9, 4:3, 3:4) [9:16]: ").strip()
        if not aspect:
            aspect = "9:16"

        # Get image size
        size = input("Image size (1K, 2K, 4K) [1K]: ").strip().upper()
        if size not in ['1K', '2K', '4K']:
            size = "1K"

        # Generate
        print("\nGenerating...")
        try:
            result_bytes = generate_image(
                image_bytes, prompt, aspect, size,
                additional_images=additional_images
            )
            output_path = save_image(result_bytes)
            print(f"\nDone! Image saved to: {output_path}\n")
        except Exception as e:
            print(f"Error generating image: {e}\n")


if __name__ == "__main__":
    run_cli()
