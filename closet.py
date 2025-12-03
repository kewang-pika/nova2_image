"""
Asset Closet - Folder-based asset storage for Nova2 Image
Stores people, outfits, locations, and styles with AI-generated descriptions
"""

import os
import io
import json
import base64
from datetime import datetime
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_DESCRIBE = "gemini-2.5-flash-lite"

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# Closet directory
CLOSET_DIR = os.path.join(os.path.dirname(__file__), "closet")
CATEGORIES = ["people", "outfit", "location", "style"]


def init_closet():
    """Initialize closet folder structure."""
    for category in CATEGORIES:
        category_path = os.path.join(CLOSET_DIR, category)
        os.makedirs(category_path, exist_ok=True)
    print(f"[Closet] Initialized at {CLOSET_DIR}")


def describe_asset(image_bytes: bytes, category: str) -> str:
    """
    Generate AI description for an asset image.

    Args:
        image_bytes: Image data as bytes
        category: Asset category (people, outfit, location, style)

    Returns:
        AI-generated description string
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Category-specific prompts
        prompts = {
            "people": "Describe this person in detail. Include their appearance, facial features, hair, and any distinctive characteristics. Be concise (1-2 sentences).",
            "outfit": "Describe this clothing/outfit in detail. Include the type of garment, color, style, and any distinctive features. Be concise (1-2 sentences).",
            "location": "Describe this location/setting in detail. Include the environment, atmosphere, and key visual elements. Be concise (1-2 sentences).",
            "style": "Describe the visual style/aesthetic of this image. Include the mood, color palette, lighting, and artistic qualities. Be concise (1-2 sentences)."
        }

        prompt = prompts.get(category, "Describe this image in detail. Be concise (1-2 sentences).")

        print(f"[Closet] Generating description for {category}...")
        response = client.models.generate_content(
            model=MODEL_DESCRIBE,
            contents=[prompt, pil_image]
        )

        description = response.text.strip()
        print(f"[Closet] Description: {description}")
        return description

    except Exception as e:
        print(f"[Closet] Error describing asset: {e}")
        return f"A {category} asset"


def save_asset(name: str, category: str, image_bytes: bytes) -> dict:
    """
    Save an asset to the closet with auto-generated description.

    Args:
        name: Asset name (used as filename, no spaces recommended)
        category: Category (people, outfit, location, style)
        image_bytes: Image data as bytes

    Returns:
        dict with asset info: {name, category, description, image_path, created_at}
    """
    if category not in CATEGORIES:
        raise ValueError(f"Invalid category: {category}. Must be one of {CATEGORIES}")

    # Ensure closet exists
    init_closet()

    # Clean name (remove spaces, special chars)
    clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    # Generate description
    description = describe_asset(image_bytes, category)

    # Save image as JPG
    category_path = os.path.join(CLOSET_DIR, category)
    image_path = os.path.join(category_path, f"{clean_name}.jpg")

    # Convert to RGB and save as JPEG
    pil_image = Image.open(io.BytesIO(image_bytes))
    if pil_image.mode in ('RGBA', 'LA', 'P'):
        rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
        if pil_image.mode == 'P':
            pil_image = pil_image.convert('RGBA')
        rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
        pil_image = rgb_image
    else:
        pil_image = pil_image.convert('RGB')

    pil_image.save(image_path, 'JPEG', quality=95)

    # Save metadata as JSON
    metadata = {
        "name": clean_name,
        "category": category,
        "description": description,
        "created_at": datetime.now().isoformat()
    }

    json_path = os.path.join(category_path, f"{clean_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[Closet] Saved asset '{clean_name}' to {category}/")

    return {
        "name": clean_name,
        "category": category,
        "description": description,
        "image_path": image_path,
        "created_at": metadata["created_at"]
    }


def get_asset(name: str) -> dict:
    """
    Get a single asset by name (searches all categories).

    Args:
        name: Asset name to find

    Returns:
        dict with asset info including image_bytes, or None if not found
    """
    for category in CATEGORIES:
        category_path = os.path.join(CLOSET_DIR, category)
        json_path = os.path.join(category_path, f"{name}.json")
        image_path = os.path.join(category_path, f"{name}.jpg")

        if os.path.exists(json_path) and os.path.exists(image_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)

            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            return {
                "name": metadata["name"],
                "category": metadata["category"],
                "description": metadata["description"],
                "image_bytes": image_bytes,
                "image_path": image_path,
                "created_at": metadata.get("created_at", "")
            }

    return None


def get_assets(category: str) -> list:
    """
    Get all assets in a category.

    Args:
        category: Category to list (people, outfit, location, style)

    Returns:
        List of asset dicts
    """
    if category not in CATEGORIES:
        return []

    assets = []
    category_path = os.path.join(CLOSET_DIR, category)

    if not os.path.exists(category_path):
        return []

    for filename in os.listdir(category_path):
        if filename.endswith('.json'):
            name = filename[:-5]  # Remove .json
            asset = get_asset(name)
            if asset:
                assets.append(asset)

    # Sort by created_at (newest first)
    assets.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return assets


def get_all_assets() -> dict:
    """
    Get all assets grouped by category.

    Returns:
        dict with category keys and list of assets as values
    """
    init_closet()
    return {category: get_assets(category) for category in CATEGORIES}


def delete_asset(name: str, category: str = None) -> bool:
    """
    Delete an asset from the closet.

    Args:
        name: Asset name to delete
        category: Optional category (if known, faster lookup)

    Returns:
        True if deleted, False if not found
    """
    categories_to_check = [category] if category else CATEGORIES

    for cat in categories_to_check:
        if cat not in CATEGORIES:
            continue

        category_path = os.path.join(CLOSET_DIR, cat)
        json_path = os.path.join(category_path, f"{name}.json")
        image_path = os.path.join(category_path, f"{name}.jpg")

        if os.path.exists(json_path):
            os.remove(json_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            print(f"[Closet] Deleted asset '{name}' from {cat}/")
            return True

    return False


def get_asset_image_base64(name: str) -> str:
    """
    Get asset image as base64 string (for display in Streamlit).

    Args:
        name: Asset name

    Returns:
        Base64 encoded image string, or empty string if not found
    """
    asset = get_asset(name)
    if asset and "image_bytes" in asset:
        return base64.b64encode(asset["image_bytes"]).decode('utf-8')
    return ""


def list_all_asset_names() -> list:
    """
    Get list of all asset names (for @mention autocomplete).

    Returns:
        List of asset names
    """
    names = []
    for category in CATEGORIES:
        assets = get_assets(category)
        names.extend([a["name"] for a in assets])
    return names


# Initialize closet on module import
init_closet()
