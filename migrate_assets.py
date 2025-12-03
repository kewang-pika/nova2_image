"""
Migration script: Transfer assets from gemini_image_app SQLite to nova2_image closet folder
"""

import os
import sqlite3
import base64
import json
from datetime import datetime
from PIL import Image
import io
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Paths
SOURCE_DB = "/mnt/nfs/ke/gemini_image_app/backend/asset_closet.db"
TARGET_CLOSET = "/mnt/nfs/ke/nova2_image/closet"

# Categories
CATEGORIES = ["people", "outfit", "location", "style"]

# Initialize Gemini for descriptions
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
MODEL_DESCRIBE = "gemini-2.5-flash-lite"


def init_closet():
    """Create closet folder structure."""
    for category in CATEGORIES:
        category_path = os.path.join(TARGET_CLOSET, category)
        os.makedirs(category_path, exist_ok=True)
    print(f"[Migration] Initialized closet at {TARGET_CLOSET}")


def describe_asset(image_bytes: bytes, category: str) -> str:
    """Generate AI description for an asset."""
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))

        prompts = {
            "people": "Describe this person in detail. Include their appearance, facial features, hair, and any distinctive characteristics. Be concise (1-2 sentences).",
            "outfit": "Describe this clothing/outfit in detail. Include the type of garment, color, style, and any distinctive features. Be concise (1-2 sentences).",
            "location": "Describe this location/setting in detail. Include the environment, atmosphere, and key visual elements. Be concise (1-2 sentences).",
            "style": "Describe the visual style/aesthetic of this image. Include the mood, color palette, lighting, and artistic qualities. Be concise (1-2 sentences)."
        }

        prompt = prompts.get(category, "Describe this image in detail. Be concise (1-2 sentences).")

        response = client.models.generate_content(
            model=MODEL_DESCRIBE,
            contents=[prompt, pil_image]
        )

        return response.text.strip()

    except Exception as e:
        print(f"    [Describe] Error: {e}")
        return f"A {category} asset"


def migrate_assets():
    """Migrate all assets from SQLite to folder-based storage."""
    init_closet()

    # Connect to source database
    conn = sqlite3.connect(SOURCE_DB)
    cursor = conn.cursor()

    # Get all assets
    cursor.execute("SELECT name, category, image_base64, created_at FROM assets")
    assets = cursor.fetchall()

    print(f"[Migration] Found {len(assets)} assets to migrate")

    migrated = 0
    errors = []

    for name, category, image_base64, created_at in assets:
        try:
            print(f"[Migration] Processing: {name} ({category})...")

            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)

            # Open and convert to RGB JPEG
            pil_image = Image.open(io.BytesIO(image_bytes))
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
                rgb_image.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                pil_image = rgb_image
            else:
                pil_image = pil_image.convert('RGB')

            # Save image
            category_path = os.path.join(TARGET_CLOSET, category)
            image_path = os.path.join(category_path, f"{name}.jpg")
            pil_image.save(image_path, 'JPEG', quality=95)

            # Generate description using AI
            print(f"  [Describe] Generating description...")
            description = describe_asset(image_bytes, category)
            print(f"  [Describe] {description[:60]}...")

            # Save metadata
            metadata = {
                "name": name,
                "category": category,
                "description": description,
                "created_at": created_at or datetime.now().isoformat()
            }

            json_path = os.path.join(category_path, f"{name}.json")
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"  ✓ Saved: {image_path}")
            migrated += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors.append((name, str(e)))

    conn.close()

    print(f"\n[Migration] Complete!")
    print(f"  Migrated: {migrated}/{len(assets)}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for name, err in errors:
            print(f"    - {name}: {err}")

    # Show final counts by category
    print(f"\n[Migration] Final asset counts:")
    for category in CATEGORIES:
        category_path = os.path.join(TARGET_CLOSET, category)
        count = len([f for f in os.listdir(category_path) if f.endswith('.json')]) if os.path.exists(category_path) else 0
        print(f"  {category}: {count}")


if __name__ == "__main__":
    migrate_assets()
