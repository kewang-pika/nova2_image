"""
Nova2 Image Creator - History Module
Save and load generation history with images
"""
import os
import json
import base64
from datetime import datetime
from typing import Optional

# History directory
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Password for viewing history
HISTORY_PASSWORD = "nova2image"


def _generate_id() -> str:
    """Generate unique ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_generation(
    original_prompt: str,
    editing_prompt: str,
    structured_prompt: str,
    style_name: str,
    main_image_bytes: bytes,
    variation_images: list = None,
    variation_prompts: list = None,
    variation_full_prompts: list = None,
    aspect_ratio: str = "9:16",
    image_size: str = "1K",
    aesthetic: dict = None
) -> str:
    """
    Save a generation to history.

    Args:
        original_prompt: User's original prompt
        editing_prompt: Rewritten/enhanced editing prompt
        structured_prompt: Full structured prompt with subjects
        style_name: Detected style name
        main_image_bytes: Generated main image
        variation_images: List of variation image bytes (optional)
        variation_prompts: List of short variation prompts (optional)
        variation_full_prompts: List of full structured variation prompts (optional)
        aspect_ratio: Aspect ratio used
        image_size: Image size used
        aesthetic: Optional aesthetic preferences dict

    Returns:
        Generation ID
    """
    gen_id = _generate_id()
    gen_dir = os.path.join(HISTORY_DIR, gen_id)
    os.makedirs(gen_dir, exist_ok=True)

    # Save main image
    main_image_path = os.path.join(gen_dir, "main.jpg")
    with open(main_image_path, 'wb') as f:
        f.write(main_image_bytes)

    # Save variation images
    variation_files = []
    if variation_images:
        for i, img_bytes in enumerate(variation_images):
            if img_bytes:
                var_path = os.path.join(gen_dir, f"variation_{i+1}.jpg")
                with open(var_path, 'wb') as f:
                    f.write(img_bytes)
                variation_files.append(f"variation_{i+1}.jpg")

    # Save metadata
    metadata = {
        "id": gen_id,
        "timestamp": datetime.now().isoformat(),
        "original_prompt": original_prompt,
        "editing_prompt": editing_prompt,
        "structured_prompt": structured_prompt,
        "style_name": style_name,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "main_image": "main.jpg",
        "variations": variation_files,
        "variation_prompts": variation_prompts or [],
        "variation_full_prompts": variation_full_prompts or [],
        "aesthetic": aesthetic
    }

    metadata_path = os.path.join(gen_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[History] Saved generation {gen_id}")
    return gen_id


def update_generation_variations(
    gen_id: str,
    variation_images: list,
    variation_prompts: list,
    variation_full_prompts: list
) -> bool:
    """
    Update an existing generation with variations.

    Args:
        gen_id: Generation ID to update
        variation_images: List of variation image bytes
        variation_prompts: List of short variation prompts
        variation_full_prompts: List of full structured variation prompts

    Returns:
        True if successful
    """
    gen_dir = os.path.join(HISTORY_DIR, gen_id)
    metadata_path = os.path.join(gen_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        print(f"[History] Generation {gen_id} not found")
        return False

    # Load existing metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Save variation images
    variation_files = []
    for i, img_bytes in enumerate(variation_images):
        if img_bytes:
            var_path = os.path.join(gen_dir, f"variation_{i+1}.jpg")
            with open(var_path, 'wb') as f:
                f.write(img_bytes)
            variation_files.append(f"variation_{i+1}.jpg")

    # Update metadata
    metadata["variations"] = variation_files
    metadata["variation_prompts"] = variation_prompts
    metadata["variation_full_prompts"] = variation_full_prompts

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"[History] Updated generation {gen_id} with {len(variation_files)} variations")
    return True


def get_generation(gen_id: str) -> Optional[dict]:
    """
    Load a generation by ID.

    Returns:
        dict with metadata and image bytes, or None if not found
    """
    gen_dir = os.path.join(HISTORY_DIR, gen_id)
    metadata_path = os.path.join(gen_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load main image
    main_path = os.path.join(gen_dir, metadata["main_image"])
    with open(main_path, 'rb') as f:
        metadata["main_image_bytes"] = f.read()

    # Load variation images
    metadata["variation_image_bytes"] = []
    for var_file in metadata.get("variations", []):
        var_path = os.path.join(gen_dir, var_file)
        if os.path.exists(var_path):
            with open(var_path, 'rb') as f:
                metadata["variation_image_bytes"].append(f.read())

    return metadata


def list_generations(limit: int = 50) -> list:
    """
    List all generations, newest first.

    Args:
        limit: Maximum number of generations to return

    Returns:
        List of generation metadata (without image bytes)
    """
    generations = []

    if not os.path.exists(HISTORY_DIR):
        return generations

    # Get all generation directories
    dirs = []
    for name in os.listdir(HISTORY_DIR):
        gen_dir = os.path.join(HISTORY_DIR, name)
        metadata_path = os.path.join(gen_dir, "metadata.json")
        if os.path.isdir(gen_dir) and os.path.exists(metadata_path):
            dirs.append((name, gen_dir, metadata_path))

    # Sort by directory name (timestamp-based) descending
    dirs.sort(key=lambda x: x[0], reverse=True)

    # Load metadata for each
    for name, gen_dir, metadata_path in dirs[:limit]:
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Add thumbnail (base64) for preview
            main_path = os.path.join(gen_dir, metadata["main_image"])
            if os.path.exists(main_path):
                with open(main_path, 'rb') as f:
                    metadata["main_image_bytes"] = f.read()

            generations.append(metadata)
        except Exception as e:
            print(f"[History] Error loading {name}: {e}")

    return generations


def delete_generation(gen_id: str) -> bool:
    """Delete a generation by ID."""
    import shutil
    gen_dir = os.path.join(HISTORY_DIR, gen_id)

    if os.path.exists(gen_dir):
        shutil.rmtree(gen_dir)
        print(f"[History] Deleted generation {gen_id}")
        return True
    return False


def verify_password(password: str) -> bool:
    """Verify the history password."""
    return password == HISTORY_PASSWORD


def get_history_stats() -> dict:
    """Get statistics about history."""
    if not os.path.exists(HISTORY_DIR):
        return {"total": 0, "with_variations": 0}

    total = 0
    with_variations = 0

    for name in os.listdir(HISTORY_DIR):
        gen_dir = os.path.join(HISTORY_DIR, name)
        metadata_path = os.path.join(gen_dir, "metadata.json")
        if os.path.isdir(gen_dir) and os.path.exists(metadata_path):
            total += 1
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                if metadata.get("variations"):
                    with_variations += 1
            except:
                pass

    return {"total": total, "with_variations": with_variations}
