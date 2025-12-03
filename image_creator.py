"""
Nova2 Image Creator
Standalone image editing module using Gemini gemini-3-pro-image-preview
"""

import os
import io
import base64
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_IMAGE = "gemini-3-pro-image-preview"
MODEL_REWRITE = "gemini-2.5-flash-lite"

# Initialize Gemini client
client = genai.Client(api_key=API_KEY)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style presets from prompt.txt
STYLE_PRESETS = {
    "instagram_v1": """The overall aesthetic is clean, modern, and editorial — blending naturalistic daylight with polished lifestyle photography. Crystal clear focus from foreground to background, no blur, no bokeh. Lighting is direct, diffused and even, no harsh contrast. great lighting. Exposure is bright yet balanced, add a little brilliance, preserving skin texture with a smooth, clean finish. The color palette favors neutrals, soft pastels, and subtle pops of saturated color, creating a calm, inviting mood with understated sophistication. Textures appear crisp and tactile, rendered with clean iPhone 16 pro camera style that adds organic and realism. Natural poses, slightly off-center composition, well lit Instagram style photo. This style prioritizes quiet confidence, natural beauty, and relaxed authenticity. It blends editorial polish with approachable everyday moments, producing aesthetic, brilliant, Instagram-ready imagery that feels both cinematic and effortlessly real.""",

    "instagram_v2": """High-end social media lifestyle photography, viral influencer aesthetic, "It Girl/Boy" energy. The image quality is pristine and ultra-sharp with 8K definition. The composition is clean and curated, focusing perfectly on the subject. The subject exudes an attitude of 'effortless cool' and 'unbothered confidence'—a relaxed, slightly aloof expression with a direct, captivating gaze (siren eyes) or a candid 'too cool to care' look. Soft, premium lighting (natural or studio-like) highlights smooth, even skin and outfit textures. The outfit should be chic, instagram style young, and make sense to the scenario. The color palette is balanced, modern, and aesthetically pleasing. diffused editorial polish with high production quality photo, Canon G12 aesthetic, no clutter. background should also be crisp clear NO blur.""",

    "instagram_v3": """High-quality social media lifestyle photography with a contemporary influencer aesthetic and effortless chic vibe. Lighting is bright, soft, and evenly balanced, creating flattering, glowy skin tones without washing out details. The color palette is modern, clean, and natural—vibrant but realistic, with true-to-life tones across both subject and background. Shot in a high-end mirrorless camera style with a 35mm lens at f/4–f/5.6, producing a crisp, clear background with full environmental detail instead of artificial blur. The entire scene—foreground, midground, and background—remains sharp and well-defined, enhancing realism and avoiding the typical AI soft-focus look. Textures appear polished and photorealistic: clean lines, sharp edges, accurate lighting falloff, and natural grain. Overall the aesthetic is editorial yet approachable, featuring clear details, structured composition, and rich visual depth. High resolution, ultra-detailed, fully in-focus environment, high-fashion lifestyle energy.""",

    "pastel": """Soft lifestyle editorial aesthetic with bright natural daylight and a clean, high-key atmosphere. The lighting is diffused window light that creates smooth, even illumination across the scene, producing soft shadows, glossy skin highlights, and a gentle glow. Colors appear fresh and vibrant — pastel pinks, cool blues, natural browns, and crisp whites — giving the composition a playful yet polished visual tone. The overall look is bright, youthful, and airy, with low contrast, fine film-grain texture, and a subtly softened focus reminiscent of modern Instagram editorials. The framing is tight and intimate, emphasizing personal space and curated objects arranged artfully around the subject. The image style blends lifestyle candidness with aesthetic intentionality, combining cozy textures, clean light, and a dreamy, contemporary color palette.""",

    "editorial": """The overall style is photorealistic, natural, editorial, soft, and high-fashion, reflecting an elevated aesthetic. The photography features medium contrast with defined but soft shadows, ensuring a balanced look. The image is well-exposed and slightly bright, not overexposed or washed out. Colors are saturated, natural tones in the environment, ensuring it is not too warm-toned. The overall texture of the image is characterized by slight film grain, natural, defined, soft, elevated. Lighting is natural and even, originating from the front and slightly overhead, illuminating both the subject and the detailed background with soft lighting and soft shadows, resulting in a slightly glowy lighting effect, natural lighting. If a sky is present in a daytime scenario, it is light blue-toned. The overall tone is balanced, and the image is moderately contrasty with vibrant, saturated colors, not muted. The overall image is soft and balanced, shadows are soft.""",

    "cool_cinematic": """The overall style is photorealistic, editorial, and high-fashion, exhibiting a naturalistic photography style with medium contrast and defined but soft shadows, and balanced exposure. The image features a cool palette with slightly muted tones. The texture is characterized by film grain, hazy, soft, polished, smooth, analog texture, and a subtle glow. The lighting is soft, diffuse natural daylight coming from above, casting soft shadows, illuminating the person clearly. Natural lighting, soft shadows, soft lighting, glowy lighting, and flat lighting, emphasizing a muted yet medium contrast appearance.""",

    "fashion_v1": """High-fashion editorial aesthetic with a clean, sculptural composition and a strong sense of spatial minimalism. Photography emphasizes precise framing, often using centered or near-centered subject placement, symmetrical blocking, and intentional negative space. The overall visual design favors bold, graphic silhouettes against uncluttered backgrounds, allowing shapes, textures, and posture to become the main storytelling elements. Lighting is soft but directional, producing smooth gradients and gentle shadow falloff that contours the figure without harsh contrast. The palette leans toward sun-kissed warmth, muted pastels, creamy neutrals, or soft Mediterranean tones, depending on the scene. Surfaces appear matte and tactile, with natural grain or subtle analog softness, maintaining a luxurious but organic finish. The camera style resembles medium-format fashion photography — crisp edges, gentle depth of field, and a polished yet unretouched feel. Textures such as fabric sheens, skin glow, flowers, or architectural materials are highlighted with clarity. Colors remain balanced and serene, avoiding saturation spikes, creating a timeless, editorial calm. Posing is structured yet nonchalant: relaxed limbs, elongated lines, a sense of ease or quiet confidence. Overall, the aesthetic merges minimalist composition, warm natural light, sculptural posing, and subtle surrealism — elegant, airy, modern, and art-directed with precision. The background should also be in focus.""",

    "fashion_v2": """High-fashion minimalist campaign aesthetic with clean, spacious composition and strong geometric blocking. The lighting is bright, soft, and evenly diffused—natural daylight or high-key studio light that eliminates harsh shadows and makes colors appear crisp and luminous. The palette favors warm neutrals, sun-kissed skin tones, creamy whites, muted pastels, and the occasional vivid statement accent color such as red, blue, yellow, or green. Textures appear tactile and premium: matte fabrics, smooth skin highlights, soft knits, polished surfaces. The camera style is straightforward and editorial, using centered framing, symmetrical balance, and generous negative space. Subjects are often placed against simple backdrops—open landscapes, blank walls, textured architectural surfaces—emphasizing silhouette and shape. Full-body shots use a fashion-catalogue perspective with slight distance; close-ups maintain clarity and sculpted lighting. Shadows are minimal and clean, enhancing the refined, modern aesthetic. The mood is calm, confident, and subtly playful, featuring relaxed yet assertive poses: upright posture, slight lean, gentle extension of limbs, or quiet stillness. The overall tone is elegant and uncluttered, blending luxury minimalism with a whimsical touch. The aesthetic is fresh, airy, understatedly chic, and compositionally meticulous.""",

    "ccd_flash": """Early 2000s vintage digital camera aesthetic, CCD sensor color science. Shot on a Canon PowerShot or Nikon Coolpix compact point-and-shoot. Harsh direct on-camera flash creating a spotlight effect on the subject with hard shadows. The skin texture appears smooth yet raw with a signature glowing "flash burnout" look. slightly low-fidelity, authentic high-ISO digital noise, chromatic aberration, vibrant and saturated colors, lo-fi candid snapshot, tumblr girl aesthetic, chaotic energy, unpolished but fashionable.""",

    "landscape": """The overall style is photorealistic, natural, editorial, soft, and elevated, with a warm golden-hour atmosphere that enhances the outdoor travel aesthetic. Poses are playful, relaxed, and natural — whether sitting, standing, walking, or even lightly jumping — always captured from a far-away, wide shot so the figure remains small within the vast landscape. The vibe is carefree, joyful, and adventurous, blending soft editorial refinement with spontaneous outdoor energy. Lighting is cinematic outdoor natural light, casting gentle highlights and long, soft shadows while creating a glowing, cinematic ambience. Reds, oranges, and yellows are slightly more saturated to enrich the warmth of the environment while keeping skin tones natural and balanced. Camera angles are primarily eye-level or slightly low, emphasizing expansive mountain ranges, forests, and open skies. The photography features medium contrast with soft shadows and a bright but controlled exposure. A subtle layer of film grain adds an elevated, organic texture. The overall tone is vivid, warm, and immersive, maintaining a soft editorial feel while showcasing broad, scenic outdoor landscapes.""",

    "celebrity": """Analog film photograph, shot on Leica M6 with Kodak Portra 400. A chaotic, high-energy candid moment. Harsh direct on-camera flash, blinding strobe lights from surrounding cameras, deep shadows, heavy film grain, motion blur, raw and authentic Magnum agency aesthetic, celebrity spotting atmosphere.""",

    "mirror_selfie": """Convert the subject into a front, a profile or back angle of confident mirror-style selfie of the subject arranging their wet hair with bold lighting and modern contrast. Pose naturally while holding the phone in a bathroom or modern interior. Lighting should emphasize body definition — strong definition with a subtle film grain. The final look should feel empowered, bold, and glossy — a high-contrast social media photo that captures confidence without overediting. Hugo Comte photography style, high definition photo, strong grainy.""",
}

# Style categories for detection
STYLE_CATEGORIES = {
    "instagram": ["instagram", "social media", "influencer", "lifestyle", "daily", "casual"],
    "pastel": ["pastel", "soft", "dreamy", "pink", "airy", "gentle"],
    "editorial": [],  # Default fallback
    "cool_cinematic": ["cinematic", "cool", "muted", "film", "moody"],
    "fashion": ["fashion", "campaign", "model", "runway", "chic", "haute"],
    "ccd_flash": ["night", "club", "party", "retro", "2000s", "flash", "ccd", "nightlife", "bar"],
    "landscape": ["nature", "outdoor", "landscape", "mountain", "hiking", "travel", "beach", "forest", "sunset"],
    "celebrity": ["celebrity", "paparazzi", "star", "idol", "famous", "red carpet"],
    "mirror_selfie": ["mirror", "selfie", "sexy", "hot", "bathroom"],
}


def detect_style(original_prompt: str) -> dict:
    """
    Analyze prompt and select the best matching style preset.
    Uses AI to understand the prompt context and select appropriate style.

    Args:
        original_prompt: The user's original prompt

    Returns:
        dict with keys:
            - style_name: Human-readable style name
            - style_key: Internal key for STYLE_PRESETS
            - style_prompt: The full preset prompt text
    """
    try:
        print("[Style] Detecting style from prompt...")

        # Build the style detection prompt
        style_instruction = f"""Analyze this image editing prompt and determine the best aesthetic style category:

Prompt: "{original_prompt}"

Available categories:
1. instagram - Daily lifestyle photos, social media, influencer content, casual moments
2. pastel - Soft, dreamy, pastel colors, gentle and airy aesthetic
3. editorial - Default professional photography style (use if unsure)
4. cool_cinematic - Cinematic, cool tones, muted colors, film-like
5. fashion - Fashion campaign, high-fashion, runway, model shoots
6. ccd_flash - Night photos, club/party, retro 2000s flash photography
7. landscape - Nature, outdoor, travel, mountains, scenic landscapes
8. celebrity - Paparazzi style, celebrity spotting, red carpet
9. mirror_selfie - Mirror selfies, bathroom photos, confident/sexy poses
10. custom - User specified their own aesthetic style in the prompt

Rules:
- If the prompt mentions a specific aesthetic/style (e.g., "anime style", "watercolor", "oil painting"), return "custom" and extract their aesthetic
- If unsure, default to "editorial"
- Return ONLY a JSON object, nothing else

Output format:
{{"category": "category_name", "custom_aesthetic": "extracted aesthetic if custom, otherwise null"}}

Examples:
- "I'm at a club dancing" -> {{"category": "ccd_flash", "custom_aesthetic": null}}
- "me hiking in the mountains" -> {{"category": "landscape", "custom_aesthetic": null}}
- "I'm eating breakfast" -> {{"category": "editorial", "custom_aesthetic": null}}
- "me in anime style" -> {{"category": "custom", "custom_aesthetic": "anime style"}}
- "portrait in watercolor aesthetic" -> {{"category": "custom", "custom_aesthetic": "watercolor aesthetic"}}

Output:"""

        response = client.models.generate_content(
            model=MODEL_REWRITE,
            contents=[style_instruction]
        )

        result_text = response.text.strip()
        print(f"[Style] AI response: {result_text}")

        # Parse JSON response
        import json
        # Handle markdown code blocks
        if "```" in result_text:
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()

        result = json.loads(result_text)
        category = result.get("category", "editorial")
        custom_aesthetic = result.get("custom_aesthetic")

        # Handle custom aesthetic
        if category == "custom" and custom_aesthetic:
            print(f"[Style] Custom aesthetic detected: {custom_aesthetic}")
            return {
                "style_name": f"Custom: {custom_aesthetic}",
                "style_key": "custom",
                "style_prompt": custom_aesthetic
            }

        # Handle Instagram - random selection
        if category == "instagram":
            version = random.choice(["v1", "v2", "v3"])
            style_key = f"instagram_{version}"
            style_name = f"Instagram {version.upper()}"
            print(f"[Style] Instagram detected, randomly selected: {version}")
        # Handle Fashion - random selection
        elif category == "fashion":
            version = random.choice(["v1", "v2"])
            style_key = f"fashion_{version}"
            style_name = f"Fashion Campaign {version.upper()}"
            print(f"[Style] Fashion detected, randomly selected: {version}")
        # Handle other categories
        elif category in ["pastel", "editorial", "cool_cinematic", "ccd_flash", "landscape", "celebrity", "mirror_selfie"]:
            style_key = category
            style_names = {
                "pastel": "Pastel",
                "editorial": "Editorial",
                "cool_cinematic": "Cool Cinematic",
                "ccd_flash": "CCD Flash (Night/Retro)",
                "landscape": "Nature/Landscape",
                "celebrity": "Celebrity/Paparazzi",
                "mirror_selfie": "Mirror Selfie"
            }
            style_name = style_names.get(category, category.title())
        else:
            # Default to editorial
            style_key = "editorial"
            style_name = "Editorial"
            print(f"[Style] Unknown category '{category}', defaulting to Editorial")

        style_prompt = STYLE_PRESETS.get(style_key, STYLE_PRESETS["editorial"])

        print(f"[Style] Selected: {style_name}")
        return {
            "style_name": style_name,
            "style_key": style_key,
            "style_prompt": style_prompt
        }

    except Exception as e:
        print(f"[Style] Error: {e}, defaulting to Editorial")
        return {
            "style_name": "Editorial",
            "style_key": "editorial",
            "style_prompt": STYLE_PRESETS["editorial"]
        }


def _rewrite_call(original_prompt: str, image_description: str) -> str:
    """
    Internal helper: Rewrite the prompt based on image description.
    """
    rewrite_instruction = f"""The user wants to edit an image. Their original prompt is:
"{original_prompt}"

For reference, the attached image shows:
{image_description}

Your task: Rewrite ONLY the user's prompt to fix grammar.

IMPORTANT RULES:
- Keep the user's ORIGINAL INTENT - what they want to DO/CREATE
- DO NOT describe what's in the image
- DO NOT repeat the subject description
- Just clean up grammar and make it natural
- Keep it SHORT and simple

Examples:
- "I'm sleeping with a sheep" -> "I am sleeping with a sheep"
- "me driving car" -> "I am driving a car"
- "I eating pizza with friend" -> "I am eating pizza with a friend"

Rewritten prompt (just the action, nothing else):"""

    response = client.models.generate_content(
        model=MODEL_REWRITE,
        contents=[rewrite_instruction]
    )

    rewritten = response.text.strip()

    # Clean up common prefixes
    prefixes_to_remove = [
        "Rewritten editing prompt: ",
        "Rewritten prompt: ",
        "Editing prompt: ",
        "Prompt: "
    ]
    for prefix in prefixes_to_remove:
        if rewritten.startswith(prefix):
            rewritten = rewritten[len(prefix):]
            break

    # Remove quotes if added
    if rewritten.startswith('"') and rewritten.endswith('"'):
        rewritten = rewritten[1:-1]

    return rewritten


def rewrite_prompt(original_prompt: str, image_bytes: bytes) -> dict:
    """
    Use AI to describe the image, rewrite the prompt, and detect style.
    Runs style detection in parallel with prompt rewriting for performance.

    Args:
        original_prompt: User's original editing prompt
        image_bytes: The input image as bytes

    Returns:
        dict with keys:
            - original_prompt: The original user prompt
            - description: AI-generated image description
            - rewritten_prompt: Enhanced editing prompt (with style appended)
            - structured_output: Formatted output with subjects, editing prompt, and style
            - style_name: The detected/selected style name
            - style_prompt: The full style preset text
    """
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Step 1: Describe the attached image (must complete first)
        describe_instruction = "Describe this person/subject in detail. Include their appearance, what they're wearing, their features, setting, etc. Be concise but descriptive (1-2 sentences)."

        print("[Rewrite] Describing attached image...")
        describe_response = client.models.generate_content(
            model=MODEL_REWRITE,
            contents=[describe_instruction, pil_image]
        )
        image_description = describe_response.text.strip()
        print(f"[Rewrite] Image description: {image_description}")

        # Build subjects list (only attached images)
        description_text = f"1. I/myself: {image_description} (attached image 1)"
        print(f"[Rewrite] Subjects:\n{description_text}")

        # Step 2: Run rewrite and style detection in PARALLEL
        print("[Rewrite] Running rewrite + style detection in parallel...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            rewrite_future = executor.submit(_rewrite_call, original_prompt, image_description)
            style_future = executor.submit(detect_style, original_prompt)

            # Wait for both to complete
            rewritten = rewrite_future.result()
            style_result = style_future.result()

        print(f"[Rewrite] Original: {original_prompt}")
        print(f"[Rewrite] Rewritten: {rewritten}")
        print(f"[Rewrite] Style: {style_result['style_name']}")

        # Format structured output (for display) - show full style prompt
        structured_output = "Subjects:\n"
        structured_output += description_text
        structured_output += f"\n\nEditing prompt: {rewritten}"
        structured_output += f"\n\nAesthetics/Style: {style_result['style_prompt']}"

        # Combine ALL parts for image generation (includes subjects)
        final_prompt = f"Subjects:\n{description_text}\n\nEditing prompt: {rewritten}\n\n{style_result['style_prompt']}"

        return {
            "original_prompt": original_prompt,
            "description": description_text,
            "rewritten_prompt": final_prompt,  # This includes style for generation
            "structured_output": structured_output,
            "style_name": style_result["style_name"],
            "style_prompt": style_result["style_prompt"]
        }

    except Exception as e:
        print(f"[Rewrite] Error: {e}, using original prompt")
        # On error, use Editorial as default style
        default_style = STYLE_PRESETS["editorial"]
        return {
            "original_prompt": original_prompt,
            "description": "",
            "rewritten_prompt": f"{original_prompt}\n\n{default_style}",
            "structured_output": f"Editing prompt: {original_prompt}\n\nAesthetics/Style: {default_style}",
            "style_name": "Editorial",
            "style_prompt": default_style
        }


def generate_image(
    image_bytes: bytes,
    prompt: str,
    aspect_ratio: str = "9:16",
    image_size: str = "1K"
) -> bytes:
    """
    Generate edited image using Gemini gemini-3-pro-image-preview.

    Args:
        image_bytes: Input image as bytes
        prompt: Editing prompt
        aspect_ratio: Output aspect ratio (1:1, 9:16, 16:9, 4:3, 3:4)
        image_size: Resolution (1K, 2K, 4K) - only for gemini-3-pro-image-preview

    Returns:
        Generated image as JPEG bytes
    """
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))

        print(f"\n[Generate] === IMAGE EDITING ===")
        print(f"[Generate] Model: {MODEL_IMAGE}")
        print(f"[Generate] Prompt: {prompt}")
        print(f"[Generate] Input size: {pil_image.size} ({pil_image.mode})")
        print(f"[Generate] Aspect ratio: {aspect_ratio}")
        print(f"[Generate] Image size: {image_size}")
        print(f"[Generate] ====================\n")

        # Build contents: [prompt, image]
        contents = [prompt, pil_image]

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


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(b64_string: str) -> bytes:
    """Convert base64 string to image bytes."""
    return base64.b64decode(b64_string)


def run_cli():
    """Interactive CLI mode for image editing."""
    print("\n" + "="*50)
    print("Nova2 Image Creator - CLI Mode")
    print("="*50)
    print(f"Model: {MODEL_IMAGE}")
    print("Type 'quit' to exit\n")

    while True:
        # Get image path
        image_path = input("Image path (or 'quit'): ").strip()
        if image_path.lower() == 'quit':
            break

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

        # Get prompt
        prompt = input("Editing prompt: ").strip()
        if not prompt:
            print("Error: Prompt cannot be empty")
            continue

        # Ask about AI rewrite (default: yes)
        use_rewrite = input("Use AI to enhance prompt? (y/n) [y]: ").strip().lower()
        if use_rewrite != 'n':
            result = rewrite_prompt(prompt, image_bytes)
            print(f"\n{result['structured_output']}\n")
            prompt = result['rewritten_prompt']

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
            result_bytes = generate_image(image_bytes, prompt, aspect, size)
            output_path = save_image(result_bytes)
            print(f"\nDone! Image saved to: {output_path}\n")
        except Exception as e:
            print(f"Error generating image: {e}\n")


if __name__ == "__main__":
    run_cli()
