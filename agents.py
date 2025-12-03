"""
Nova2 Image Creator - Agent Classes
StyleAgent, PromptAgent, VariationsAgent
"""
import os
import io
import re
import json
import base64
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ============== CONFIG ==============
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_REWRITE = "gemini-2.5-flash"
client = genai.Client(api_key=API_KEY)

# ============== STYLE PRESETS ==============
STYLE_PRESETS = {
    "instagram_v1": """The overall aesthetic is clean, modern, and editorial — blending naturalistic daylight with polished lifestyle photography. Crystal clear focus from foreground to background, no blur, no bokeh. Lighting is direct, diffused and even, no harsh contrast. great lighting. Exposure is bright yet balanced, add a little brilliance, preserving skin texture with a smooth, clean finish. The color palette favors neutrals, soft pastels, and subtle pops of saturated color, creating a calm, inviting mood with understated sophistication. Textures appear crisp and tactile, rendered with clean iPhone 16 pro camera style that adds organic and realism. Natural poses, slightly off-center composition, well lit Instagram style photo. This style prioritizes quiet confidence, natural beauty, and relaxed authenticity. It blends editorial polish with approachable everyday moments, producing aesthetic, brilliant, Instagram-ready imagery that feels both cinematic and effortlessly real.""",

    "instagram_v2": """High-end social media lifestyle photography, viral influencer aesthetic, "It Girl/Boy" energy. The image quality is pristine and ultra-sharp with 8K definition. The composition is clean and curated, focusing perfectly on the subject. The subject exudes an attitude of 'effortless cool' and 'unbothered confidence'—a relaxed, slightly aloof expression with a direct, captivating gaze (siren eyes) or a candid 'too cool to care' look. Soft, premium lighting (natural or studio-like) highlights smooth, even skin and outfit textures. The outfit should be chic, instagram style young, and make sense to the scenario. The color palette is balanced, modern, and aesthetically pleasing. diffused editorial polish with high production quality photo, Canon G12 aesthetic, no clutter. background should also be crisp clear NO blur.""",

    "instagram_v3": """High-quality social media lifestyle photography with a contemporary influencer aesthetic and effortless chic vibe. Lighting is bright, soft, and evenly balanced, creating flattering, glowy skin tones without washing out details. The color palette is modern, clean, and natural—vibrant but realistic, with true-to-life tones across both subject and background. Shot in a high-end mirrorless camera style with a 35mm lens at f/4–f/5.6, producing a crisp, clear background with full environmental detail instead of artificial blur. The entire scene—foreground, midground, and background—remains sharp and well-defined, enhancing realism and avoiding the typical AI soft-focus look. Textures appear polished and photorealistic: clean lines, sharp edges, accurate lighting falloff, and natural grain. Overall the aesthetic is editorial yet approachable, featuring clear details, structured composition, and rich visual depth. High resolution, ultra-detailed, fully in-focus environment, high-fashion lifestyle energy.""",

    "pastel": """Soft lifestyle editorial aesthetic with bright natural daylight and a clean, high-key atmosphere. The lighting is diffused window light that creates smooth, even illumination across the scene, producing soft shadows, glossy skin highlights, and a gentle glow. Colors appear fresh and vibrant — pastel pinks, cool blues, natural browns, and crisp whites — giving the composition a playful yet polished visual tone. The overall look is bright, youthful, and airy, with low contrast, fine film-grain texture, and a subtly softened focus reminiscent of modern Instagram editorials. The framing is tight and intimate, emphasizing personal space and curated objects arranged artfully around the subject. The image style blends lifestyle candidness with aesthetic intentionality, combining cozy textures, clean light, and a dreamy, contemporary color palette.""",

    "fuji_sunglow": """Photorealistic, crisp, and soft-editorial aesthetic with a distinctly Fujifilm-inspired grainy finish. The images feature glowing, luminous skin with polished highlights and smooth tonal transitions, enhanced by subtle lighting-panel adjustments that create a bright, airy glow without losing detail. Colors are vibrant yet refined — clean ocean blues, warm skin tones, and natural landscape hues — rendered with Fujifilm's signature saturation, gentle contrast, and film-like color depth. The overall look blends sharpness and softness: high-resolution detail in hair, skin, and fabric paired with a noticeable layer of fine film-style grain that adds texture and warmth. Lighting is natural and directional, creating soft, flattering shadows and a slight radiance around the subject. Backgrounds remain crisp and scenic, with landscapes and indoor settings captured in clear, true-to-life tones. The final aesthetic is elevated, glowing, slightly nostalgic, and editorial — a fusion of modern clarity with classic Fujifilm grain and dreamy softness.""",

    "editorial": """The overall style is photorealistic, natural, editorial, soft, and high-fashion, reflecting an elevated aesthetic. The photography features medium contrast with defined but soft shadows, ensuring a balanced look. The image is well-exposed and slightly bright, not overexposed or washed out. Colors are saturated, natural tones in the environment, ensuring it is not too warm-toned. The overall texture of the image is characterized by slight film grain, natural, defined, soft, elevated. Lighting is natural and even, originating from the front and slightly overhead, illuminating both the subject and the detailed background with soft lighting and soft shadows, resulting in a slightly glowy lighting effect, natural lighting. If a sky is present in a daytime scenario, it is light blue-toned. The overall tone is balanced, and the image is moderately contrasty with vibrant, saturated colors, not muted. The overall image is soft and balanced, shadows are soft.""",

    "cool_cinematic": """The overall style is photorealistic, editorial, and high-fashion, exhibiting a naturalistic photography style with medium contrast and defined but soft shadows, and balanced exposure. The image features a cool palette with slightly muted tones. The texture is characterized by film grain, hazy, soft, polished, smooth, analog texture, and a subtle glow. The lighting is soft, diffuse natural daylight coming from above, casting soft shadows, illuminating the person clearly. Natural lighting, soft shadows, soft lighting, glowy lighting, and flat lighting, emphasizing a muted yet medium contrast appearance.""",

    "fashion_v1": """High-fashion editorial aesthetic with a clean, sculptural composition and a strong sense of spatial minimalism. Photography emphasizes precise framing, often using centered or near-centered subject placement, symmetrical blocking, and intentional negative space. The overall visual design favors bold, graphic silhouettes against uncluttered backgrounds, allowing shapes, textures, and posture to become the main storytelling elements. Lighting is soft but directional, producing smooth gradients and gentle shadow falloff that contours the figure without harsh contrast. The palette leans toward sun-kissed warmth, muted pastels, creamy neutrals, or soft Mediterranean tones, depending on the scene. Surfaces appear matte and tactile, with natural grain or subtle analog softness, maintaining a luxurious but organic finish. The camera style resembles medium-format fashion photography — crisp edges, gentle depth of field, and a polished yet unretouched feel. Textures such as fabric sheens, skin glow, flowers, or architectural materials are highlighted with clarity. Colors remain balanced and serene, avoiding saturation spikes, creating a timeless, editorial calm. Posing is structured yet nonchalant: relaxed limbs, elongated lines, a sense of ease or quiet confidence. Overall, the aesthetic merges minimalist composition, warm natural light, sculptural posing, and subtle surrealism — elegant, airy, modern, and art-directed with precision. The background should also be in focus.""",

    "fashion_v2": """High-fashion minimalist campaign aesthetic with clean, spacious composition and strong geometric blocking. The lighting is bright, soft, and evenly diffused—natural daylight or high-key studio light that eliminates harsh shadows and makes colors appear crisp and luminous. The palette favors warm neutrals, sun-kissed skin tones, creamy whites, muted pastels, and the occasional vivid statement accent color such as red, blue, yellow, or green. Textures appear tactile and premium: matte fabrics, smooth skin highlights, soft knits, polished surfaces. The camera style is straightforward and editorial, using centered framing, symmetrical balance, and generous negative space. Subjects are often placed against simple backdrops—open landscapes, blank walls, textured architectural surfaces—emphasizing silhouette and shape. Full-body shots use a fashion-catalogue perspective with slight distance; close-ups maintain clarity and sculpted lighting. Shadows are minimal and clean, enhancing the refined, modern aesthetic. The mood is calm, confident, and subtly playful, featuring relaxed yet assertive poses: upright posture, slight lean, gentle extension of limbs, or quiet stillness. The overall tone is elegant and uncluttered, blending luxury minimalism with a whimsical touch. The aesthetic should evoke a fresh, airy, understatedly chic, and compositionally meticulous Jacquemus-style visual identity. Do NOT add text 'Jacquemus'.""",

    "ccd_flash": """Early 2000s vintage digital camera aesthetic, CCD sensor color science. Shot on a Canon PowerShot or Nikon Coolpix compact point-and-shoot. Harsh direct on-camera flash creating a spotlight effect on the subject with hard shadows. The skin texture appears smooth yet raw with a signature glowing "flash burnout" look. slightly low-fidelity, authentic high-ISO digital noise, chromatic aberration, vibrant and saturated colors, lo-fi candid snapshot, tumblr girl aesthetic, chaotic energy, unpolished but fashionable.""",

    "landscape": """The overall style is photorealistic, natural, editorial, soft, and elevated, with a warm golden-hour atmosphere that enhances the outdoor travel aesthetic. Poses are playful, relaxed, and natural — whether sitting, standing, walking, or even lightly jumping — always captured from a far-away, wide shot so the figure remains small within the vast landscape. The vibe is carefree, joyful, and adventurous, blending soft editorial refinement with spontaneous outdoor energy. Lighting is cinematic outdoor natural light, casting gentle highlights and long, soft shadows while creating a glowing, cinematic ambience. Reds, oranges, and yellows are slightly more saturated to enrich the warmth of the environment while keeping skin tones natural and balanced. Camera angles are primarily eye-level or slightly low, emphasizing expansive mountain ranges, forests, and open skies. The photography features medium contrast with soft shadows and a bright but controlled exposure. A subtle layer of film grain adds an elevated, organic texture. The overall tone is vivid, warm, and immersive, maintaining a soft editorial feel while showcasing broad, scenic outdoor landscapes.""",

    "celebrity": """Analog film photograph, shot on Leica M6 with Kodak Portra 400. A chaotic, high-energy candid moment. Harsh direct on-camera flash, blinding strobe lights from surrounding cameras, deep shadows, heavy film grain, motion blur, raw and authentic Magnum agency aesthetic, celebrity spotting atmosphere.""",

    "mirror_selfie": """Convert the subject into a front, a profile or back angle of confident mirror-style selfie of the subject arranging their wet hair with bold lighting and modern contrast. If female: Pose naturally while holding the phone in a bathroom or modern interior. Outfit can be a string bikini, or a set of hot sportswear, or hot crop outfit that highlights an impressive sexy body curve and proportion. Hair is a little bit wet or as specified. If male: Shirtless or wearing an open casual top or gym attire, in similar lighting and mirror composition. Keep reflections realistic and posture natural. Lighting should emphasize body definition — strong definition with a subtle film grain. The final look should feel empowered, bold, and glossy — a high-contrast social media photo that captures confidence without overediting. A bit more hot and curvy. Hair is wet, Hugo Comte photography style, high definition photo, strong grainy.""",
}

# Style categories for detection
STYLE_CATEGORIES = {
    "instagram": ["instagram", "social media", "influencer", "lifestyle", "daily", "casual"],
    "pastel": ["pastel", "soft", "dreamy", "pink", "airy", "gentle"],
    "fuji_sunglow": ["fuji", "fujifilm", "sun glow", "sunglow", "glow", "grain", "film grain", "nostalgic", "luminous"],
    "editorial": [],  # Default fallback
    "cool_cinematic": ["cinematic", "cool", "muted", "film", "moody"],
    "fashion": ["fashion", "campaign", "model", "runway", "chic", "haute"],
    "ccd_flash": ["night", "club", "party", "retro", "2000s", "flash", "ccd", "nightlife", "bar"],
    "landscape": ["nature", "outdoor", "landscape", "mountain", "hiking", "travel", "beach", "forest", "sunset"],
    "celebrity": ["celebrity", "paparazzi", "star", "idol", "famous", "red carpet"],
    "mirror_selfie": ["mirror", "selfie", "sexy", "hot", "bathroom"],
}


# ============================================
# AGENT 1: StyleAgent - Detect/determine style
# ============================================

# Default system prompt for StyleAgent
DEFAULT_STYLE_SYSTEM_PROMPT = """Analyze this image editing prompt and determine the best aesthetic style category:

Prompt: "{prompt}"

Available categories:
1. instagram - Daily lifestyle photos, social media, influencer content, casual moments
2. pastel - Soft, dreamy, pastel colors, gentle and airy aesthetic
3. fuji_sunglow - Fujifilm-inspired, sun glow, luminous skin, film grain, nostalgic warmth
4. editorial - Default professional photography style (use if unsure)
5. cool_cinematic - Cinematic, cool tones, muted colors, film-like
6. fashion - Fashion campaign, high-fashion, runway, model shoots
7. ccd_flash - Night photos, club/party, retro 2000s flash photography
8. landscape - Nature, outdoor, travel, mountains, scenic landscapes
9. celebrity - Paparazzi style, celebrity spotting, red carpet
10. mirror_selfie - Mirror selfies, bathroom photos, confident/sexy poses
11. custom - User specified their own aesthetic style in the prompt

Rules:
- If the prompt mentions a specific aesthetic/style (e.g., "anime style", "watercolor", "oil painting"), return "custom" and extract their aesthetic
- If unsure, default to "editorial"
- Return ONLY a JSON object, nothing else

Output format:
{{"category": "category_name", "custom_aesthetic": "extracted aesthetic if custom, otherwise null"}}

Output:"""


class StyleAgent:
    """Agent for detecting and selecting style presets."""

    def __init__(self, system_prompt: str = None):
        self.client = client
        self.model = MODEL_REWRITE
        self.presets = STYLE_PRESETS
        self.categories = STYLE_CATEGORIES
        self.system_prompt = system_prompt or DEFAULT_STYLE_SYSTEM_PROMPT

    def set_system_prompt(self, prompt: str):
        """Update the system prompt used for style detection."""
        self.system_prompt = prompt
        print(f"[StyleAgent] System prompt updated ({len(prompt)} chars)")

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self.system_prompt

    def detect(self, prompt: str) -> dict:
        """
        Analyze prompt and select the best matching style preset.
        Uses AI to understand the prompt context and select appropriate style.

        Returns:
            dict with style_name, style_key, style_prompt
        """
        try:
            print("[StyleAgent] Detecting style from prompt...")

            # Use the configurable system prompt with {prompt} placeholder
            style_instruction = self.system_prompt.format(prompt=prompt)

            response = self.client.models.generate_content(
                model=self.model,
                contents=[style_instruction]
            )

            result_text = response.text.strip()
            print(f"[StyleAgent] AI response: {result_text}")

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
                print(f"[StyleAgent] Custom aesthetic detected: {custom_aesthetic}")
                return {
                    "style_name": f"Custom: {custom_aesthetic}",
                    "style_key": "custom",
                    "style_prompt": custom_aesthetic
                }

            # Handle Instagram - prioritize v3
            if category == "instagram":
                style_key = "instagram_v3"
                style_name = "Instagram V3"
                print(f"[StyleAgent] Instagram detected, using prioritized: v3")
            # Handle Fashion - prioritize v2
            elif category == "fashion":
                style_key = "fashion_v2"
                style_name = "Fashion Campaign V2"
                print(f"[StyleAgent] Fashion detected, using prioritized: v2")
            # Handle other categories
            elif category in ["pastel", "fuji_sunglow", "editorial", "cool_cinematic", "ccd_flash", "landscape", "celebrity", "mirror_selfie"]:
                style_key = category
                style_names = {
                    "pastel": "Pastel",
                    "fuji_sunglow": "Fuji Film Sun Glow",
                    "editorial": "Editorial",
                    "cool_cinematic": "Cool Cinematic",
                    "ccd_flash": "CCD Flash (Night/Retro)",
                    "landscape": "Nature/Landscape",
                    "celebrity": "Celebrity/Paparazzi",
                    "mirror_selfie": "Mirror Selfie"
                }
                style_name = style_names.get(category, category.title())
            else:
                style_key = "editorial"
                style_name = "Editorial"
                print(f"[StyleAgent] Unknown category '{category}', defaulting to Editorial")

            style_prompt = self.presets.get(style_key, self.presets["editorial"])

            print(f"[StyleAgent] Selected: {style_name}")
            return {
                "style_name": style_name,
                "style_key": style_key,
                "style_prompt": style_prompt
            }

        except Exception as e:
            print(f"[StyleAgent] Error: {e}, defaulting to Editorial")
            return {
                "style_name": "Editorial",
                "style_key": "editorial",
                "style_prompt": self.presets["editorial"]
            }

    def detect_with_aesthetic(self, prompt: str, aesthetic: dict = None) -> dict:
        """
        Detect style using optional aesthetic preferences.

        Args:
            prompt: User's editing prompt
            aesthetic: Optional dict with 'visual_style' and 'avoids' keys
                Example: {
                    "visual_style": "Soft editorial, 35mm film grain, muted colors",
                    "avoids": ["neon", "high contrast", "cartoon"]
                }

        Returns:
            dict with style_name, style_key, style_prompt
        """
        # If no aesthetic provided, use regular detection
        if not aesthetic or not aesthetic.get("visual_style"):
            return self.detect(prompt)

        visual_style = aesthetic.get("visual_style", "")
        avoids = aesthetic.get("avoids", [])

        print(f"[StyleAgent] Using aesthetic preferences:")
        print(f"[StyleAgent]   Visual style: {visual_style}")
        print(f"[StyleAgent]   Avoids: {avoids}")

        try:
            # Build aesthetic-aware instruction
            avoids_str = ", ".join(avoids) if avoids else "none specified"

            aesthetic_instruction = f"""Given the user's aesthetic preferences, select the best matching style OR create a custom one.

User's editing prompt: "{prompt}"

User's aesthetic preferences:
- Visual style: {visual_style}
- Avoids: {avoids_str}

Available preset categories:
1. instagram - Daily lifestyle, social media, influencer content
2. pastel - Soft, dreamy, pastel colors, gentle and airy
3. fuji_sunglow - Fujifilm-inspired, sun glow, luminous skin, film grain, nostalgic
4. editorial - Professional photography, natural, soft shadows
5. cool_cinematic - Cinematic, cool tones, muted colors, film-like
6. fashion - Fashion campaign, high-fashion, runway, model shoots
7. ccd_flash - Night photos, club/party, retro 2000s flash
8. landscape - Nature, outdoor, travel, mountains, scenic
9. celebrity - Paparazzi style, celebrity spotting
10. mirror_selfie - Mirror selfies, confident/sexy poses

Instructions:
1. If the user's visual_style closely matches a preset, use that preset
2. If it doesn't match well, return "custom" with a detailed style prompt
3. The custom style prompt should incorporate the visual_style AND explicitly avoid the items in 'avoids'

Return JSON:
{{"category": "preset_name OR custom", "custom_style_prompt": "detailed style prompt if custom, otherwise null"}}

Output:"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=[aesthetic_instruction]
            )

            result_text = response.text.strip()
            print(f"[StyleAgent] AI response: {result_text}")

            # Handle markdown code blocks
            if "```" in result_text:
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            result = json.loads(result_text)
            category = result.get("category", "editorial")
            custom_style_prompt = result.get("custom_style_prompt")

            # Handle custom aesthetic - create new style prompt
            if category == "custom" and custom_style_prompt:
                # Append avoids explicitly
                if avoids:
                    avoids_clause = f" AVOID: {', '.join(avoids)}. Do NOT include these elements."
                    custom_style_prompt = custom_style_prompt.rstrip('.') + '.' + avoids_clause

                print(f"[StyleAgent] Created custom style from aesthetic")
                return {
                    "style_name": f"Custom Aesthetic",
                    "style_key": "custom_aesthetic",
                    "style_prompt": custom_style_prompt
                }

            # Use matched preset but check avoids
            style_key = category
            style_prompt = self.presets.get(style_key, self.presets["editorial"])

            # Append avoids to existing preset
            if avoids:
                avoids_clause = f" AVOID: {', '.join(avoids)}. Do NOT include these elements."
                style_prompt = style_prompt.rstrip('.') + '.' + avoids_clause

            # Map to display names
            style_names = {
                "instagram": "Instagram V3",
                "instagram_v3": "Instagram V3",
                "pastel": "Pastel",
                "fuji_sunglow": "Fuji Film Sun Glow",
                "editorial": "Editorial",
                "cool_cinematic": "Cool Cinematic",
                "fashion": "Fashion Campaign V2",
                "fashion_v2": "Fashion Campaign V2",
                "ccd_flash": "CCD Flash (Night/Retro)",
                "landscape": "Nature/Landscape",
                "celebrity": "Celebrity/Paparazzi",
                "mirror_selfie": "Mirror Selfie"
            }

            # Handle category mapping
            if category == "instagram":
                style_key = "instagram_v3"
            elif category == "fashion":
                style_key = "fashion_v2"

            style_name = style_names.get(style_key, category.title())
            style_prompt = self.presets.get(style_key, self.presets["editorial"])

            # Append avoids
            if avoids:
                avoids_clause = f" AVOID: {', '.join(avoids)}. Do NOT include these elements."
                style_prompt = style_prompt.rstrip('.') + '.' + avoids_clause

            print(f"[StyleAgent] Selected with aesthetic: {style_name}")
            return {
                "style_name": f"{style_name} (Aesthetic)",
                "style_key": style_key,
                "style_prompt": style_prompt
            }

        except Exception as e:
            print(f"[StyleAgent] Error with aesthetic: {e}, falling back to regular detection")
            return self.detect(prompt)

    def get_preset(self, style_key: str) -> str:
        """Get style preset text by key."""
        return self.presets.get(style_key, self.presets["editorial"])

    def list_styles(self) -> list:
        """List all available style names."""
        return list(self.presets.keys())


# ============================================
# AGENT 2: PromptAgent - Rewrite/enhance prompts
# ============================================
class PromptAgent:
    """Agent for parsing and rewriting prompts."""

    def __init__(self, style_agent: StyleAgent = None):
        self.client = client
        self.model = MODEL_REWRITE
        self.style_agent = style_agent or StyleAgent()

    def parse_mentions(self, prompt: str) -> list:
        """Parse @mentions from a prompt."""
        mentions = re.findall(r'@(\w+)', prompt)
        seen = set()
        unique = []
        for m in mentions:
            if m.lower() not in seen:
                seen.add(m.lower())
                unique.append(m)
        return unique

    def remove_mentions(self, prompt: str) -> str:
        """Remove @mentions, keeping just asset names."""
        return re.sub(r'@(\w+)', r'\1', prompt)

    def describe_image(self, image_bytes: bytes) -> str:
        """AI describes the attached image."""
        pil_image = Image.open(io.BytesIO(image_bytes))
        describe_instruction = "Describe this person/subject in detail. Include their appearance, what they're wearing, their features, setting, etc. Be concise but descriptive (1-2 sentences)."

        response = self.client.models.generate_content(
            model=self.model,
            contents=[describe_instruction, pil_image]
        )
        return response.text.strip()

    def _rewrite_call(self, original_prompt: str, image_description: str, asset_names: list = None) -> str:
        """Internal helper: Rewrite the prompt based on image description."""
        if asset_names is None:
            asset_names = []

        asset_instruction = ""
        if asset_names:
            asset_list = ", ".join(asset_names)
            asset_instruction = f"""
ASSET REFERENCES:
The following assets are defined in the Subjects section: {asset_list}
When referencing these assets, use "the [assetName]" format (e.g., "the wetdress", "the demi").
This creates a proper reference to the defined asset."""

        rewrite_instruction = f"""The user wants to edit an image. Their original prompt is:
"{original_prompt}"

For reference, the attached image shows:
{image_description}
{asset_instruction}

Your task: Rewrite ONLY the user's prompt to fix grammar.

IMPORTANT RULES:
- Keep the user's ORIGINAL INTENT - what they want to DO/CREATE
- DO NOT describe what's in the image
- DO NOT repeat the subject description
- Just clean up grammar and make it natural
- Keep it SHORT and simple
- When referencing defined assets, use "the [assetName]" (e.g., "the wetdress" not "a wet dress")

Examples:
- "I'm sleeping with a sheep" -> "I am sleeping with a sheep"
- "me driving car" -> "I am driving a car"
- "I wearing @redDress" -> "I am wearing the redDress"
- "me and @demi at beach" -> "Demi and I are at the beach"

Rewritten prompt (just the action, nothing else):"""

        response = self.client.models.generate_content(
            model=self.model,
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

    def rewrite(self, original_prompt: str, image_bytes: bytes, mentioned_assets: list = None, aesthetic: dict = None) -> dict:
        """
        Full prompt enhancement pipeline.
        Runs style detection in parallel with prompt rewriting.

        Args:
            original_prompt: User's original prompt
            image_bytes: Base image bytes
            mentioned_assets: List of asset dicts with name, description, image_bytes
            aesthetic: Optional dict with visual_style and avoids keys
                Example: {"visual_style": "35mm film grain", "avoids": ["neon", "cartoon"]}

        Returns:
            dict with original_prompt, description, rewritten_prompt,
            structured_output, style_name, style_prompt, all_images
        """
        if mentioned_assets is None:
            mentioned_assets = []

        try:
            # Step 1: Describe the attached image
            print("[PromptAgent] Describing attached image...")
            image_description = self.describe_image(image_bytes)
            print(f"[PromptAgent] Image description: {image_description}")

            # Build subjects list - base image first
            subjects = [f"1. I/myself: {image_description} (attached image 1)"]
            all_images = [image_bytes]

            # Add mentioned assets
            for i, asset in enumerate(mentioned_assets, start=2):
                asset_name = asset.get("name", f"asset{i}")
                asset_desc = asset.get("description", f"An asset named {asset_name}")
                subjects.append(f"{i}. {asset_name}: {asset_desc} (attached image {i})")
                all_images.append(asset.get("image_bytes", b""))

            description_text = "\n".join(subjects)
            print(f"[PromptAgent] Subjects:\n{description_text}")

            # Step 2: Run rewrite and style detection in PARALLEL
            clean_prompt = self.remove_mentions(original_prompt)
            asset_names = [asset.get("name", "") for asset in mentioned_assets if asset.get("name")]

            print("[PromptAgent] Running rewrite + style detection in parallel...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                rewrite_future = executor.submit(self._rewrite_call, clean_prompt, image_description, asset_names)
                # Use aesthetic-aware detection if aesthetic provided
                if aesthetic and aesthetic.get("visual_style"):
                    style_future = executor.submit(self.style_agent.detect_with_aesthetic, original_prompt, aesthetic)
                else:
                    style_future = executor.submit(self.style_agent.detect, original_prompt)

                rewritten = rewrite_future.result()
                style_result = style_future.result()

            print(f"[PromptAgent] Original: {original_prompt}")
            print(f"[PromptAgent] Rewritten: {rewritten}")
            print(f"[PromptAgent] Style: {style_result['style_name']}")

            # Format structured output
            structured_output = "Subjects:\n"
            structured_output += description_text
            structured_output += f"\n\nEditing prompt: {rewritten}"
            structured_output += f"\n\nAesthetics/Style: {style_result['style_prompt']}"

            # Combine ALL parts for image generation
            final_prompt = f"Subjects:\n{description_text}\n\nEditing prompt: {rewritten}\n\n{style_result['style_prompt']}"

            return {
                "original_prompt": original_prompt,
                "description": description_text,
                "rewritten_prompt": final_prompt,
                "structured_output": structured_output,
                "style_name": style_result["style_name"],
                "style_prompt": style_result["style_prompt"],
                "all_images": all_images
            }

        except Exception as e:
            print(f"[PromptAgent] Error: {e}, using original prompt")
            default_style = STYLE_PRESETS["editorial"]
            return {
                "original_prompt": original_prompt,
                "description": "",
                "rewritten_prompt": f"{original_prompt}\n\n{default_style}",
                "structured_output": f"Editing prompt: {original_prompt}\n\nAesthetics/Style: {default_style}",
                "style_name": "Editorial",
                "style_prompt": default_style,
                "all_images": [image_bytes]
            }


# ============================================
# AGENT 3: VariationsAgent - Generate variations
# ============================================
class VariationsAgent:
    """Agent for generating image variations."""

    def __init__(self, generate_fn=None):
        self.client = client
        self.model = MODEL_REWRITE
        self.generate_fn = generate_fn  # Reference to generate_image()

    def create_prompts(self, structured_prompt: str, image_bytes: bytes) -> list:
        """
        AI generates 4 diverse variation prompts.

        Returns:
            List of 4 variation prompt strings
        """
        try:
            print("[VariationsAgent] Generating 4 variation prompts...")

            pil_image = Image.open(io.BytesIO(image_bytes))

            variation_instruction = f"""Look at this generated image and the prompt that created it:

{structured_prompt}

Create 4 DIFFERENT shots for an Instagram carousel - same scene, different angles/poses/moments.

KEEP THE SAME:
- Same people/subjects (identity, face, outfit)
- Same SCENE and ENVIRONMENT (exact same location/setting as the first image)
- Same visual style/aesthetic
- Same story/concept

VARY THESE within the same scene:
1. CAMERA ANGLE: close-up / medium / full body / wide / from above / from below / side angle
2. POSE/ACTION: different body positions, gestures, movements
3. MOMENT: different expressions, interactions, timing
4. COMPOSITION: subject placement in frame

Think like a photographer doing multiple shots at the SAME LOCATION - capturing different angles and moments of the same scene.

Output 4 SHORT prompts (1-2 sentences), numbered 1-4.

Example for "encounter a T-Rex on beach" (all on the SAME beach):
1. Extreme close-up of terrified faces, T-Rex towering behind them
2. Wide shot from behind, two figures facing the massive T-Rex on the sandy beach
3. Low angle looking up at them running, T-Rex in pursuit, sand flying
4. Side profile shot, frozen mid-scream, T-Rex's head entering frame

Example for "dinner date at Beijing night market" (all at the SAME market):
1. Close-up of couple laughing, street food stall lights in background
2. Wide shot of them walking through the crowded market, lanterns overhead
3. Over-shoulder view as they share food, market bustle behind
4. Medium shot sitting at a small table, steam rising from dishes

Your 4 variations (same scene, different shots):"""

            response = self.client.models.generate_content(
                model=self.model,
                contents=[variation_instruction, pil_image]
            )

            result_text = response.text.strip()
            print(f"[VariationsAgent] AI response:\n{result_text}")

            # Parse the 4 prompts - ONLY accept lines starting with 1-4
            lines = result_text.split('\n')
            prompts = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # ONLY accept lines that start with a number 1-4
                # Formats: "1.", "1)", "1:", "1 ", etc.
                if len(line) > 2 and line[0] in '1234':
                    # Check for separator after number
                    if line[1] in '.):- ' or (line[1].isspace()):
                        # Extract the actual prompt text
                        prompt_text = line[2:].strip()
                        # Handle "1. " format (extra space after separator)
                        if prompt_text.startswith(' '):
                            prompt_text = prompt_text.strip()
                        if prompt_text and len(prompt_text) > 5:
                            prompts.append(prompt_text)

            # Ensure we have exactly 4 prompts
            defaults = [
                "Close-up portrait with natural expression, subject filling the frame",
                "Wide cinematic shot from above, subject small in vast environment",
                "Action shot mid-movement, dramatic angle, motion blur",
                "Candid moment from behind or side, looking into the distance"
            ]
            while len(prompts) < 4:
                prompts.append(defaults[len(prompts)])

            prompts = prompts[:4]
            print(f"[VariationsAgent] Parsed prompts: {prompts}")
            return prompts

        except Exception as e:
            print(f"[VariationsAgent] Error generating prompts: {e}")
            return [
                "Close-up portrait with natural expression",
                "Full body shot from a slight distance",
                "Candid moment, looking away from camera",
                "Wide shot with more environment visible"
            ]

    def generate(self, original_selfie_bytes: bytes,
                 generated_image_bytes: bytes,
                 selfie_description: str,
                 variation_prompts: list,
                 aspect_ratio: str = "9:16",
                 image_size: str = "1K",
                 additional_assets: list = None) -> dict:
        """
        Generate 4 variations IN PARALLEL.

        Returns:
            dict with:
                - images: List of 4 generated image bytes
                - full_prompts: List of 4 full structured prompts sent to Gemini
        """
        if not self.generate_fn:
            raise ValueError("generate_fn not set - wire up with generate_image()")

        if additional_assets is None:
            additional_assets = []

        print(f"[VariationsAgent] Generating {len(variation_prompts)} variations in parallel...")

        # Build subjects section once
        subjects = f"Subjects:\n1. I/myself: {selfie_description} (attached image 1)\n"
        subjects += f"2. Style reference: Match the visual style and color grading of this image (attached image 2)\n"

        for i, asset in enumerate(additional_assets, start=3):
            subjects += f"{i}. {asset['name']}: {asset['description']} (attached image {i})\n"

        # Build full prompts list FIRST (before parallel execution)
        full_prompts_list = []
        for variation_prompt in variation_prompts:
            full_prompt = f"{subjects}\nEditing prompt: {variation_prompt}\n\nIMPORTANT: Keep the SAME scene/setting/environment as the reference image (attached image 2). Only vary the camera angle, pose, and moment - the location and atmosphere should match!"
            full_prompts_list.append(full_prompt)

        def generate_single(full_prompt: str) -> bytes:
            print(f"[VariationsAgent] Generating: {full_prompt[200:250]}...")

            all_additional = [generated_image_bytes] + [a['image_bytes'] for a in additional_assets]

            return self.generate_fn(
                image_bytes=original_selfie_bytes,
                prompt=full_prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                additional_images=all_additional
            )

        # Generate all 4 in parallel using full prompts
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(generate_single, full_prompts_list))

        print(f"[VariationsAgent] Generated {len(results)} variations")
        return {
            "images": results,
            "full_prompts": full_prompts_list
        }


# ============== UTILITY FUNCTIONS ==============
def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(b64_string: str) -> bytes:
    """Convert base64 string to image bytes."""
    return base64.b64decode(b64_string)
