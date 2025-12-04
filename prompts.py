"""
Prompt constants for Nova2 Image Creator.
All prompts used for LLM interactions are defined here as template strings.
"""

# ============================================
# STYLE PRESETS
# ============================================
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
# STYLE DETECTION
# ============================================
STYLE_DETECTION_PROMPT = """Analyze this image editing prompt and determine the best aesthetic style category:

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


AESTHETIC_DETECTION_PROMPT = """Given the user's aesthetic preferences, select the best matching style OR create a custom one.

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


# ============================================
# IMAGE DESCRIPTION
# ============================================
DESCRIBE_IMAGE_PROMPT = "Describe this person/subject in detail. Include their appearance, what they're wearing, their features, setting, etc. Be concise but descriptive (1-2 sentences)."

DESCRIBE_PEOPLE_PROMPT = "Describe this person in detail. Include their appearance, facial features, hair, and any distinctive characteristics. Be concise (1-2 sentences)."

DESCRIBE_OUTFIT_PROMPT = "Describe this clothing/outfit in detail. Include the type of garment, color, style, and any distinctive features. Be concise (1-2 sentences)."

DESCRIBE_LOCATION_PROMPT = "Describe this location/setting in detail. Include the environment, atmosphere, and key visual elements. Be concise (1-2 sentences)."

DESCRIBE_STYLE_PROMPT = "Describe the visual style/aesthetic of this image. Include the mood, color palette, lighting, and artistic qualities. Be concise (1-2 sentences)."

DESCRIBE_GENERIC_PROMPT = "Describe this image in detail. Be concise (1-2 sentences)."

# Mapping category to description prompt
CATEGORY_DESCRIPTION_PROMPTS = {
    "people": DESCRIBE_PEOPLE_PROMPT,
    "outfit": DESCRIBE_OUTFIT_PROMPT,
    "location": DESCRIBE_LOCATION_PROMPT,
    "style": DESCRIBE_STYLE_PROMPT,
}


# ============================================
# PROMPT REWRITING
# ============================================
ASSET_REFERENCE_INSTRUCTION = """
ASSET REFERENCES:
The following assets are defined in the Subjects section: {asset_list}
When referencing these assets, use "the [assetName]" format (e.g., "the wetdress", "the demi").
This creates a proper reference to the defined asset."""


REWRITE_PROMPT_INSTRUCTION = """The user wants to edit an image. Their original prompt is:
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


# ============================================
# OUTFIT HANDLING
# ============================================
OUTFIT_ADAPTIVE_INSTRUCTION = """Subject Outfits: ADAPTIVE MODE
- You may creatively adapt, modify, or enhance the subject's clothing to better fit the scene, lighting, and mood
- Maintain the general style/vibe of referenced outfits but adjust details (colors, fit, accessories) as needed
- Prioritize visual coherence with the environment over exact outfit replication
- IMPORTANT: If the user explicitly specifies an outfit in their prompt, respect and use that outfit"""


OUTFIT_STRICT_INSTRUCTION = """Subject Outfits: STRICT PRESERVATION MODE
- You MUST exactly replicate the outfit from the attached reference image(s)
- Do NOT modify colors, patterns, fit, or any clothing details
- The outfit should appear identical to the attachment, only adjusted for pose/angle
- This is a hard requirement - outfit accuracy is critical
- IMPORTANT: If the user explicitly specifies an outfit in their prompt, respect and use that outfit"""


# ============================================
# VARIATION GENERATION
# ============================================
VARIATION_PROMPT_INSTRUCTION = """Look at this generated image and the prompt that created it:

{structured_prompt}

Create 4 DIFFERENT shots for an Instagram carousel - same scene, different angles/poses/moments.

KEEP THE SAME (CRITICAL):
- Same people/subjects (identity, face)
- **OUTFIT MUST BE IDENTICAL** - exact same clothing, colors, patterns, accessories as the first image. Do NOT change any clothing details.
- Same SCENE and ENVIRONMENT (exact same location/setting as the first image)
- Same visual style/aesthetic
- Same story/concept

VARY THESE within the same scene:
1. CAMERA ANGLE: close-up / medium / full body / wide / from above / from below / side angle
2. POSE/ACTION: different body positions, gestures, movements
3. MOMENT: different expressions, interactions, timing
4. COMPOSITION: subject placement in frame

Think like a photographer doing multiple shots at the SAME LOCATION - capturing different angles and moments of the same scene. The subject's outfit should be EXACTLY the same in every shot.

Output 4 SHORT prompts (1-2 sentences), numbered 1-4.

Example for "encounter a T-Rex on beach" (all on the SAME beach, SAME outfit):
1. Extreme close-up of terrified faces, T-Rex towering behind them
2. Wide shot from behind, two figures facing the massive T-Rex on the sandy beach
3. Low angle looking up at them running, T-Rex in pursuit, sand flying
4. Side profile shot, frozen mid-scream, T-Rex's head entering frame

Example for "dinner date at Beijing night market" (all at the SAME market, SAME outfit):
1. Close-up of couple laughing, street food stall lights in background
2. Wide shot of them walking through the crowded market, lanterns overhead
3. Over-shoulder view as they share food, market bustle behind
4. Medium shot sitting at a small table, steam rising from dishes

Your 4 variations (same scene, same outfit, different shots):"""


VARIATION_FULL_PROMPT_TEMPLATE = """{subjects}
Editing prompt: {variation_prompt}

CRITICAL REQUIREMENTS:
- OUTFIT MUST BE IDENTICAL to the reference image (attached image 2) - same clothing, colors, patterns, accessories. Do NOT change any clothing.
- Keep the SAME scene/setting/environment as the reference image
- Only vary camera angle, pose, and moment - location, atmosphere, and outfit should match exactly!"""


VARIATION_DEFAULT_PROMPTS = [
    "Close-up portrait with natural expression, subject filling the frame",
    "Wide cinematic shot from above, subject small in vast environment",
    "Action shot mid-movement, dramatic angle, motion blur",
    "Candid moment from behind or side, looking into the distance"
]


VARIATION_FALLBACK_PROMPTS = [
    "Close-up portrait with natural expression",
    "Full body shot from a slight distance",
    "Candid moment, looking away from camera",
    "Wide shot with more environment visible"
]
