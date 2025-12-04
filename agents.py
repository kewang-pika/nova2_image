"""
Nova2 Image Creator - Agent Classes
StyleAgent, PromptAgent, VariationsAgent
"""
import os
import io
import re
import json
import base64
import asyncio
from PIL import Image
from dotenv import load_dotenv

# Import llm_utils functions
from llm_utils import run_gemini, run_gemini_json, run_gemini_with_image

# Import prompts from centralized prompts.py
from prompts import (
    STYLE_PRESETS,
    STYLE_CATEGORIES,
    STYLE_DETECTION_PROMPT,
    AESTHETIC_DETECTION_PROMPT,
    DESCRIBE_IMAGE_PROMPT,
    ASSET_REFERENCE_INSTRUCTION,
    REWRITE_PROMPT_INSTRUCTION,
    OUTFIT_ADAPTIVE_INSTRUCTION,
    OUTFIT_STRICT_INSTRUCTION,
    VARIATION_PROMPT_INSTRUCTION,
    VARIATION_FULL_PROMPT_TEMPLATE,
    VARIATION_DEFAULT_PROMPTS,
    VARIATION_FALLBACK_PROMPTS,
    VARIATION_DEFAULT_STRUCTURED,
    VARIATION_FALLBACK_STRUCTURED,
)

load_dotenv()

# ============== CONFIG ==============
MODEL_REWRITE = "gemini-2.5-flash"

# Backward compatibility alias
DEFAULT_STYLE_SYSTEM_PROMPT = STYLE_DETECTION_PROMPT


# ============================================
# AGENT 1: StyleAgent - Detect/determine style
# ============================================


class StyleAgent:
    """Agent for detecting and selecting style presets."""

    def __init__(self, system_prompt: str = None):
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

    async def detect(self, prompt: str) -> dict:
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

            # Use run_gemini_json for JSON output
            result = await run_gemini_json(
                prompt=style_instruction,
                model=self.model,
                temperature=0.7
            )

            print(f"[StyleAgent] AI response: {result}")

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

    async def detect_with_aesthetic(self, prompt: str, aesthetic: dict = None) -> dict:
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
            return await self.detect(prompt)

        visual_style = aesthetic.get("visual_style", "")
        avoids = aesthetic.get("avoids", [])

        print(f"[StyleAgent] Using aesthetic preferences:")
        print(f"[StyleAgent]   Visual style: {visual_style}")
        print(f"[StyleAgent]   Avoids: {avoids}")

        try:
            # Build aesthetic-aware instruction using template from prompts.py
            avoids_str = ", ".join(avoids) if avoids else "none specified"
            aesthetic_instruction = AESTHETIC_DETECTION_PROMPT.format(
                prompt=prompt,
                visual_style=visual_style,
                avoids_str=avoids_str
            )

            # Use run_gemini_json for JSON output
            result = await run_gemini_json(
                prompt=aesthetic_instruction,
                model=self.model,
                temperature=0.7
            )

            print(f"[StyleAgent] AI response: {result}")

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
            return await self.detect(prompt)

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

    async def describe_image(self, image_bytes: bytes) -> str:
        """AI describes the attached image."""
        # Use run_gemini_with_image for text output with image input
        result = await run_gemini_with_image(
            image=image_bytes,
            prompt=DESCRIBE_IMAGE_PROMPT,
            model=self.model
        )
        return result

    async def _rewrite_call(self, original_prompt: str, image_description: str, asset_names: list = None) -> str:
        """Internal helper: Rewrite the prompt based on image description."""
        if asset_names is None:
            asset_names = []

        # Build asset instruction using template from prompts.py
        asset_instruction = ""
        if asset_names:
            asset_list = ", ".join(asset_names)
            asset_instruction = ASSET_REFERENCE_INSTRUCTION.format(asset_list=asset_list)

        # Build rewrite instruction using template from prompts.py
        rewrite_instruction = REWRITE_PROMPT_INSTRUCTION.format(
            original_prompt=original_prompt,
            image_description=image_description,
            asset_instruction=asset_instruction
        )

        # Use run_gemini for text output
        rewritten = await run_gemini(
            prompt=rewrite_instruction,
            model=self.model,
            temperature=0.7
        )

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

    async def rewrite(self, original_prompt: str, image_bytes: bytes, mentioned_assets: list = None, aesthetic: dict = None, outfit_adaptive: bool = True) -> dict:
        """
        Full prompt enhancement pipeline.
        Runs style detection in parallel with prompt rewriting.

        Args:
            original_prompt: User's original prompt
            image_bytes: Base image bytes
            mentioned_assets: List of asset dicts with name, description, image_bytes
            aesthetic: Optional dict with visual_style and avoids keys
                Example: {"visual_style": "35mm film grain", "avoids": ["neon", "cartoon"]}
            outfit_adaptive: If True (default), AI can adapt outfits to scene.
                If False, strictly preserve outfits from attachments.

        Returns:
            dict with original_prompt, description, rewritten_prompt,
            structured_output, style_name, style_prompt, all_images
        """
        if mentioned_assets is None:
            mentioned_assets = []

        try:
            # Step 1: Describe the attached image
            print("[PromptAgent] Describing attached image...")
            image_description = await self.describe_image(image_bytes)
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

            # Step 2: Run rewrite and style detection in PARALLEL using asyncio.gather
            clean_prompt = self.remove_mentions(original_prompt)
            asset_names = [asset.get("name", "") for asset in mentioned_assets if asset.get("name")]

            print("[PromptAgent] Running rewrite + style detection in parallel...")

            # Use aesthetic-aware detection if aesthetic provided
            if aesthetic and aesthetic.get("visual_style"):
                style_coro = self.style_agent.detect_with_aesthetic(original_prompt, aesthetic)
            else:
                style_coro = self.style_agent.detect(original_prompt)

            # Run both in parallel
            rewritten, style_result = await asyncio.gather(
                self._rewrite_call(clean_prompt, image_description, asset_names),
                style_coro
            )

            print(f"[PromptAgent] Original: {original_prompt}")
            print(f"[PromptAgent] Rewritten: {rewritten}")
            print(f"[PromptAgent] Style: {style_result['style_name']}")
            print(f"[PromptAgent] Outfit mode: {'ADAPTIVE' if outfit_adaptive else 'STRICT'}")

            # Build outfit instruction based on mode using constants from prompts.py
            outfit_instruction = OUTFIT_ADAPTIVE_INSTRUCTION if outfit_adaptive else OUTFIT_STRICT_INSTRUCTION

            # Format structured output with outfit instruction
            structured_output = "Subjects:\n"
            structured_output += description_text
            structured_output += f"\n\n{outfit_instruction}"
            structured_output += f"\n\nEditing prompt: {rewritten}"
            structured_output += f"\n\nAesthetics/Style: {style_result['style_prompt']}"

            # Combine ALL parts for image generation
            final_prompt = f"Subjects:\n{description_text}\n\n{outfit_instruction}\n\nEditing prompt: {rewritten}\n\n{style_result['style_prompt']}"

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
        self.model = MODEL_REWRITE
        self.generate_fn = generate_fn  # Reference to generate_image()

    async def create_prompts(self, structured_prompt: str, image_bytes: bytes) -> list:
        """
        AI generates 4 diverse variation prompts with subject subsets.

        Returns:
            List of 4 dicts: [{"subjects": [...], "editing_prompt": "..."}, ...]
        """
        try:
            print("[VariationsAgent] Generating 4 variation prompts...")

            # Build variation instruction using template from prompts.py
            variation_instruction = VARIATION_PROMPT_INSTRUCTION.format(
                structured_prompt=structured_prompt
            )

            # Use run_gemini_with_image for text output with image input
            result_text = await run_gemini_with_image(
                image=image_bytes,
                prompt=variation_instruction,
                model=self.model
            )

            print(f"[VariationsAgent] AI response:\n{result_text}")

            # Parse JSON response
            # Clean markdown code fences
            clean_text = re.sub(r'```json\s*|\s*```', '', result_text).strip()

            # Find JSON array
            json_match = re.search(r'\[[\s\S]*\]', clean_text)
            if json_match:
                try:
                    variations = json.loads(json_match.group())

                    # Validate structure
                    valid_variations = []
                    for v in variations:
                        if isinstance(v, dict) and "subjects" in v and "editing_prompt" in v:
                            # Normalize subjects to list
                            subjects = v["subjects"]
                            if isinstance(subjects, str):
                                subjects = [subjects]
                            valid_variations.append({
                                "subjects": subjects,
                                "editing_prompt": v["editing_prompt"]
                            })

                    if len(valid_variations) >= 4:
                        print(f"[VariationsAgent] Parsed {len(valid_variations)} structured prompts")
                        return valid_variations[:4]
                    else:
                        print(f"[VariationsAgent] Only got {len(valid_variations)} valid variations, using defaults")

                except json.JSONDecodeError as e:
                    print(f"[VariationsAgent] JSON parse error: {e}")

            # Fallback to structured defaults
            print("[VariationsAgent] Using default structured prompts")
            return VARIATION_DEFAULT_STRUCTURED

        except Exception as e:
            print(f"[VariationsAgent] Error generating prompts: {e}")
            return VARIATION_FALLBACK_STRUCTURED

    async def generate(self, original_selfie_bytes: bytes,
                 generated_image_bytes: bytes,
                 selfie_description: str,
                 variation_prompts: list,
                 aspect_ratio: str = "9:16",
                 image_size: str = "1K",
                 additional_assets: list = None) -> dict:
        """
        Generate 4 variations IN PARALLEL with dynamic subject filtering.

        Args:
            variation_prompts: List of dicts with {"subjects": [...], "editing_prompt": "..."}
                - subjects can include: "I", person names, "scene" (no people), "POV" (first-person)

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

        # Build asset name -> data mapping (case-insensitive)
        asset_map = {asset['name'].lower(): asset for asset in additional_assets}

        # Prepare prompts and image configs for each variation
        full_prompts_list = []
        image_configs = []  # Store which additional images to use per variation
        use_selfie_flags = []  # Track whether to use selfie as base image

        for variation in variation_prompts:
            selected_subjects = variation.get("subjects", ["I"])
            editing_prompt = variation.get("editing_prompt", "")

            # Normalize subject names to lowercase for matching
            selected_lower = [s.lower() for s in selected_subjects]

            # Check for special variation types
            is_scene_only = "scene" in selected_lower
            is_pov = "pov" in selected_lower

            # For POV/scene: check if other subjects are also included (e.g., ["POV", "demi"])
            other_subjects = [s for s in selected_lower if s not in ["i", "myself", "scene", "pov"]]
            has_i = "i" in selected_lower or "myself" in selected_lower

            # Build subjects/reference section
            subjects_text = "Subjects/Reference:\n"
            img_idx = 1
            images_for_this_variation = []
            use_selfie = True  # Default: use selfie as base image

            if is_scene_only and not other_subjects and not has_i:
                # Pure scene shot: only style reference, no people
                subjects_text += f"{img_idx}. Style reference: Match the visual style of this image (attached image {img_idx})\n"
                img_idx += 1
                use_selfie = False
            elif is_pov:
                # POV shot: first-person perspective
                # Style reference first
                subjects_text += f"{img_idx}. Style reference: Match the visual style of this image (attached image {img_idx})\n"
                img_idx += 1
                subjects_text += "Note: FIRST-PERSON POV - show what the subject sees, not the subject themselves\n"
                use_selfie = False

                # But if other people are visible in POV (e.g., ["POV", "I"] or ["POV", "demi"]), include them!
                if has_i:
                    subjects_text += f"{img_idx}. I/myself: {selfie_description} (attached image {img_idx})\n"
                    images_for_this_variation.append(original_selfie_bytes)
                    img_idx += 1

                # Add other visible subjects in POV
                for subject_name in selected_subjects:
                    subject_lower = subject_name.lower()
                    if subject_lower not in ["i", "myself", "scene", "pov"] and subject_lower in asset_map:
                        asset = asset_map[subject_lower]
                        subjects_text += f"{img_idx}. {asset['name']}: {asset['description']} (attached image {img_idx})\n"
                        images_for_this_variation.append(asset['image_bytes'])
                        img_idx += 1
            else:
                # Normal variation with people
                # Include "I/myself" if in selected subjects
                if has_i:
                    subjects_text += f"{img_idx}. I/myself: {selfie_description} (attached image {img_idx})\n"
                    img_idx += 1

                # Style reference
                subjects_text += f"{img_idx}. Style reference: Match the visual style of this image (attached image {img_idx})\n"
                img_idx += 1

                # Add ALL selected assets - if subject is explicitly in the subjects array, include it!
                for subject_name in selected_subjects:
                    subject_lower = subject_name.lower()
                    if subject_lower not in ["i", "myself", "scene", "pov"] and subject_lower in asset_map:
                        asset = asset_map[subject_lower]
                        # Always include explicitly selected subjects
                        subjects_text += f"{img_idx}. {asset['name']}: {asset['description']} (attached image {img_idx})\n"
                        images_for_this_variation.append(asset['image_bytes'])
                        img_idx += 1

            # Build full prompt
            full_prompt = VARIATION_FULL_PROMPT_TEMPLATE.format(
                subjects=subjects_text,
                variation_prompt=editing_prompt
            )
            full_prompts_list.append(full_prompt)
            image_configs.append(images_for_this_variation)
            use_selfie_flags.append(use_selfie)

            print(f"[VariationsAgent] Variation subjects: {selected_subjects}, use_selfie: {use_selfie}, prompt: {editing_prompt[:50]}...")

        # Generate with filtered images per variation
        async def generate_single(idx: int) -> bytes:
            full_prompt = full_prompts_list[idx]
            use_selfie = use_selfie_flags[idx]

            if use_selfie:
                # Normal: selfie as base, style ref + assets as additional
                base_image = original_selfie_bytes
                additional = [generated_image_bytes] + image_configs[idx]
            else:
                # Scene/POV: generated image as base, only assets as additional
                base_image = generated_image_bytes
                additional = image_configs[idx] if image_configs[idx] else None

            return await self.generate_fn(
                image_bytes=base_image,
                prompt=full_prompt,
                aspect_ratio=aspect_ratio,
                image_size=image_size,
                additional_images=additional
            )

        # Generate all 4 in parallel using asyncio.gather
        results = await asyncio.gather(*[generate_single(i) for i in range(len(variation_prompts))])

        print(f"[VariationsAgent] Generated {len(results)} variations")
        return {
            "images": list(results),
            "full_prompts": full_prompts_list
        }


# ============== UTILITY FUNCTIONS ==============
def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def base64_to_image(b64_string: str) -> bytes:
    """Convert base64 string to image bytes."""
    return base64.b64decode(b64_string)
