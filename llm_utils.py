"""LLM utility functions for GPT and Gemini API calls."""

import logging
import logging.config
import traceback
import time
import mimetypes
import json
import re
from typing import Optional, List, Dict, Union, Any
import io

from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

import httpx
import asyncio
import os

# Load environment variables BEFORE reading them
load_dotenv()

logger = logging.getLogger("nova")

# Initialize clients - support both GEMINI_API_KEY and GOOGLE_API_KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    gemini_async_client = gemini_client.aio
else:
    gemini_client = None
    gemini_async_client = None
    logger.warning("GEMINI_API_KEY/GOOGLE_API_KEY not set - Gemini functions will not work")



async def http_get_bytes(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 6,
    backoff_factor: float = 0.5,
    timeout: float = 20.0,
) -> bytes:
    """
    Download raw bytes from a URL with retries and exponential backoff.
    """
    return await _request_with_retries(
        "GET",
        url,
        headers=headers,
        timeout=timeout,
        retries=retries,
        backoff_factor=backoff_factor,
        return_content=True,
    )


async def _request_with_retries(
    method: str,
    url: str,
    *,
    # parameters forwarded to client.request
    data: Optional[Dict[str, Any]] = None,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float,
    retries: int,
    backoff_factor: float,
    # post-processing of the response
    return_content: bool = False,
) -> Union[bytes, Any]:
    """
    Generic HTTP request with retries & exponential backoff.
    If return_content is True, returns raw bytes; otherwise returns resp.json().
    """
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.request(
                    method, url, data=data, json=json, headers=headers,
                )
                resp.raise_for_status()
                return resp.content if return_content else resp.json()
        except Exception as e:
            if attempt == retries:
                logger.error(
                    f"{method.upper()} {url} failed after {retries} attempts: {e!r}"
                )
                raise
            delay = backoff_factor * (2 ** (attempt - 1))
            logger.warning(
                f"{method.upper()} attempt {attempt}/{retries} failed: {e!r}. "
                f"Retrying in {delay:.1f}s..."
            )
            await asyncio.sleep(delay)


async def run_gemini(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: str = None,
    model: str = "gemini-2.5-flash-lite",
):
    """
    Run a prompt through Gemini with system and user context.

    Args:
        prompt: User prompt text
        temperature: Temperature for sampling (default: 0.7)
        max_tokens: Maximum tokens in response (default: None/unlimited)
        system_prompt: Optional system instruction
        model: Gemini model to use (default: gemini-2.5-flash-lite)

    Returns:
        str: Generated text response (cleaned)

    Raises:
        Exception: If API call fails or Gemini client not initialized
    """
    if not gemini_client:
        raise Exception("Gemini client not initialized - GEMINI_API_KEY missing")

    try:
        start_time = time.time()
        cfg = types.GenerateContentConfig(
            temperature=float(temperature),
            max_output_tokens=int(max_tokens) if max_tokens is not None else None,
            system_instruction=system_prompt if system_prompt else None,
        )

        response = await gemini_async_client.models.generate_content(
            model=model,
            contents=[prompt],
            config=cfg,
        )

        # Extract and clean text
        text = (getattr(response, "text", None) or "").strip()
        text = " ".join(text.split()).strip().strip('"\u201c\u201d\u2018\u2019')

        end_time = time.time()
        logger.info(f"Gemini response time: {end_time - start_time:.2f} seconds")
        logger.info(f"Gemini response: {text}")

        return text
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Gemini API error: {str(e)}")
        raise e

async def run_gemini_with_image(
    image: Union[str, bytes, Image.Image],
    prompt: str,
    model: str = "gemini-2.5-flash",
):
    """
    Run a prompt through Gemini with an image input.

    Args:
        image: Image as URL string, bytes, or PIL Image
        prompt: User prompt text
        model: Gemini model to use (default: gemini-2.5-flash)

    Returns:
        str: Generated text response (cleaned)

    Raises:
        Exception: If API call fails or Gemini client not initialized
    """
    if not gemini_client:
        raise Exception("Gemini client not initialized - GEMINI_API_KEY missing")

    try:
        start_time = time.time()

        # Handle different image input types
        if isinstance(image, str):
            # URL - download and create Part
            image_data = await http_get_bytes(image)
            mime = mimetypes.guess_type(image)[0] or "image/png"
            img = types.Part.from_bytes(data=image_data, mime_type=mime)
        elif isinstance(image, bytes):
            # Bytes - convert to PIL Image (Gemini SDK accepts PIL directly)
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            # PIL Image - use directly
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        response = await gemini_async_client.models.generate_content(
            model=model,
            contents=[prompt, img],
        )

        # Extract and clean text
        text = (getattr(response, "text", None) or "").strip()
        text = " ".join(text.split()).strip().strip('"\u201c\u201d\u2018\u2019')

        end_time = time.time()
        logger.info(f"Gemini with image response time: {end_time - start_time:.2f} seconds")

        return text
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Gemini API error with image: {str(e)}")
        raise e


async def run_gemini_json(
    prompt: str,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.7,
    parse_as_array: bool = False,
) -> Union[Dict, List]:
    """
    Run a prompt through Gemini and parse JSON response.
    
    Args:
        prompt: User prompt text (should request JSON response)
        model: Gemini model to use (default: gemini-2.5-flash)
        temperature: Temperature for sampling (default: 0.7)
        parse_as_array: If True, expects JSON array; if False, expects JSON object
    
    Returns:
        Dict or List: Parsed JSON response
    
    Raises:
        Exception: If API call fails or JSON parsing fails
    """
    if not gemini_client:
        raise Exception("Gemini client not initialized - GEMINI_API_KEY missing")
    
    try:
        start_time = time.time()
        
        cfg = types.GenerateContentConfig(
            temperature=float(temperature),
        )
        
        response = await gemini_async_client.models.generate_content(
            model=model,
            contents=[prompt],
            config=cfg,
        )
        
        response_text = response.text.strip()
        
        # Clean markdown code fences
        clean_text = re.sub(r'```json\s*|\s*```', '', response_text)
        clean_text = clean_text.strip()
        
        # Remove trailing commas
        clean_text = re.sub(r',(\s*[}\]])', r'\1', clean_text)
        
        # Parse JSON
        if parse_as_array:
            # Try to find JSON array
            json_match = re.search(r'\[[\s\S]*\]', clean_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try aggressive cleanup
                    ultra_clean = re.sub(r',(\s*[}\]])', r'\1', clean_text)
                    ultra_clean = re.sub(r',(\s+,)', ',', ultra_clean)
                    json_match = re.search(r'\[[\s\S]*\]', ultra_clean, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        else:
            # Try to find JSON object
            json_match = re.search(r'\{[\s\S]*\}', clean_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try aggressive cleanup
                    ultra_clean = re.sub(r',(\s*[}\]])', r'\1', clean_text)
                    ultra_clean = re.sub(r',(\s+,)', ',', ultra_clean)
                    json_match = re.search(r'\{[\s\S]*\}', ultra_clean, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
        
        raise ValueError("Could not parse JSON from response")
        
    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Gemini JSON API error: {str(e)}")
        raise e


async def run_gemini_generate_image(
    prompt: str,
    images: List[Image.Image],
    model: str = "gemini-3-pro-image-preview",
    aspect_ratio: str = "9:16",
    image_size: str = "1K",
) -> bytes:
    """
    Generate an image using Gemini with multiple input images.

    Args:
        prompt: Text prompt for image generation
        images: List of PIL Image objects to use as references
        model: Gemini model to use (default: gemini-3-pro-image-preview)
        aspect_ratio: Image aspect ratio (default: 9:16)
        image_size: Resolution (1K, 2K, 4K) - default: 1K

    Returns:
        bytes: Generated image data (raw bytes from API)

    Raises:
        Exception: If API call fails or no image generated
    """
    if not gemini_client:
        raise Exception("Gemini client not initialized - GEMINI_API_KEY missing")

    try:
        start_time = time.time()

        # Build contents array: prompt + images
        contents = [prompt] + images

        # Call Gemini API
        response = await gemini_async_client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
                image_config=types.ImageConfig(
                    aspectRatio=aspect_ratio,
                    imageSize=image_size
                )
            )
        )

        # Extract image from response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                end_time = time.time()
                logger.info(f"Gemini image generation time: {end_time - start_time:.2f} seconds")
                return part.inline_data.data

        raise ValueError("No image generated in response")

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(f"Gemini image generation error: {str(e)}")
        raise e