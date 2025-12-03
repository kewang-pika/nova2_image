# Nova2 Image Creator

AI-powered image editing using Google Gemini's image generation model.

## Features

- **Image Editing**: Transform photos with natural language prompts
- **Style Detection**: Auto-detects aesthetic style from prompt (Instagram, Editorial, Cinematic, etc.)
- **Asset Closet**: Save and reuse people, outfits, locations with @mentions
- **4 Variations**: Generate diverse shots of the same scene in parallel
- **Aesthetic Preferences**: Paste JSON to customize visual style and avoid certain elements
- **History Gallery**: Password-protected gallery of all generations

## Quick Start

```bash
# Install dependencies
pip install streamlit pillow google-genai python-dotenv

# Set up API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# Run
streamlit run streamlit_app.py
```

## Usage

1. Upload a base photo
2. Type a prompt (e.g., "I'm at the beach at sunset")
3. Use `@assetName` to reference saved assets
4. Click Generate
5. Optionally generate 4 variations

## Aesthetic JSON (Optional)

Paste in sidebar to customize style:
```json
{
  "aesthetic": {
    "visual_style": "35mm film grain, muted colors, natural light",
    "avoids": ["neon", "high contrast", "cartoon"]
  }
}
```

## Project Structure

```
nova2_image/
├── streamlit_app.py    # Web UI
├── image_creator.py    # Main module
├── agents.py           # StyleAgent, PromptAgent, VariationsAgent
├── closet.py           # Asset management
├── history.py          # Generation history
└── .env                # API key
```

## History Password

Default: `nova2image`
