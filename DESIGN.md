# Nova2 Image Creator - System Design

## Overview

Nova2 Image Creator is an AI-powered image editing system built on Google Gemini's image generation capabilities. The system uses a **multi-agent architecture** where specialized agents handle different aspects of the image generation pipeline.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Upload   │  │ Prompt   │  │ Settings │  │ History Gallery  │    │
│  │ Image    │  │ Input    │  │ Sidebar  │  │ (Password Prot.) │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      IMAGE_CREATOR.PY (Orchestrator)                │
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │ Config      │   │ Convenience │   │ Core        │              │
│   │ Management  │   │ Wrappers    │   │ Generation  │              │
│   └─────────────┘   └─────────────┘   └─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENTS.PY                                   │
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │ StyleAgent  │   │ PromptAgent │   │ Variations  │              │
│   │             │◄──│             │   │ Agent       │              │
│   │ detect()    │   │ rewrite()   │   │ generate()  │              │
│   └─────────────┘   └─────────────┘   └─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SUPPORTING MODULES                             │
│                                                                     │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │
│   │ closet.py   │   │ history.py  │   │ .env        │              │
│   │ Asset Mgmt  │   │ Gallery     │   │ API Keys    │              │
│   └─────────────┘   └─────────────┘   └─────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GOOGLE GEMINI API                              │
│                                                                     │
│   ┌────────────────────────┐   ┌────────────────────────┐          │
│   │ gemini-3-pro-image     │   │ gemini-2.5-flash       │          │
│   │ (Image Generation)     │   │ (Text/Analysis)        │          │
│   └────────────────────────┘   └────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Three-Agent Architecture

### Agent 1: StyleAgent

**Purpose:** Detect and determine the visual style/aesthetic for image generation.

```
┌─────────────────────────────────────────────────────┐
│                    StyleAgent                        │
├─────────────────────────────────────────────────────┤
│ Attributes:                                          │
│   - system_prompt (configurable)                     │
│   - presets: 14 style templates                      │
│   - categories: keyword mappings                     │
├─────────────────────────────────────────────────────┤
│ Methods:                                             │
│   detect(prompt) → {style_name, style_key,          │
│                     style_prompt}                    │
│   detect_with_aesthetic(prompt, aesthetic) → dict   │
│   get_preset(style_key) → str                       │
│   set_system_prompt(prompt)                         │
│   get_system_prompt() → str                         │
└─────────────────────────────────────────────────────┘
```

**Available Style Presets:**
| Key | Description |
|-----|-------------|
| `instagram_v1/v2/v3` | Social media, lifestyle, influencer |
| `pastel` | Soft, dreamy, pastel colors |
| `fuji_sunglow` | Fujifilm-inspired, warm film grain |
| `editorial` | Professional photography (default) |
| `cool_cinematic` | Cinematic, cool tones, muted |
| `fashion_v1/v2` | High-fashion, runway, editorial |
| `ccd_flash` | Night/club, retro 2000s flash |
| `landscape` | Nature, outdoor, scenic |
| `celebrity` | Paparazzi, red carpet style |
| `mirror_selfie` | Mirror selfies, confident poses |

**Style Detection Flow:**
```
User Prompt
     │
     ▼
┌────────────────┐     ┌────────────────┐
│ Has aesthetic  │ Yes │ detect_with_   │
│ JSON input?    │────▶│ aesthetic()    │
└────────────────┘     └────────────────┘
     │ No                     │
     ▼                        ▼
┌────────────────┐     ┌────────────────┐
│ detect()       │     │ Match preset   │
│ AI analysis    │     │ OR create      │
└────────────────┘     │ custom style   │
     │                 └────────────────┘
     ▼                        │
┌────────────────────────────┐│
│ Return: style_name,        │◀
│ style_key, style_prompt    │
└────────────────────────────┘
```

---

### Agent 2: PromptAgent

**Purpose:** Parse, enhance, and structure user prompts for optimal image generation.

```
┌─────────────────────────────────────────────────────┐
│                    PromptAgent                       │
├─────────────────────────────────────────────────────┤
│ Attributes:                                          │
│   - style_agent: Reference to StyleAgent            │
│   - model: gemini-2.5-flash                         │
├─────────────────────────────────────────────────────┤
│ Methods:                                             │
│   parse_mentions(prompt) → list of @mentions        │
│   remove_mentions(prompt) → clean string            │
│   describe_image(image_bytes) → description         │
│   rewrite(prompt, image, assets, aesthetic) → dict  │
└─────────────────────────────────────────────────────┘
```

**Prompt Enhancement Pipeline:**
```
                    User Input
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: Parse @mentions                              │
│ "@outfit1 @beach" → ["outfit1", "beach"]            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: Describe base image (AI)                     │
│ "Person wearing blue dress, brown hair..."          │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 3: PARALLEL EXECUTION                           │
│  ┌──────────────┐    ┌──────────────┐               │
│  │ Rewrite      │    │ Style        │               │
│  │ Prompt (AI)  │    │ Detection    │               │
│  └──────────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 4: Build Structured Output                      │
│                                                      │
│ Subjects:                                            │
│ 1. I/myself: [description] (attached image 1)       │
│ 2. outfit1: [description] (attached image 2)        │
│                                                      │
│ Editing prompt: [enhanced prompt]                   │
│                                                      │
│ Aesthetics/Style: [style preset text]               │
└─────────────────────────────────────────────────────┘
```

**Return Structure:**
```python
{
    "original_prompt": "I'm at the beach",
    "description": "Person description...",
    "rewritten_prompt": "Standing on sandy beach...",
    "structured_output": "Full structured prompt...",
    "style_name": "Instagram V3",
    "style_prompt": "Style instructions...",
    "all_images": [base_image, asset1, asset2...]
}
```

**Structured Prompt Format:**
```
Subjects:
1. I/myself: [description] (attached image 1)
2. outfit1: [description] (attached image 2)

Subject Outfits: [ADAPTIVE or STRICT MODE]
- ADAPTIVE: AI can modify outfit to fit scene
- STRICT: Exact replication of attached outfit

Editing prompt: [enhanced user prompt]

Aesthetics/Style: [style preset text]
```

---

### Agent 3: VariationsAgent

**Purpose:** Generate diverse variations of the initial generated image.

```
┌─────────────────────────────────────────────────────┐
│                  VariationsAgent                     │
├─────────────────────────────────────────────────────┤
│ Attributes:                                          │
│   - generate_fn: Reference to generate_image()      │
│   - model: gemini-2.5-flash                         │
├─────────────────────────────────────────────────────┤
│ Methods:                                             │
│   create_prompts(structured_prompt, image) → list   │
│   generate(selfie, generated, desc, prompts,        │
│            aspect, size, assets) → dict             │
└─────────────────────────────────────────────────────┘
```

**Variation Generation Flow:**
```
┌─────────────────────────────────────────────────────┐
│ Input: Generated image + Original structured prompt  │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 1: AI generates 4 variation prompts             │
│                                                      │
│ "Generate 4 diverse variations. Keep same:          │
│  - Person identity, outfit, environment, style      │
│  Create different:                                   │
│  - Camera angle, pose, composition, moment"         │
│                                                      │
│ Output:                                              │
│   1. "Close-up portrait, looking over shoulder"     │
│   2. "Full body shot, walking toward camera"        │
│   3. "Side profile, wind in hair"                   │
│   4. "Candid moment, laughing naturally"            │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│ Step 2: PARALLEL IMAGE GENERATION (4 threads)        │
│                                                      │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ Var 1  │ │ Var 2  │ │ Var 3  │ │ Var 4  │       │
│  │generate│ │generate│ │generate│ │generate│       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
│       │         │         │         │               │
│       └─────────┴─────────┴─────────┘               │
│                    │                                 │
│                    ▼                                 │
│          Return: 4 image bytes                      │
└─────────────────────────────────────────────────────┘
```

**Return Structure:**
```python
{
    "images": [bytes, bytes, bytes, bytes],  # 4 variations
    "full_prompts": [str, str, str, str]     # Full prompts used
}
```

---

## Agent Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE GENERATION FLOW                        │
└─────────────────────────────────────────────────────────────────────┘

User uploads image + enters prompt + (optional) aesthetic JSON
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. PromptAgent.parse_mentions()                                     │
│    Extract @asset references from prompt                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. Load assets from Closet                                          │
│    Resolve @mentions to actual images + descriptions                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. PromptAgent.rewrite()                                            │
│    ├─ describe_image() → AI describes the base selfie               │
│    ├─ [PARALLEL]                                                    │
│    │   ├─ _rewrite_call() → Grammar fix + enhance                  │
│    │   └─ StyleAgent.detect() OR detect_with_aesthetic()           │
│    └─ Build structured prompt with subjects + style                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. generate_image()                                                 │
│    Send to Gemini gemini-3-pro-image-preview with:                  │
│    - Base image                                                     │
│    - Additional asset images                                        │
│    - Structured prompt                                              │
│    - Aspect ratio + image size                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. history.save_generation()                                        │
│    Save: prompts, style, image, settings                            │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. [OPTIONAL] Generate Variations                                   │
│    ├─ VariationsAgent.create_prompts() → 4 diverse ideas           │
│    ├─ VariationsAgent.generate() → [PARALLEL] 4 images             │
│    └─ history.update_generation_variations()                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Current Features

### 1. Aesthetic JSON Import

Users can paste JSON to influence style detection:

```json
{
  "aesthetic": {
    "visual_style": "Soft editorial, 35mm film grain, muted colors, natural light",
    "avoids": ["neon", "high contrast", "heavy makeup", "3d render", "cartoon"]
  }
}
```

**Processing:**
1. If `visual_style` matches a preset → Use that preset + append AVOID clause
2. If no match → Create custom style prompt incorporating preferences
3. `avoids` list is always appended as explicit exclusions

---

### 2. System Prompt Editing

The StyleAgent's system prompt is configurable via the UI:

```
┌─────────────────────────────────────────────────────┐
│ agent_config.json                                    │
│ {                                                    │
│   "style_agent_system_prompt": "custom prompt..."   │
│ }                                                    │
└─────────────────────────────────────────────────────┘
            │                           ▲
            ▼                           │
┌─────────────────────────────────────────────────────┐
│ StyleAgent                                           │
│   self.system_prompt = loaded OR default            │
└─────────────────────────────────────────────────────┘
```

**Functions:**
- `get_style_system_prompt()` → Get current prompt
- `save_style_system_prompt(prompt)` → Save to config file
- `reset_style_system_prompt()` → Restore default

---

### 3. Generation History Gallery

Password-protected gallery storing all generations:

```
history/
├── 20241201_143052_123456/
│   ├── metadata.json
│   ├── main.jpg
│   ├── variation_1.jpg
│   ├── variation_2.jpg
│   ├── variation_3.jpg
│   └── variation_4.jpg
└── 20241201_150030_789012/
    ├── metadata.json
    └── main.jpg
```

**metadata.json Structure:**
```json
{
  "id": "20241201_143052_123456",
  "timestamp": "2024-12-01T14:30:52.123456",
  "original_prompt": "I'm at the beach",
  "editing_prompt": "Standing on sandy beach...",
  "structured_prompt": "Full structured prompt...",
  "style_name": "Instagram V3",
  "aspect_ratio": "9:16",
  "image_size": "1K",
  "main_image": "main.jpg",
  "variations": ["variation_1.jpg", ...],
  "variation_prompts": ["Close-up...", ...],
  "variation_full_prompts": ["Full prompt 1...", ...],
  "aesthetic": {"visual_style": "...", "avoids": [...]}
}
```

**Gallery Features:**
- Password: `nova2image`
- Grid view with thumbnails
- Detail view with full prompts
- Delete functionality

---

## File Structure

```
nova2_image/
├── streamlit_app.py      # Web UI (Streamlit)
├── image_creator.py      # Orchestrator + Core Generation
├── agents.py             # 3 Agent Classes + Style Presets
├── closet.py             # Asset Management (@mentions)
├── history.py            # Generation History
├── agent_config.json     # User settings (system prompt)
├── .env                  # API keys
├── generated_images/     # Output directory
├── closet/               # Saved assets
│   ├── people/
│   ├── outfit/
│   ├── location/
│   └── style/
└── history/              # Generation history
```

---

## API Models Used

| Model | Purpose | Used By |
|-------|---------|---------|
| `gemini-3-pro-image-preview` | Image generation/editing | `generate_image()` |
| `gemini-2.5-flash` | Text analysis, prompts | All 3 agents |

---

## Parallelization Strategy

The system uses `ThreadPoolExecutor` for parallel operations:

1. **PromptAgent.rewrite()**: Rewrite + Style detection run in parallel (2 threads)
2. **VariationsAgent.generate()**: All 4 variations generated in parallel (4 threads)

```python
# Example: Parallel style detection + prompt rewrite
with ThreadPoolExecutor(max_workers=2) as executor:
    rewrite_future = executor.submit(self._rewrite_call, ...)
    style_future = executor.submit(self.style_agent.detect, ...)

    rewritten = rewrite_future.result()
    style_result = style_future.result()
```

---

## Configuration

### Environment Variables (.env)
```
GOOGLE_API_KEY=your_gemini_api_key
```

### Agent Config (agent_config.json)
```json
{
  "style_agent_system_prompt": "Custom system prompt..."
}
```

---

## Security

- History gallery is password-protected
- Default password: `nova2image`
- API keys stored in `.env` (not committed)
- User data (`history/`, `agent_config.json`) excluded from git
