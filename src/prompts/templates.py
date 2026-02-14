"""Prompt templates for driving scene description generation.

Eight prompt variants with increasing sophistication, from zero-shot baseline
to an optimized combination of the best strategies.
"""

from __future__ import annotations

# ── JSON schema description (shared across prompts that use it) ────────────────

OUTPUT_SCHEMA_DESCRIPTION = """
Output MUST be a JSON object with these fields:
{
  "summary": "2-3 sentence overview of the driving scene",
  "objects": [{"category": "car", "count": 2, "details": "two cars ahead in adjacent lane"}],
  "weather": "clear | rainy | foggy | snowy | overcast",
  "lighting": "daytime | night | dawn | dusk",
  "road_type": "highway | city_street | residential | intersection | parking_lot",
  "hazards": ["list of potential hazards for the ego vehicle"],
  "meta_actions": ["recommended actions: brake, accelerate, slow_down, maintain_speed, lane_change_left, lane_change_right, yield, stop"]
}
""".strip()

# ── Few-shot examples (for v5) ────────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
Example 1:
Image: A clear daytime highway scene with several vehicles.
Output:
{
  "summary": "A clear daytime highway scene with moderate traffic. Several vehicles are traveling in the same direction across multiple lanes, with an overpass visible ahead.",
  "objects": [
    {"category": "car", "count": 5, "details": "3 cars ahead in the same lane, 2 in the adjacent lane"},
    {"category": "truck", "count": 1, "details": "1 truck in the rightmost lane"},
    {"category": "traffic sign", "count": 2, "details": "highway signs on overhead gantry"}
  ],
  "weather": "clear",
  "lighting": "daytime",
  "road_type": "highway",
  "hazards": ["moderate traffic density requires safe following distance"],
  "meta_actions": ["maintain_speed"]
}

Example 2:
Image: A rainy night city intersection with pedestrians.
Output:
{
  "summary": "A rainy nighttime city intersection with wet road surfaces creating glare from streetlights. Pedestrians are crossing at the crosswalk while vehicles wait at a red light.",
  "objects": [
    {"category": "person", "count": 3, "details": "3 pedestrians crossing at crosswalk"},
    {"category": "car", "count": 2, "details": "2 cars stopped at the intersection"},
    {"category": "traffic light", "count": 2, "details": "red traffic lights visible"}
  ],
  "weather": "rainy",
  "lighting": "night",
  "road_type": "intersection",
  "hazards": ["wet road reduces traction", "pedestrians crossing in low visibility", "glare from wet road surface"],
  "meta_actions": ["stop", "yield"]
}
""".strip()


# ── Prompt Templates ──────────────────────────────────────────────────────────

PROMPT_TEMPLATES: dict[str, dict[str, str]] = {
    "v1_baseline": {
        "strategy": "Zero-shot basic",
        "description": "Simple direct request",
        "template": (
            "Describe this driving scene image. Include all visible objects, "
            "weather conditions, lighting, road type, potential hazards, and "
            "what driving actions should be taken.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v2_structured": {
        "strategy": "Zero-shot + detailed schema",
        "description": "Detailed field-by-field instructions",
        "template": (
            "Analyze this driving scene image and provide a structured description.\n\n"
            "For each field, follow these guidelines:\n"
            "- summary: Write 2-3 sentences describing the overall scene\n"
            "- objects: List EVERY visible object with exact counts. Categories: "
            "car, bus, truck, motorcycle, bicycle, pedestrian, rider, traffic_light, traffic_sign\n"
            "- weather: Determine from sky, road wetness, visibility (clear/rainy/foggy/snowy/overcast)\n"
            "- lighting: Determine from shadows, sky brightness (daytime/night/dawn/dusk)\n"
            "- road_type: Identify road type (highway/city_street/residential/intersection/parking_lot)\n"
            "- hazards: List all potential dangers to the ego vehicle\n"
            "- meta_actions: List recommended driving actions based on the scene\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v3_role": {
        "strategy": "Role-play AD engineer",
        "description": "Expert persona for domain-specific accuracy",
        "template": (
            "You are a senior autonomous driving perception engineer at a leading "
            "self-driving car company. Your task is to analyze dashboard camera images "
            "and produce structured scene descriptions that would be used to train "
            "and validate AD perception systems.\n\n"
            "Analyze this driving scene with the precision and thoroughness expected "
            "in a safety-critical AD system. Be accurate in object counting, "
            "conservative in hazard assessment, and specific in action recommendations.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v4_cot": {
        "strategy": "Chain-of-thought",
        "description": "Step-by-step reasoning before final answer",
        "template": (
            "Analyze this driving scene image step by step:\n\n"
            "Step 1: OBSERVE - Scan the entire image systematically (left to right, "
            "near to far). List every visible object.\n"
            "Step 2: ENVIRONMENT - Assess weather from sky/road conditions and "
            "lighting from shadows/brightness.\n"
            "Step 3: ROAD CONTEXT - Identify road type, lanes, intersections, and traffic flow.\n"
            "Step 4: HAZARDS - Identify potential dangers considering object motion, "
            "visibility, and road conditions.\n"
            "Step 5: ACTIONS - Recommend driving actions based on the hazard assessment.\n\n"
            "After your analysis, provide the final structured output.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v5_few_shot": {
        "strategy": "Few-shot with examples",
        "description": "Two annotated examples to set quality expectations",
        "template": (
            "Analyze this driving scene image and produce a structured description. "
            "Follow the format and level of detail shown in these examples:\n\n"
            f"{FEW_SHOT_EXAMPLES}\n\n"
            "Now analyze the provided image with the same level of detail.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v6_safety": {
        "strategy": "Safety-focused",
        "description": "Emphasize hazard detection and conservative actions",
        "template": (
            "You are analyzing this driving scene from a SAFETY perspective. "
            "Your primary goal is to identify all hazards and recommend safe driving actions.\n\n"
            "CRITICAL: Prioritize safety over speed. It is better to over-report hazards "
            "than to miss a potential danger. Consider:\n"
            "- Vulnerable road users (pedestrians, cyclists, children)\n"
            "- Vehicles that may change lane or turn unexpectedly\n"
            "- Road surface conditions (wet, icy, debris)\n"
            "- Visibility limitations (glare, fog, occlusion)\n"
            "- Traffic control devices and their current state\n\n"
            "Report all objects accurately, then focus on hazard analysis.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v7_grounded": {
        "strategy": "Grounded / anti-hallucination",
        "description": "Explicit instructions to only report visible objects",
        "template": (
            "Analyze this driving scene image. IMPORTANT RULES:\n\n"
            "1. ONLY report objects you can CLEARLY SEE in the image. Do NOT infer "
            "or imagine objects that are not visible.\n"
            "2. If you are unsure about an object, do NOT include it.\n"
            "3. Count objects carefully - zoom in mentally to each region.\n"
            "4. For weather and lighting, base your assessment ONLY on visual evidence "
            "(sky color, shadows, road reflections, etc.).\n"
            "5. For hazards, only list hazards that are directly evidenced by the scene.\n"
            "6. For actions, only recommend actions justified by visible conditions.\n\n"
            "Accuracy is more important than completeness.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
    "v8_combined": {
        "strategy": "Best combination",
        "description": "Merges role-play + CoT + grounding + structured format",
        "template": (
            "You are a senior autonomous driving perception engineer. Analyze this "
            "dashboard camera image with precision and thoroughness.\n\n"
            "RULES:\n"
            "- ONLY report objects clearly visible in the image. No hallucination.\n"
            "- Count every object carefully and precisely.\n"
            "- Be conservative in hazard assessment but don't miss obvious dangers.\n\n"
            "ANALYZE step-by-step:\n"
            "1. SCAN the image systematically: foreground → midground → background\n"
            "2. IDENTIFY all objects with exact counts and spatial descriptions\n"
            "3. ASSESS weather (from sky, road surface) and lighting (from shadows, brightness)\n"
            "4. CLASSIFY road type and traffic conditions\n"
            "5. DETECT hazards considering all road users and conditions\n"
            "6. RECOMMEND driving actions justified by the hazards identified\n\n"
            "Provide your analysis as structured JSON.\n\n"
            f"{OUTPUT_SCHEMA_DESCRIPTION}"
        ),
    },
}


def get_prompt(variant_id: str) -> str:
    """Get the prompt template text for a given variant ID."""
    if variant_id not in PROMPT_TEMPLATES:
        available = ", ".join(sorted(PROMPT_TEMPLATES.keys()))
        raise ValueError(f"Unknown prompt variant '{variant_id}'. Available: {available}")
    return PROMPT_TEMPLATES[variant_id]["template"]


def get_prompt_info(variant_id: str) -> dict[str, str]:
    """Get metadata about a prompt variant."""
    if variant_id not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt variant '{variant_id}'")
    info = PROMPT_TEMPLATES[variant_id].copy()
    info["id"] = variant_id
    return info


def list_prompts() -> list[dict[str, str]]:
    """List all available prompt variants with metadata."""
    return [
        {"id": k, "strategy": v["strategy"], "description": v["description"]}
        for k, v in PROMPT_TEMPLATES.items()
    ]
