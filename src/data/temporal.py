"""Temporal scene description: process consecutive frames for scene evolution.

Groups BDD100K images by video ID and generates scene evolution narratives
by comparing descriptions across frames from the same driving sequence.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FrameInfo:
    """Information about a single frame in a video sequence."""

    image_name: str
    frame_id: str
    index: int  # Sort key for ordering within a video


@dataclass
class VideoSequence:
    """A group of consecutive frames from the same video."""

    video_id: str
    frames: list[FrameInfo] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def extract_video_id(image_name: str) -> tuple[str, str]:
    """Extract video ID and frame ID from a BDD100K image name.

    BDD100K format: {video_id}-{frame_id}.jpg
    Example: c0035eda-6e1b34d6.jpg → ('c0035eda', '6e1b34d6')

    Returns:
        Tuple of (video_id, frame_id).
    """
    name = image_name.rsplit(".", 1)[0]  # Strip extension
    parts = name.split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return name, "0"


def find_video_sequences(
    image_names: list[str],
    min_frames: int = 2,
) -> list[VideoSequence]:
    """Group images by video ID and return sequences with multiple frames.

    Args:
        image_names: List of BDD100K image filenames.
        min_frames: Minimum number of frames for a valid sequence.

    Returns:
        List of VideoSequence objects, sorted by video_id.
    """
    groups: dict[str, list[FrameInfo]] = defaultdict(list)

    for idx, name in enumerate(image_names):
        video_id, frame_id = extract_video_id(name)
        groups[video_id].append(
            FrameInfo(image_name=name, frame_id=frame_id, index=idx)
        )

    sequences = []
    for vid, frames in sorted(groups.items()):
        if len(frames) >= min_frames:
            frames.sort(key=lambda f: f.frame_id)  # Order by frame ID
            sequences.append(VideoSequence(video_id=vid, frames=frames))

    logger.info(
        f"Found {len(sequences)} video sequences with {min_frames}+ frames "
        f"out of {len(groups)} total videos"
    )
    return sequences


def build_temporal_prompt(
    previous_description: str | None = None,
    frame_number: int = 1,
    total_frames: int = 1,
) -> str:
    """Build a temporal-aware prompt for scene evolution description.

    Args:
        previous_description: The VLM description of the previous frame (if any).
        frame_number: Current frame number in the sequence.
        total_frames: Total frames in the sequence.

    Returns:
        A prompt string that instructs the VLM to describe scene changes.
    """
    if previous_description is None or frame_number == 1:
        # First frame — standard description + temporal awareness
        return """You are analyzing frame {frame_number} of {total_frames} from a dashcam driving sequence.

Describe this driving scene with detailed awareness of:
1. **Scene Layout**: Road type, lane configuration, surroundings
2. **Dynamic Objects**: All vehicles, pedestrians, cyclists with positions and apparent motion
3. **Environment**: Weather, lighting, visibility conditions
4. **Hazards**: Any potential safety concerns

Pay special attention to object positions and their likely trajectory, as this will be compared with subsequent frames.

Output MUST be a JSON object with these fields:
{{
  "summary": "2-3 sentence overview emphasizing dynamic elements",
  "objects": [{{"category": "car|truck|person|...", "count": N, "details": "position and motion"}}],
  "weather": "clear|rainy|foggy|snowy|overcast",
  "lighting": "daytime|night|dawn|dusk",
  "road_type": "highway|city_street|residential|intersection|parking_lot",
  "hazards": ["list of potential hazards"],
  "meta_actions": ["brake|accelerate|lane_change_left|lane_change_right|yield|maintain_speed|stop|slow_down"]
}}""".format(frame_number=frame_number, total_frames=total_frames)

    # Subsequent frames — describe changes
    return """You are analyzing frame {frame_number} of {total_frames} from a dashcam driving sequence.

The PREVIOUS frame was described as:
---
{previous_description}
---

Compare this frame with the previous description and describe:
1. **New objects**: Objects that appeared since the last frame
2. **Departed objects**: Objects no longer visible
3. **State changes**: Objects that moved, changed lane, braked, accelerated
4. **Scene evolution**: How the driving situation has evolved
5. **Updated hazard assessment**: Current safety concerns

Output MUST be a JSON object with these fields:
{{
  "summary": "2-3 sentences focusing on CHANGES from the previous frame",
  "objects": [{{"category": "car|truck|person|...", "count": N, "details": "position, motion, and change from previous"}}],
  "weather": "clear|rainy|foggy|snowy|overcast",
  "lighting": "daytime|night|dawn|dusk",
  "road_type": "highway|city_street|residential|intersection|parking_lot",
  "hazards": ["current hazards, noting any new or resolved ones"],
  "meta_actions": ["recommended actions based on scene evolution"],
  "scene_changes": {{
    "new_objects": ["objects that appeared"],
    "departed_objects": ["objects that left the scene"],
    "state_changes": ["descriptions of how objects changed"]
  }}
}}""".format(
        frame_number=frame_number,
        total_frames=total_frames,
        previous_description=previous_description,
    )
