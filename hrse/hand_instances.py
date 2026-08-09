"""Backward-compatible hand-instance metadata and contact parsing.

Historically ArtHOI used ``left`` and ``right`` both as instance identifiers
and as MANO handedness.  That cannot represent a handover recorded with two
right hands.  The instance schema separates those concepts::

    "hand_instances": {
        "giver": {"handedness": "right"},
        "receiver": {"handedness": "right"}
    }

Legacy contact files with ``appeared`` and ``l/r_*`` fields remain supported.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


VALID_HANDEDNESS = {"left", "right"}


def normalize_handedness(value: str) -> str:
    handedness = str(value).strip().lower()
    aliases = {"l": "left", "lh": "left", "r": "right", "rh": "right"}
    handedness = aliases.get(handedness, handedness)
    if handedness not in VALID_HANDEDNESS:
        raise ValueError(
            f"handedness must be one of {sorted(VALID_HANDEDNESS)}, got {value!r}"
        )
    return handedness


def hand_instance_map(payload: Mapping) -> dict[str, str]:
    """Return an ordered ``instance_id -> handedness`` mapping.

    New files use ``hand_instances``.  Legacy files use ``appeared`` where the
    instance ID itself is the handedness.
    """
    raw_instances = payload.get("hand_instances")
    instances: dict[str, str] = {}
    if isinstance(raw_instances, Mapping):
        items = raw_instances.items()
    elif isinstance(raw_instances, list):
        items = []
        for item in raw_instances:
            if not isinstance(item, Mapping) or "id" not in item:
                raise ValueError("hand_instances list entries require an 'id'")
            items.append((item["id"], item))
    elif raw_instances is None:
        items = ((instance_id, {"handedness": instance_id})
                 for instance_id in payload.get("appeared", []))
    else:
        raise ValueError("hand_instances must be a mapping or list")

    for instance_id, metadata in items:
        instance_id = str(instance_id)
        if not instance_id:
            raise ValueError("hand instance IDs cannot be empty")
        if instance_id in instances:
            raise ValueError(f"duplicate hand instance ID: {instance_id}")
        if isinstance(metadata, str):
            handedness = metadata
        elif isinstance(metadata, Mapping):
            handedness = metadata.get("handedness")
        else:
            raise ValueError(f"invalid metadata for hand instance {instance_id!r}")
        if handedness is None:
            raise ValueError(f"hand instance {instance_id!r} has no handedness")
        instances[instance_id] = normalize_handedness(handedness)

    if not instances:
        raise ValueError("contact metadata contains no hand instances")
    return instances


def _legacy_contact(frame: Mapping, instance_id: str) -> tuple[bool, list[str]]:
    prefix = {"left": "l", "right": "r"}.get(instance_id, instance_id)
    contact = bool(frame.get(f"{prefix}_contact", False))
    fingers = list(frame.get(f"{prefix}_fingers", []))
    return contact, fingers


def contact_tracks(
    payload: Mapping,
    frame_count: int,
    instances: Mapping[str, str] | None = None,
) -> dict[str, list]:
    """Convert contact JSON into tracks keyed by hand instance.

    New per-frame records store ``hands[instance_id] = {contact, fingers}``.
    Flat legacy records (``l_contact``, ``r_fingers``) are accepted too.
    """
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    instances = dict(instances or hand_instance_map(payload))
    tracks: dict[str, list] = {}
    for instance_id in instances:
        tracks[instance_id] = [False] * frame_count
        tracks[f"{instance_id}_fingers"] = [[] for _ in range(frame_count)]

    frames = payload.get("contacts", [])
    if not frames:
        return tracks
    first_frame = min(int(frame["frame"]) for frame in frames)
    populated: set[int] = set()
    for frame in frames:
        index = int(frame["frame"]) - first_frame
        if not 0 <= index < frame_count:
            raise ValueError(
                f"contact frame {frame['frame']} maps outside 0..{frame_count - 1}"
            )
        if index in populated:
            raise ValueError(f"duplicate contact frame {frame['frame']}")
        populated.add(index)
        per_hand = frame.get("hands", {})
        for instance_id in instances:
            if instance_id in per_hand:
                item = per_hand[instance_id]
                if not isinstance(item, Mapping):
                    raise ValueError(
                        f"contact for hand instance {instance_id!r} must be a mapping"
                    )
                contact = bool(item.get("contact", False))
                fingers = list(item.get("fingers", []))
            else:
                contact, fingers = _legacy_contact(frame, instance_id)
            tracks[instance_id][index] = contact
            tracks[f"{instance_id}_fingers"][index] = fingers if contact else []
    return tracks


def xyz_pullback_instances(config, instance_ids: Iterable[str]) -> set[str]:
    """Resolve which hand instances may use XYZ (rather than Z-only) pull-back."""
    instance_ids = list(instance_ids)
    configured = getattr(config, "xyz_pullback_instances", None)
    if configured is None:
        configured = getattr(config, "xyz_pullback_instance", None)
    if configured is None:
        return set(instance_ids) if getattr(config, "optimize_hand_xy", False) else set()
    if isinstance(configured, str):
        configured = [configured]
    selected = {str(value) for value in configured}
    unknown = selected.difference(instance_ids)
    if unknown:
        raise ValueError(
            f"xyz_pullback_instances contains unknown hand instances: {sorted(unknown)}"
        )
    return selected
