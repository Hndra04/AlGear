"""Zone definitions and zone-based people counting.

A zone is a named polygonal region on the frame.  People counting per zone
is done by checking whether the **centre of a person's bounding box** falls
inside the polygon.

Default configuration: a single full-frame zone is created automatically
if no zones are defined.
"""

from dataclasses import dataclass, field

import numpy as np
from loguru import logger


@dataclass
class Zone:
    name: str
    polygon: np.ndarray  # (N, 2) — list of (x, y) vertices in pixel coords

    def contains(self, point: tuple[float, float]) -> bool:
        """Return True if *point* (x, y) is inside the polygon (ray-casting)."""
        x, y = point
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside


def full_frame_zone(img_w: int, img_h: int) -> Zone:
    """Create a zone that covers the entire frame."""
    return Zone(
        name="full_frame",
        polygon=np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]], dtype=float),
    )


def count_per_zone(
    zones: list[Zone],
    person_centres: list[tuple[float, float]],
) -> dict[str, int]:
    """Count how many person centres fall inside each zone.

    Parameters
    ----------
    zones : list[Zone]
        Zones to evaluate.
    person_centres : list[tuple[float, float]]
        (cx, cy) pixel coordinates for each detected person.

    Returns
    -------
    dict mapping zone name → person count.
    """
    counts: dict[str, int] = {z.name: 0 for z in zones}
    for cx, cy in person_centres:
        for z in zones:
            if z.contains((cx, cy)):
                counts[z.name] += 1
    return counts


def load_zones_from_config(
    zone_configs: list[dict] | None,
    img_w: int,
    img_h: int,
) -> list[Zone]:
    """Build Zone objects from a list of config dicts.

    Each dict has:
        'name': str
        'polygon': list of [x, y] pairs (normalised 0-1 or absolute pixels)

    If zone_configs is empty or None, returns a single full-frame zone.
    """
    if not zone_configs:
        return [full_frame_zone(img_w, img_h)]

    zones: list[Zone] = []
    for cfg in zone_configs:
        pts = np.array(cfg["polygon"], dtype=float)
        # Normalised coordinates (all values ≤ 1) → scale to pixels
        if pts.max() <= 1.0:
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
        zones.append(Zone(name=cfg["name"], polygon=pts))

    logger.info(f"Loaded {len(zones)} zones: {[z.name for z in zones]}")
    return zones
