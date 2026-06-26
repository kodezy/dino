from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Match:
    confidence: float
    x: int
    y: int
    w: int
    h: int
