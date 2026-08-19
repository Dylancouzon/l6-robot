"""Boxes and text drawn onto the camera feed.

The panel is HTML (see page.html); only the annotated video is drawn here.
Colours are BGR, and the same palette is repeated as CSS custom properties at
the top of page.html.
"""
import cv2

BG = (246, 244, 240)
INK = (45, 38, 32)
RED = (76, 36, 220)      # Qdrant red: unknown
TEAL = (136, 150, 0)     # recognized
ORANGE = (30, 140, 235)  # the "maybe" band: close to the bar, not over it
VIOLET = (255, 71, 96)
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, xy, scale=0.8, color=INK, thick=1):
    cv2.putText(img, s, xy, FONT, scale, color, thick, cv2.LINE_AA)


def _chip(img, s, xy, bg, scale=0.8):
    """A filled label tag above a box, readable from across a room."""
    (w, h), _ = cv2.getTextSize(s, FONT, scale, 2)
    x, y = xy
    cv2.rectangle(img, (x - 6, y - h - 8), (x + w + 6, y + 8), bg, -1)
    text(img, s, (x, y), scale, (255, 255, 255), 2)


def draw_feed(frame, tracks, focused):
    """Draw every displayed track. Three states, not two: recognized (teal),
    a near-miss guess (orange, "hat? 0.87"), or unknown (red). The focused
    track gets the thick box."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        known = t.label is not None
        color = TEAL if known else (ORANGE if t.guess else RED)
        thick = 6 if t is focused else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        if t.last_query:
            tag = (f"{t.label}  {t.score:.2f}" if known
                   else f"{t.guess}?  {t.score:.2f}" if t.guess
                   else f"UNKNOWN  {t.score:.2f}")
            _chip(frame, tag, (x1 + 4, max(30, y1 - 12)), color)
    return frame
