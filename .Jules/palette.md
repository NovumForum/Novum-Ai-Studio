## 2025-05-14 - [Color Picker for V3 Nodes]
**Learning:** V3 nodes using `IO.Int.Input` for color selection (expecting 0xRRGGBB) should explicitly set `display_mode=IO.NumberDisplay.color`. This metadata signals the frontend to render a visual color picker instead of a numeric input field, significantly reducing cognitive load.
**Action:** Always check if integer inputs represent colors and apply the `color` display mode to enable the appropriate UI widget.
