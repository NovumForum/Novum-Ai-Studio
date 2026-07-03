## 2026-07-03 - Visual Color Picker for Hex Integer Inputs
**Learning:** Using `display_mode=IO.NumberDisplay.color` for `IO.Int.Input` fields enables a visual hex color picker in the frontend. This is significantly more intuitive than requiring users to manually enter hex codes or decimal equivalents for color keying operations.
**Action:** Always prefer `NumberDisplay.color` for integer inputs that represent RGB hex values (e.g., 0xRRGGBB) to improve accessibility and provide immediate visual feedback.
