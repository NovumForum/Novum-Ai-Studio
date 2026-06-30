## 2025-05-14 - Integer-based Color Picker
**Learning:** Nodes that expect integer values for colors (e.g., 0xRRGGBB) can still benefit from a visual color picker by setting `display_mode=IO.NumberDisplay.color` in the `Int.Input` definition. This maintains backend type consistency while providing a better UX than a raw number input.
**Action:** Check for nodes that take integer color inputs and ensure they use `IO.NumberDisplay.color` to enable the frontend color picker widget.
