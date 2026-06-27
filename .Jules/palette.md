## 2025-05-15 - [Color Picker for Numeric Inputs]
**Learning:** To enable a color picker in the UI for V3 API nodes that expect an integer color (0xRRGGBB), the `display_mode=IO.NumberDisplay.color` attribute should be used on an `IO.Int.Input`. This is more architecturally consistent than using a separate `IO.Color` string input which would require manual conversion and could lead to type mismatches.
**Action:** Prefer `IO.Int.Input(..., display_mode=IO.NumberDisplay.color)` for integer color inputs to provide a visual picker while maintaining backend type safety.
