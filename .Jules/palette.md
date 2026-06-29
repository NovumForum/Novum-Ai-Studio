## 2026-03-07 - Integer Color Picker Pattern
**Learning:** For nodes that represent colors as integers (0xRRGGBB), using `IO.Int.Input` with `display_mode=IO.NumberDisplay.color` is preferred over `IO.Color.Input`. This maintains backend type safety and consistency with integer-based color processing (like bitwise shifts) while still providing a delightful color picker in the UI.
**Action:** Apply `IO.NumberDisplay.color` to integer-based color inputs in future node schemas to enhance usability without breaking backend logic.
