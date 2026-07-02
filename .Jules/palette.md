## 2026-07-02 - [Integer Color Picker]
**Learning:** When providing color inputs for nodes that expect integer values (0xRRGGBB), using `IO.Int.Input` with `display_mode=IO.NumberDisplay.color` provides a much better UX than a plain number input, while maintaining backend type consistency.
**Action:** Use `IO.NumberDisplay.color` for any integer field that represents a hex color to enable the visual color picker in the frontend.
