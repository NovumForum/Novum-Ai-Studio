## 2025-05-15 - Visual Color Pickers for Integer Inputs

**Learning:** In the V3 API, nodes that represent colors as integers (0xRRGGBB) can be enhanced with a visual color picker in the frontend by setting `display_mode=IO.NumberDisplay.color`. This reduces user cognitive load by eliminating the need to manually calculate integer values for colors.

**Action:** Prefer using `IO.Int.Input` with `display_mode=IO.NumberDisplay.color` for any integer-based color input to provide a better user experience while maintaining backend type consistency.
