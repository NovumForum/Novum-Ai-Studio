## 2025-05-15 - Visual Color Pickers for Integers
**Learning:** The V3 API can support visual color pickers for integer inputs (0xRRGGBB) by using `display_mode=IO.NumberDisplay.color`. This significantly reduces cognitive load compared to manual hex-to-int conversion.
**Action:** Use `IO.NumberDisplay.color` for any integer inputs representing colors, and ensure `Int.Input.as_dict` correctly serializes the display mode value.
