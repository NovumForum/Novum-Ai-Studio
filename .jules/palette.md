## 2025-01-24 - Visual Picker UX for Specialized Numeric Inputs
**Learning:** For specialized numeric inputs like hex colors (0xRRGGBB) or normalized ratios (0.0-1.0), using dedicated visual pickers (`display_mode=IO.NumberDisplay.color`) or sliders (`display_mode=IO.NumberDisplay.slider`) significantly reduces cognitive load compared to raw number inputs.
**Action:** Always prefer visual pickers for color integers and sliders for percentage/normalized float ranges when using the V3 API. Ensure the `NumberDisplay` enum and its serialization support these modes.
