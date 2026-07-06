## 2026-07-06 - Enhancing color selection with visual pickers
**Learning:** Using `IO.NumberDisplay.color` (serializing to `"display": "color"`) enables a visual color picker for integer inputs in the V3 API. This significantly reduces cognitive load for users compared to manual hex/integer input for color-based nodes.
**Action:** Always prefer `IO.NumberDisplay.color` for nodes involving specific color selection (like chroma keying or background colors) when the backend expects an integer.
