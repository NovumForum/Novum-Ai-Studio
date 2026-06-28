## 2025-05-15 - Integer Color Picker Pattern
**Learning:** When a node expects a color as an integer (e.g. 0xRRGGBB), the best UX is achieved by using `IO.Int.Input` with `display_mode=IO.NumberDisplay.color`. This avoids string-to-int conversion in the `execute` method and provides a native color picker in the UI.
**Action:** Always prefer `display_mode=IO.NumberDisplay.color` for integer color inputs in V3 nodes.
