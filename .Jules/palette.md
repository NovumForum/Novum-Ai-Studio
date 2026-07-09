## 2025-05-15 - NumberDisplay Enum Serialization
**Learning:** Even if an Enum inherits from (str, Enum), it may not be automatically serialized to a string by all JSON serializers (like the standard 'json' module) and can cause TypeErrors if passed directly to dictionaries intended for API responses.
**Action:** Always use .value when serializing Enum members to ensure compatibility and prevent potential runtime crashes in API layers.
