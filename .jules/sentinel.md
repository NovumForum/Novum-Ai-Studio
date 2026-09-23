## 2026-03-31 - Safe Query Parameter Validation on Server Endpoints
**Vulnerability:** Unvalidated integer query parameters (e.g. `max_items`, `offset`) on API endpoints like `/history` could trigger unhandled `ValueError` exceptions and 500 Internal Server Errors when passed invalid string types or out-of-range values.
**Learning:** Endpoints converting query parameters directly via `int()` without `try...except` exception handling leak stack traces and cause unhandled 500 errors.
**Prevention:** Always wrap parameter type conversions in `try...except (ValueError, TypeError)` and validate numeric ranges, returning HTTP 400 Bad Request with descriptive JSON error bodies.
