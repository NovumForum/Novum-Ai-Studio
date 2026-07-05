import os

def safe_join(base, path):
    base = os.path.abspath(base)
    joined = os.path.abspath(os.path.join(base, path))
    try:
        common = os.path.commonpath([base, joined])
    except ValueError:
        # Happens on Windows if paths are on different drives
        raise ValueError(f"Path traversal detected: {path}")
    if common != base:
        raise ValueError(f"Path traversal detected: {path}")
    return joined

def test():
    base = os.path.abspath("input")
    if not os.path.exists(base): os.makedirs(base)

    cases = [
        ("dataset1", True),
        ("../outside", False),
        ("/etc/passwd", False),
        ("dataset1/../../outside", False),
        (".", True),
    ]

    for path, expected_safe in cases:
        try:
            result = safe_join(base, path)
            print(f"Path: {path:25} -> {result}")
            if not expected_safe:
                print(f"  FAILED: {path} should have been blocked")
        except ValueError as e:
            print(f"Path: {path:25} -> BLOCKED: {e}")
            if expected_safe:
                print(f"  FAILED: {path} should have been allowed")

if __name__ == "__main__":
    test()
