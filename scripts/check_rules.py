"""Check the course rule: OpenCV only for input/output, no other image libraries.

    python scripts/check_rules.py        (exit code 1 if a rule is broken)

Scans pixelbridge/ only. tests/ and scripts/ may use OpenCV as a reference
to check our own implementations, but the program itself may not.
"""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "pixelbridge")

ALLOWED_CV2 = {
    "imread", "imwrite", "imdecode", "imencode", "imshow", "waitKey", "destroyAllWindows",
    "namedWindow", "VideoCapture", "IMREAD_UNCHANGED", "IMREAD_COLOR", "IMREAD_GRAYSCALE",
    "IMWRITE_PNG_COMPRESSION", "IMWRITE_JPEG_QUALITY",
}
FORBIDDEN_IMPORTS = ("scipy", "skimage", "sklearn", "PIL", "imageio", "torch", "tensorflow")

def check_source(text: str, rel: str) -> list[str]:
    """Inspect real code (not comments or docstrings) with the ast module."""
    problems = []
    tree = ast.parse(text)
    cv2_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    problems.append(f"{rel}:{node.lineno}: imports {a.name} (not allowed in the program)")
                if a.name == "cv2":
                    cv2_names.add(a.asname or "cv2")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                problems.append(f"{rel}:{node.lineno}: imports {node.module} (not allowed in the program)")
            if root == "cv2":
                for a in node.names:
                    if a.name not in ALLOWED_CV2:
                        problems.append(f"{rel}:{node.lineno}: cv2.{a.name} is not an I/O function")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in cv2_names and node.attr not in ALLOWED_CV2):
            problems.append(f"{rel}:{node.lineno}: cv2.{node.attr} is not an I/O function")
    return problems


def check(package: str = PACKAGE) -> list[str]:
    problems = []
    for folder, _, files in os.walk(package):
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as f:
                    problems += check_source(f.read(), os.path.relpath(path, ROOT))
    return problems


def main() -> int:
    problems = check()
    if problems:
        print("Course rule violations:")
        print("\n".join("  " + p for p in problems))
        return 1
    print("OK: OpenCV is used only for input/output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
