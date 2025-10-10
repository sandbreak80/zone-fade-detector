#!/usr/bin/env python3
"""
Simple syntax checker for the Zone Fade Detector project.
This checks Python syntax without requiring external dependencies.
"""

import ast
import os
import sys
from pathlib import Path

def check_python_file(file_path):
    """Check if a Python file has valid syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the AST to check syntax
        ast.parse(content, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Check all Python files in the project."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    tests_dir = project_root / "tests"
    
    python_files = []
    
    # Find all Python files
    for directory in [src_dir, tests_dir]:
        if directory.exists():
            python_files.extend(directory.rglob("*.py"))
    
    print("🔍 Checking Python syntax...")
    print(f"Found {len(python_files)} Python files")
    print()
    
    errors = []
    passed = 0
    
    for file_path in sorted(python_files):
        relative_path = file_path.relative_to(project_root)
        is_valid, error = check_python_file(file_path)
        
        if is_valid:
            print(f"✅ {relative_path}")
            passed += 1
        else:
            print(f"❌ {relative_path}: {error}")
            errors.append((relative_path, error))
    
    print()
    print(f"📊 Results: {passed} passed, {len(errors)} failed")
    
    if errors:
        print("\n❌ Syntax errors found:")
        for file_path, error in errors:
            print(f"  {file_path}: {error}")
        return 1
    else:
        print("\n✅ All Python files have valid syntax!")
        return 0

if __name__ == "__main__":
    sys.exit(main())