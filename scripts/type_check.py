#!/usr/bin/env python3
"""
Simple type checker for the Zone Fade Detector project.
This performs basic type checking without requiring mypy.
"""

import ast
import os
import sys
from pathlib import Path

def check_type_annotations(file_path):
    """Check if a Python file has proper type annotations."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(file_path))
        
        issues = []
        
        # Check for functions without type annotations
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check return type annotation
                if node.returns is None and not node.name.startswith('_'):
                    issues.append(f"Function '{node.name}' missing return type annotation")
                
                # Check parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is None and not arg.arg.startswith('_'):
                        issues.append(f"Parameter '{arg.arg}' in '{node.name}' missing type annotation")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Error parsing file: {e}"]

def main():
    """Check type annotations in all Python files."""
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    
    python_files = list(src_dir.rglob("*.py"))
    
    print("🔍 Checking type annotations...")
    print(f"Found {len(python_files)} Python files")
    print()
    
    total_issues = 0
    files_with_issues = 0
    
    for file_path in sorted(python_files):
        relative_path = file_path.relative_to(project_root)
        is_valid, issues = check_type_annotations(file_path)
        
        if is_valid:
            print(f"✅ {relative_path}")
        else:
            print(f"⚠️  {relative_path}")
            files_with_issues += 1
            total_issues += len(issues)
            for issue in issues:
                print(f"    - {issue}")
    
    print()
    print(f"📊 Results: {len(python_files) - files_with_issues} files OK, {files_with_issues} files with issues")
    print(f"Total issues: {total_issues}")
    
    if total_issues == 0:
        print("\n✅ All files have proper type annotations!")
        return 0
    else:
        print(f"\n⚠️  Found {total_issues} type annotation issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())