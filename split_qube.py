#!/usr/bin/env python3
"""
Script to split qube.py into multiple files (qube1.py, qube2.py, etc.)
Each file will contain all the method definitions but otherwise have no changes.
"""

import re
import os

def split_qube_file():
    # Read the original file
    with open('polymath/qube.py', 'r') as f:
        content = f.read()

    # Find all method definitions and their end positions
    method_pattern = r'^    def [^:]+:'
    lines = content.split('\n')

    method_starts = []
    method_ends = []

    for i, line in enumerate(lines):
        if re.match(method_pattern, line):
            method_starts.append(i)

    # Find the end of each method (next method start or end of file)
    for i, start in enumerate(method_starts):
        if i + 1 < len(method_starts):
            # Find the last non-empty line before the next method
            end = method_starts[i + 1] - 1
            while end > start and not lines[end].strip():
                end -= 1
            method_ends.append(end)
        else:
            # Last method goes to end of file
            method_ends.append(len(lines) - 1)

    # Calculate how many files we want (let's say 5 files)
    num_files = 5
    methods_per_file = len(method_starts) // num_files
    remainder = len(method_starts) % num_files

    # Create the split files
    current_method = 0

    for file_num in range(1, num_files + 1):
        # Calculate how many methods go in this file
        if file_num <= remainder:
            methods_in_this_file = methods_per_file + 1
        else:
            methods_in_this_file = methods_per_file

        # Get the range of methods for this file
        start_method = current_method
        end_method = current_method + methods_in_this_file

        # Create the file content
        file_content = []

        # Add header
        file_content.append("################################################################################")
        file_content.append(f"# polymath/qube{file_num}.py: Part {file_num} of Qube class methods.")
        file_content.append("################################################################################")
        file_content.append("")
        file_content.append("from __future__ import division")
        file_content.append("import numpy as np")
        file_content.append("import numbers")
        file_content.append("")
        file_content.append("from polymath.units import Units")
        file_content.append("")
        file_content.append("class Qube(object):")
        file_content.append("    \"\"\"The base class for all PolyMath subclasses.")
        file_content.append("")
        file_content.append("    This is a split file containing part of the Qube class methods.")
        file_content.append("    The complete class definition is spread across multiple files.")
        file_content.append("    \"\"\"")
        file_content.append("")

        # Add class constants and attributes (from original file)
        # Find the section before the first method
        first_method_start = method_starts[0]
        class_content = lines[:first_method_start]

        # Add the class content (constants, attributes, etc.)
        for line in class_content:
            if line.strip() and not line.startswith('################################################################################'):
                file_content.append(line)

        # Add the methods for this file
        for method_idx in range(start_method, end_method):
            if method_idx < len(method_starts):
                method_start = method_starts[method_idx]
                method_end = method_ends[method_idx]

                # Add the method
                for line_idx in range(method_start, method_end + 1):
                    file_content.append(lines[line_idx])
                file_content.append("")  # Add blank line after method

        # Write the file
        filename = f'polymath/qube{file_num}.py'
        with open(filename, 'w') as f:
            f.write('\n'.join(file_content))

        print(f"Created {filename} with methods {start_method + 1}-{end_method}")

        current_method = end_method

if __name__ == "__main__":
    split_qube_file()