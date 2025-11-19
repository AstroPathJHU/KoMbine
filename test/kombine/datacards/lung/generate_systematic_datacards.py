#!/usr/bin/env python3
"""
Generate systematic datacard variants from the poisson datacards.

This script creates four folders with datacards that have additional
systematics:
1. poisson_and_flatfielding: poisson_density with 25 flatfielding
   systematics (4% cells, 2.6% donuts)
2. poisson_and_uncorrected_flatfielding: same but with 5x larger errors
   (20% cells, 13% donuts)
3. flatfielding_systematic: fixed observable_type with flatfielding
   systematics
4. uncorrected_flatfielding_systematic: fixed observable_type with larger
   systematics
"""

import pathlib
import sys


def read_datacard(filepath):
  """Read a datacard and return its lines."""
  with open(filepath, 'r', encoding='utf-8') as f:
    return f.readlines()


def count_patients(lines):
  """Count the number of patients from the survival_time line."""
  for line in lines:
    if line.startswith('survival_time'):
      # Split by tabs and count values (skip the label)
      values = line.strip().split('\t')[1:]
      return len(values)
  raise ValueError("Could not find survival_time line")


def add_flatfielding_systematics(lines, cell_error, donut_error, is_cells):
  """
  Add 25 flatfielding systematic lines to the datacard.

  Args:
    lines: List of datacard lines
    cell_error: Error value for cells (e.g., 1.04 for 4%)
    donut_error: Error value for donuts (e.g., 1.026 for 2.6%)
    is_cells: True if this is a cells datacard, False for donuts
  """
  num_patients = count_patients(lines)

  # Find the last line (should be empty or just a newline)
  # We'll add systematics before the last newline
  if lines and lines[-1].strip() == '':
    insert_pos = len(lines) - 1
  else:
    insert_pos = len(lines)
    lines.append('\n')

  # Add separator if not already present
  if not any(line.strip() == '------------'
             for line in lines[insert_pos-2:insert_pos]):
    lines.insert(insert_pos, '------------\n')
    insert_pos += 1

  # Choose the appropriate error value
  error_value = cell_error if is_cells else donut_error

  # Add 25 flatfielding systematics (one per patient)
  for patient_idx in range(1, min(26, num_patients + 1)):
    # Create a line with error for this patient, '-' for all others
    values = ['-'] * num_patients
    values[patient_idx - 1] = str(error_value)

    syst_line = (f"flatfielding_patient{patient_idx}\tlnN\t"
                 + "\t".join(values) + "\n")
    lines.insert(insert_pos, syst_line)
    insert_pos += 1

  return lines


def convert_to_fixed_observable(lines):
  """
  Convert a poisson_density datacard to a fixed observable_type.

  Changes:
  - observable_type: poisson_density -> fixed
  - Replaces num and area lines with a single observable line (num/area)
  """
  new_lines = []
  num_values = None
  area_values = None

  for line in lines:
    if line.startswith('observable_type'):
      new_lines.append('observable_type fixed\n')
    elif line.startswith('num\t') or line.startswith('num '):
      # Store num values but don't add to output yet
      num_values = line.strip().split('\t')[1:]
    elif line.startswith('area\t') or line.startswith('area '):
      # Store area values but don't add to output yet
      area_values = line.strip().split('\t')[1:]
    else:
      new_lines.append(line)

  # Now add the observable line (num/area) before separator or systematics
  if num_values and area_values:
    # Calculate observable = num / area
    observable_values = []
    for num_str, area_str in zip(num_values, area_values):
      num = float(num_str)
      area = float(area_str)
      if area != 0:
        observable = num / area
      else:
        observable = 0.0
      observable_values.append(str(observable))

    # Find where to insert (before the separator line or at the end)
    insert_pos = len(new_lines)
    for i, line in enumerate(new_lines):
      if line.strip() == '------------' and i > 5:  # Skip first separator
        insert_pos = i
        break

    observable_line = 'observable\t' + '\t'.join(observable_values) + '\n'
    new_lines.insert(insert_pos, observable_line)

  return new_lines


def generate_datacards_for_folder(source_folder, target_folder, *,
                                   add_systematics, convert_to_fixed,
                                   cell_error, donut_error):
  # pylint: disable=too-many-arguments
  """
  Generate datacards for a target folder based on source datacards.

  Args:
    source_folder: Path to source folder (usually poisson)
    target_folder: Path to target folder
    add_systematics: Whether to add flatfielding systematics
    convert_to_fixed: Whether to convert to fixed observable_type
    cell_error: Error value for cells datacards
    donut_error: Error value for donuts datacards
  """
  source_path = pathlib.Path(source_folder)
  target_path = pathlib.Path(target_folder)

  # Process each datacard in the source folder
  for source_file in source_path.glob('datacard_*.txt'):
    lines = read_datacard(source_file)

    # Determine if this is a cells or donuts datacard
    is_cells = 'cells' in source_file.name

    # Apply transformations
    if convert_to_fixed:
      lines = convert_to_fixed_observable(lines)

    if add_systematics:
      lines = add_flatfielding_systematics(lines, cell_error, donut_error,
                                            is_cells)

    # Write to target folder
    target_file = target_path / source_file.name
    with open(target_file, 'w', encoding='utf-8') as f:
      f.writelines(lines)

    print(f"Generated: {target_file}")


def main():
  """Generate all systematic datacard variants."""
  script_dir = pathlib.Path(__file__).parent
  poisson_folder = script_dir / 'poisson'

  if not poisson_folder.exists():
    print(f"Error: Source folder {poisson_folder} does not exist",
          file=sys.stderr)
    sys.exit(1)

  # Generate poisson_and_flatfielding (4% cells, 2.6% donuts)
  print("\nGenerating poisson_and_flatfielding datacards...")
  generate_datacards_for_folder(
    poisson_folder,
    script_dir / 'poisson_and_flatfielding',
    add_systematics=True,
    convert_to_fixed=False,
    cell_error=1.04,
    donut_error=1.026
  )

  # Generate poisson_and_uncorrected_flatfielding (20% cells, 13% donuts)
  print("\nGenerating poisson_and_uncorrected_flatfielding datacards...")
  generate_datacards_for_folder(
    poisson_folder,
    script_dir / 'poisson_and_uncorrected_flatfielding',
    add_systematics=True,
    convert_to_fixed=False,
    cell_error=1.20,
    donut_error=1.13
  )

  # Generate flatfielding_systematic (4% cells, 2.6% donuts, fixed)
  print("\nGenerating flatfielding_systematic datacards...")
  generate_datacards_for_folder(
    poisson_folder,
    script_dir / 'flatfielding_systematic',
    add_systematics=True,
    convert_to_fixed=True,
    cell_error=1.04,
    donut_error=1.026
  )

  # Generate uncorrected_flatfielding_systematic (20% cells, 13% donuts)
  print("\nGenerating uncorrected_flatfielding_systematic datacards...")
  generate_datacards_for_folder(
    poisson_folder,
    script_dir / 'uncorrected_flatfielding_systematic',
    add_systematics=True,
    convert_to_fixed=True,
    cell_error=1.20,
    donut_error=1.13
  )

  print("\n✓ All systematic datacards generated successfully!")


if __name__ == '__main__':
  main()
