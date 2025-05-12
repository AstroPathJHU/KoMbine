"""
Check the log from latex for warnings and errors.
"""

import argparse
import textwrap

from texoutparse import LatexLogParser

def checklatex(filename):
  """
  Check the log from latex for warnings and errors.
  """
  p = LatexLogParser()
  with open(filename, encoding="utf-8") as f:
    p.process(f)

  message = []
  if p.errors:
    message.append(
      "Errors:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.errors)
    )
  if p.warnings:
    message.append(
      "Warnings:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.warnings)
    )
  if p.badboxes:
    message.append(
      "Bad boxes:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.badboxes)
    )
  missing_refs = getattr(p, "missing_refs", [])
  if missing_refs:
    message.append(
      "Bad refs:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in missing_refs)
    )

  if message:
    raise RuntimeError(
      f"Latex gave some {'errors' if p.errors else 'warnings'}:\n\n\n" + "\n\n\n".join(message)
    )

def main(args=None):
  """
  Check the log from latex for warnings and errors.
  """
  p = argparse.ArgumentParser()
  p.add_argument("filename")
  args = p.parse_args(args=args)
  checklatex(args.filename)

if __name__ == "__main__":
  main()
