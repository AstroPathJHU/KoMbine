"""
Check the log from latex for warnings and errors.
"""

import argparse
import re
import textwrap

from texoutparse import LatexLogParser

def checklatex(filename, ignore_regexes=[]):
  """
  Check the log from latex for warnings and errors.
  """
  p = LatexLogParser()
  with open(filename, encoding="utf-8") as f:
    p.process(f)

  errors = [e for e in p.errors if not any(r.search(e.info["message"]) for r in ignore_regexes)]
  warnings = [e for e in p.warnings if not any(r.search(e.info["message"]) for r in ignore_regexes)]
  badboxes = [e for e in p.badboxes if not any(r.search(e.info["message"]) for r in ignore_regexes)]
  missing_refs = [e for e in getattr(p, "missing_refs", []) if not any(r.search(e.info["message"]) for r in ignore_regexes)]

  message = []
  if errors:
    message.append(
      "Errors:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in errors)
    )
  if warnings:
    message.append(
      "Warnings:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in warnings)
    )
  if badboxes:
    message.append(
      "Bad boxes:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in badboxes)
    )
  if missing_refs:
    message.append(
      "Bad refs:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in missing_refs)
    )

  if message:
    raise RuntimeError(
      f"Latex gave some {'errors' if errors else 'warnings'}:\n\n\n" + "\n\n\n".join(message)
    )

def main(args=None):
  """
  Check the log from latex for warnings and errors.
  """
  p = argparse.ArgumentParser()
  p.add_argument("filename")
  p.add_argument("--ignore-regex", action="append", help="Ignore regex", type=re.compile, default=[])
  args = p.parse_args(args=args)
  checklatex(args.filename, ignore_regexes=args.ignore_regex)

if __name__ == "__main__":
  main()
