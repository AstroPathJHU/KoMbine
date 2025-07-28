"""
Check the log from latex for warnings and errors.
"""

import argparse
import collections
import re
import textwrap

from texoutparse import LatexLogParser

class LatexMessage(collections.namedtuple("LatexMessage", ["type", "logmessage"])):
  """
  Wrapper for the log message from texoutparse.
  """
  @property
  def message(self):
    """
    Get the message from the log message.
    """
    return self.logmessage.info.get("message", None)
  def __str__(self) -> str:
    return str(self.logmessage)


def checklatex(filename, ignore_regexes=()):
  """
  Check the log from latex for warnings and errors.
  """
  p = LatexLogParser()
  with open(filename, encoding="utf-8") as f:
    p.process(f)

  messages = [
    *(LatexMessage("error", e) for e in p.errors),
    *(LatexMessage("warning", e) for e in p.warnings),
    *(LatexMessage("badbox", e) for e in p.badboxes),
    *(LatexMessage("missing_ref", e) for e in getattr(p, "missing_refs", [])),
  ]

  messages = [
    m for m in messages if not any(r.search(str(m.message)) for r in ignore_regexes)
  ]

  errors = [m for m in messages if m.type == "error"]
  warnings = [m for m in messages if m.type == "warning"]
  badboxes = [m for m in messages if m.type == "badbox"]
  missing_refs = [m for m in messages if m.type == "missing_ref"]

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
      f"Found {'errors' if errors else 'warnings'} in {filename}:\n\n\n" + "\n\n\n".join(message)
    )

def main(args=None):
  """
  Check the log from latex for warnings and errors.
  """
  p = argparse.ArgumentParser()
  p.add_argument("filename", nargs="+", help="Filenames of the log files to check")
  p.add_argument("--ignore-regex", action="append", help="Ignore regex",
                 type=re.compile, default=[])
  args = p.parse_args(args=args)
  for filename in args.filename:
    print(f"Checking {filename}...")
    checklatex(filename, ignore_regexes=args.ignore_regex)

if __name__ == "__main__":
  main()
