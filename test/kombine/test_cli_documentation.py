"""
Test that all CLI options are documented in the CLI documentation file.
"""

import argparse
import pathlib
import re

from kombine.command_line_interface import (
    _make_kombine_parser,
    _make_kombine_twogroups_parser,
)


def extract_cli_arguments(parser: argparse.ArgumentParser) -> set[str]:
    """
    Extract all argument names from an ArgumentParser.
    
    Returns a set of argument names without the leading dashes.
    """
    arguments = set()
    for action in parser._actions:  # pylint: disable=protected-access
        if action.option_strings:
            for opt in action.option_strings:
                # Remove leading dashes and convert to the canonical form
                arg_name = opt.lstrip('-')
                arguments.add(arg_name)
    return arguments


def extract_documented_arguments(doc_file: pathlib.Path) -> set[str]:
    """
    Extract all documented CLI arguments from the documentation file.
    
    Returns a set of argument names found in code blocks or inline code.
    """
    documented = set()
    content = doc_file.read_text()
    
    # Find all instances of --argument-name in the documentation
    # Matches both inline code and code blocks
    pattern = r'--([a-z0-9-]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        documented.add(match)
    
    return documented


def test_cli_documentation_completeness():
    """
    Test that all CLI arguments are documented in 05_command_line_interface.md
    and that all documented arguments actually exist in the CLI.
    """
    here = pathlib.Path(__file__).parent.parent.parent
    doc_file = here / "docs" / "kombine" / "05_command_line_interface.md"
    
    # Check that the documentation file exists
    assert doc_file.exists(), f"CLI documentation file not found: {doc_file}"
    
    # Get all documented arguments
    documented_args = extract_documented_arguments(doc_file)
    
    # Get arguments from kombine command
    parser_kombine = _make_kombine_parser()
    kombine_args = extract_cli_arguments(parser_kombine)
    
    # Get arguments from kombine_twogroups command
    parser_twogroups = _make_kombine_twogroups_parser()
    twogroups_args = extract_cli_arguments(parser_twogroups)
    
    # Combine all unique arguments from both commands
    all_cli_args = kombine_args | twogroups_args
    
    # Remove 'h' and 'help' as these are standard argparse arguments
    all_cli_args.discard('h')
    all_cli_args.discard('help')
    
    # Check which arguments are missing from documentation
    missing_from_docs = all_cli_args - documented_args
    
    # Check which documented arguments don't exist in CLI
    # Filter out some common words that might appear in documentation
    common_words = {
        'default', 'example', 'options', 'required', 'optional', 'note',
        'kombine', 'twogroups', 'kaplan', 'meier', 'format',
        'months', 'rate', 'analysis', 'survival', 'only', 'v', 'g', 'f',
        'x', 'y', 'all', 'and', 'or', 'the', 'in', 'to', 'a', 'for',
        'mutually', 'exclusive', 'with', 'blue', 'red', 'green', 'purple',
        'orange', 'teal', 'brown', 'pink', 'group', 'groups', 'high',
        'low', 'two', 'single', 'command', 'line', 'interface', 'this',
        'document', 'describes', 'provides', 'tools', 'generate', 'plots',
        'comparing', 'vs', 'values', 'path', 'save', 'output', 'plot',
        'containing', 'patient', 'data', 'control', 'which', 'error',
        'bands', 'are', 'displayed', 'alone', 'include', 'binomial',
        'patient-wise', 'exponential', 'greenwood', 'band', 'exclude',
        'full', 'nll', 'negative', 'log-likelihood', 'from', 'nominal',
        'customize', 'visual', 'appearance', 'of', 'scheme', 'size',
        'inches', 'do', 'not', 'use', 'tight', 'layout', 'time',
        'axis', 'range', 'limits', 'text', 'elements', 'title',
        'label', 'probability', 'location', 'legend', 'if', 'provided',
        'will', 'be', 'left', 'off', 'main', 'median', 'number', 'patients',
        'each', 'labels', 'suffixes', 'added', 'suffix', 'add', 'font',
        'sizes', 'adjust', 'various', 'computational', 'calculation',
        'value', 'filtering', 'epsilon',
        'disable', 'collapsing', 'consecutive', 'death', 'times', 'no',
        'intervening', 'censoring', 'slower', 'but', 'may', 'more',
        'accurate', 'print', 'progress', 'messages', 'during', 'computation',
        'specific', 'arguments', 'these', 'separation', 'separating',
        'threshold', 'p-value', 'conventional',
        'logrank', 'comparison', 'likelihood', 'method', 'handling',
        'ties', 'breslow', 'approximation', 'string', 'examples', 'basic',
        'custom', 'limit', 'advanced', 'tick', 'txt', 'datacard',
        'pdf', 'png', 'jpg', 'html',
    }
    
    # Arguments in docs but not in CLI (potential typos or outdated docs)
    extra_in_docs = documented_args - all_cli_args - common_words
    
    # Report findings
    if missing_from_docs:
        print(f"\n❌ CLI arguments missing from documentation: {sorted(missing_from_docs)}")
    
    if extra_in_docs:
        print(f"\n❌ Documented arguments that don't exist in CLI: {sorted(extra_in_docs)}")
        print("   These may be typos or outdated documentation.")
    
    # The test passes if:
    # 1. All CLI arguments are documented
    # 2. All documented arguments (except common words) exist in CLI
    assert not missing_from_docs, (
        f"The following CLI arguments are not documented in {doc_file}: "
        f"{', '.join(sorted(missing_from_docs))}"
    )
    
    assert not extra_in_docs, (
        f"The following documented arguments don't exist in the CLI: "
        f"{', '.join(sorted(extra_in_docs))}. "
        "These may be typos or outdated documentation."
    )
    
    print(f"\n✓ All {len(all_cli_args)} CLI arguments are documented in {doc_file.name}")
    print("✓ All documented arguments exist in the CLI")


if __name__ == "__main__":
    test_cli_documentation_completeness()
    print("\n✅ CLI documentation completeness test passed!")

