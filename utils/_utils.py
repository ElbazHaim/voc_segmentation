import yaml
import argparse

with open("parameters.yaml", "r") as yaml_file:
    parameters = yaml.safe_load(yaml_file)


def _get_dev_run():
    """
    Get status for PyTorch-Lightning fast_dev_run mode.

    Returns:
        bool: True if fast_dev_run is enabled, False otherwise.

    Example:
    >>> import sys
    >>> sys.argv = ["main.py", "--dev"]
    >>> _get_dev_run()
    Using fast_dev_run=True, testing pipeline - model is not properly trained.
    True
    >>> sys.argv = ["main.py"]
    >>> _get_dev_run()
    Running model training.
    False
    """
    parser = argparse.ArgumentParser(
        description="Enable PyTorch-Lightning fast_dev_run mode."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable fast dev run mode to go over the code without proper training.",
    )
    args = parser.parse_args()

    if args.dev:
        print(
            "Using fast_dev_run=True, testing pipeline - model is not properly trained."
        )
        fast_dev_run = True
    else:
        print("Running model training.")
        fast_dev_run = False

    return fast_dev_run
