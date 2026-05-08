"""
PyPOTS CLI (Command Line Interface) tool
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

# Modifications made by Yuan Feng et al., 2026:
# Using absolute import paths; Adjustments aimed at tuning GTEE hyperparameters via NNI.

from argparse import ArgumentParser

from auto_tuning.tuning import TuningCommand


def main():
    parser = ArgumentParser(
        "PyPOTS Command-Line-Interface tool", usage="pypots-cli <command> [<args>]"
    )
    commands_parser = parser.add_subparsers(help="pypots-cli command helpers")

    # Register commands here
    TuningCommand.register_subcommand(commands_parser)

    # parse all arguments
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # then run
    service = args.func(args)
    service.run()


if __name__ == "__main__":
    main()
