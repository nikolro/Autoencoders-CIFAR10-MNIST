import os
import sys
import glob
import argparse
import subprocess
import jupyter_utils as jupyter_utils

def parse_cli():
    """ Parse command-line arguments. """
    def is_file(filename):
        if not os.path.isfile(filename):
            raise argparse.ArgumentTypeError(f"{filename} is not a file")
        else:
            return filename

    parser = argparse.ArgumentParser(description="Run Jupyter Notebook")
    subparsers = parser.add_subparsers(help="Sub-command help")

    # Run notebooks
    sp_run = subparsers.add_parser("run-nb", help="Run Jupyter notebooks")
    sp_run.set_defaults(subcmd_fn=run_notebooks)
    sp_run.add_argument(
        "nb_paths", type=is_file, help="Notebook to run", metavar="NB_PATH", nargs="+"
    )
    sp_run.add_argument(
        "--allow-errors",
        "-E",
        action="store_true",
        help="Allow errors when running notebooks",
        required=False,
    )
    sp_run.add_argument(
        "--timeout",
        "-T",
        type=int,
        default=10800,  # Default timeout of 3 hours
        help="Timeout in seconds for cell execution (default: 10800 seconds / 3 hours)",
        required=False,
    )

    parsed = parser.parse_args()

    if "subcmd_fn" not in parsed:
        parser.print_help()
        sys.exit()

    return parsed

def run_notebooks(nb_paths, allow_errors=False, timeout=10800, **kwargs):
    """ Executes Jupyter notebooks using nbconvert. """
    print(f">> Running {len(nb_paths)} notebooks with a timeout of {timeout} seconds...")
    nb_paths.sort()
    for nb_path in nb_paths:
        try:
            # Construct the nbconvert command with timeout parameters
            cmd = [
                "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                f"--ExecutePreprocessor.timeout={timeout}",
                f"--ExecutePreprocessor.iopub_timeout={timeout}",
                nb_path
            ]
            if allow_errors:
                cmd.append("--allow-errors")

            # Run the command
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            if allow_errors:
                print(f"⚠ Notebook {nb_path} encountered errors but continuing.")
            else:
                print(f"❌ Notebook {nb_path} failed: {e}", file=sys.stderr)
                sys.exit(1)

if __name__ == "__main__":
    parsed_args = parse_cli()
    parsed_args.subcmd_fn(**vars(parsed_args))