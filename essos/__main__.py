"""Main command line interface to ESSOS."""
import sys
import tomllib

def main(cl_args=sys.argv[1:]):
    """Run the main ESSOS code from the command line.

    Reads and parses user input from command line, runs the code,
    and prints and plots the resulting simulation.

    """
    if len(cl_args) == 0:
        print("Using standard input parameters instead of an input TOML file.")
        output = 0
    else:
        parameters = tomllib.load(open(cl_args[0], "rb"))
        output = 0

if __name__ == "__main__":
    main(sys.argv[1:])