import argparse
import subprocess

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--i', type=int, default=0, help="Index to start the subprocess loop")
args = parser.parse_args()

# Python executable
python = "/opt/anaconda3/bin/python"

# Range of values for --i
values_for_i = range(args.i, 20)  # Start from provided `--i` and loop up to 19

# Run subprocess for each value
for i in values_for_i:
    print(f"Running mega_loop_inc.py with --i={i}")
    subprocess.run([python, "mega_loop_inc.py", "--i", str(i)])
