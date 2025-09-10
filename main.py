import argparse
import subprocess
import sys

def run_experiment(exp_number):
    if exp_number == 1:
        subprocess.run([sys.executable, "exp1.py"])
    elif exp_number == 2:
        subprocess.run([sys.executable, "exp2.py"])
    elif exp_number == 3:
        subprocess.run([sys.executable, "exp3.py"])
    else:
        print("Invalid experiment number")

def main():
    parser = argparse.ArgumentParser(description="Run Legal Case Annotation Experiments")
    parser.add_argument(
        "--exp",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Experiment number to run: 1 (Multi-Model), 2 (Legal-specific), 3 (Fine-tuning)"
    )
    args = parser.parse_args()

    run_experiment(args.exp)

if __name__ == "__main__":
    main()
