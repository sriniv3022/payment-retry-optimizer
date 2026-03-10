import subprocess
import sys

def run_step(description, command):

    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}\n")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"Error in step: {description}")
        sys.exit(1)


def main():

    run_step(
        "Generating synthetic dataset",
        "python src/data/generate_dataset.py"
    )

    run_step(
        "Running feature engineering",
        "python src/features/feature_engineering.py"
    )

    run_step(
        "Training ML model",
        "python src/models/train_model.py"
    )

    run_step(
        "Running strategy evaluation",
        "python src/simulation/evaluate_strategy.py"
    )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()