import os

# Define the standard scientific directory structure
DIRECTORIES = [
    "src",
    "notebooks",
    "results/data",
    "results/figures"
]

FILES = {
    "config.yaml": "# Hyperparameters for the cognitive model\nthreshold: 1.5\ndrift_rate: 0.3\nnoise_sigma: 0.1\n",
    "src/__init__.py": "",
    "src/model.py": "# Core cognitive architecture classes go here\n",
    "src/utils.py": "# Math helpers and data transformations go here\n",
    "run_simulation.py": "# Main execution script\n# Loads config -> runs model -> saves data\n",
    "README.md": "# Cognitive Model Project\n\n## Structure\n- `src/`: Core model logic\n- `notebooks/`: Data exploration\n- `results/`: Saved data and plots\n"
}

def create_workflow():
    # Create folders
    for folder in DIRECTORIES:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}")
       
    # Create boilerplate files
    for file_path, initial_content in FILES.items():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                f.write(initial_content)
            print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_workflow()
    print("\n[SUCCESS] Scientific workflow structure created! You can delete this setup script.")

#EOF