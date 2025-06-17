import sys
import subprocess

# List of required packages
required_packages = [
    "kagglehub",
    "pandas",
    "matplotlib",
    "scikit-learn",
	"seaborn"
]

# Function to install packages if not already installed
def install_missing_packages(packages):
    for package in packages:
        try:
            __import__(package if package != "scikit-learn" else "sklearn")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install missing ones
install_missing_packages(required_packages)