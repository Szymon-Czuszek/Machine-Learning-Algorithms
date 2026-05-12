"""
===============================================================================
Automatic Python Package Installer
===============================================================================

This script checks whether required Python libraries are installed
in the current environment.

If a package is missing:
- it is automatically installed using pip

The script is useful for:
- data science projects
- Jupyter notebooks
- automated environments
- cloud platforms
- reproducible workflows

Required Libraries
-------------------------------------------------------------------------------
- kagglehub      -> Accessing Kaggle datasets
- pandas         -> Data manipulation and analysis
- matplotlib     -> Data visualization
- scikit-learn   -> Machine learning
- seaborn        -> Statistical data visualization
"""

import sys
import subprocess


#==============================================================================
# STEP 1: Define required packages
#==============================================================================

"""
List of Python packages required by the project.

These libraries support:
- machine learning
- visualization
- data analysis
- dataset management
"""

required_packages = [

    # Kaggle dataset integration
    "kagglehub",

    # Data manipulation
    "pandas",

    # Plotting and visualization
    "matplotlib",

    # Machine learning library
    "scikit-learn",

    # Statistical visualization
    "seaborn"
]


#==============================================================================
# STEP 2: Define installation function
#==============================================================================

def install_missing_packages(packages):
    """
    Installs missing Python packages automatically.

    Parameters
    -------------------------------------------------------------------
    packages : list
        List of package names to verify and install.

    Behavior
    -------------------------------------------------------------------
    - Checks whether each package is installed
    - Installs missing packages using pip
    """

    # Iterate through package list
    for package in packages:

        try:

            # Attempt to import package
            __import__(
                package
                if package != "scikit-learn"
                else "sklearn"
            )

        except ImportError:

            # Package is missing
            print(f"Installing {package}...")

            # Install package using pip
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    package
                ]
            )


#==============================================================================
# STEP 3: Install missing packages
#==============================================================================

"""
Execute installation process.

Only missing packages will be installed.
Already installed libraries are skipped.
"""

install_missing_packages(required_packages)


#==============================================================================
# Commentary
#==============================================================================

"""
sys.executable
-------------------------------------------------------------------------------
Returns the path to the currently running Python interpreter.

Example:
-------------------------------------------------------------------------------
/usr/bin/python3

Using sys.executable ensures:
- packages are installed into the correct environment
- compatibility with virtual environments
- compatibility with Jupyter kernels


subprocess.check_call()
-------------------------------------------------------------------------------
Executes shell commands from Python.

In this script:
-------------------------------------------------------------------------------
pip install <package>

is executed programmatically.


Why Use Automatic Installation?
-------------------------------------------------------------------------------
Advantages:
- simplifies environment setup
- improves reproducibility
- reduces manual dependency management
- useful for cloud notebooks and shared projects


__import__()
-------------------------------------------------------------------------------
Dynamically imports modules during runtime.

Used here to:
- verify whether a package exists
- avoid unnecessary installations


Special Handling for scikit-learn
-------------------------------------------------------------------------------
Package name:
    scikit-learn

Import name:
    sklearn

Because the import name differs from the pip package name,
special handling is required:

-------------------------------------------------------------------------------
__import__("sklearn")
-------------------------------------------------------------------------------


Error Handling
-------------------------------------------------------------------------------
try:
    attempts package import

except ImportError:
    installs package automatically

This ensures:
- smoother execution
- reduced setup errors


Practical Use Cases
-------------------------------------------------------------------------------
- Machine learning pipelines
- Kaggle competitions
- Data science notebooks
- Cloud-based analytics
- Automated deployment
- Educational environments
- Reproducible research


Commonly Installed Libraries
-------------------------------------------------------------------------------

pandas
--------
Used for:
- DataFrames
- CSV processing
- data cleaning
- analytics

matplotlib
------------
Used for:
- charts
- graphs
- visual analytics

scikit-learn
-------------
Used for:
- machine learning
- classification
- regression
- clustering

seaborn
--------
Used for:
- advanced statistical visualizations
- heatmaps
- correlation analysis

kagglehub
----------
Used for:
- downloading Kaggle datasets
- dataset integration
- competition workflows


Potential Improvements
-------------------------------------------------------------------------------
Possible enhancements include:
- version control for packages
- logging installation status
- parallel installations
- requirements.txt integration
- virtual environment management
"""
