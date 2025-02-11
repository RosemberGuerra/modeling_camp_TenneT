import subprocess
import sys

def install_requirements(requirements_file="requirements.txt"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print("All requirements are installed.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

# Run the function
install_requirements()
