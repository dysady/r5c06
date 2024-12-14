import subprocess
import sys

def install_requirements(requirements_file='requirements.txt'):
    try:
        print(f"Installing dependencies from {requirements_file}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_file], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Installation successful!")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation: {e}")
    except FileNotFoundError:
        print("requirements.txt file not found. Please ensure it exists.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Appel de la fonction
if __name__ == "__main__":
    install_requirements()
