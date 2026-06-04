#!/usr/bin/env python3
"""
Script to create a virtual environment and install dependencies.
Creates .venv directory and installs packages from requirements.txt
"""

import subprocess
import sys
import os
from pathlib import Path


def create_venv(venv_path=".venv"):
    """Create a virtual environment at the specified path."""
    print(f"Creating virtual environment at {venv_path}...")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    print(f"✓ Virtual environment created at {venv_path}")


def get_pip_executable(venv_path=".venv"):
    """Get the pip executable path for the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "pip")
    else:
        return os.path.join(venv_path, "bin", "pip")


def install_requirements(venv_path=".venv", requirements_file="requirements.txt"):
    """Install packages from requirements.txt using the virtual environment's pip."""
    pip_executable = get_pip_executable(venv_path)
    
    if not Path(requirements_file).exists():
        print(f"Error: {requirements_file} not found")
        sys.exit(1)
    
    print(f"Installing packages from {requirements_file}...")
    subprocess.run(
        [pip_executable, "install", "-r", requirements_file],
        check=True
    )
    print(f"✓ Packages installed successfully")


def main():
    """Main function to set up the virtual environment and install dependencies."""
    venv_path = ".venv"
    requirements_file = "requirements.txt"
    
    try:
        # Create venv if it doesn't already exist
        if not Path(venv_path).exists():
            create_venv(venv_path)
        else:
            print(f"Virtual environment already exists at {venv_path}")
        
        # Install requirements
        install_requirements(venv_path, requirements_file)
        
        print("\n✓ Setup complete!")
        print(f"\nTo activate the virtual environment, run:")
        if sys.platform == "win32":
            print(f"  {venv_path}\\Scripts\\activate")
        else:
            print(f"  source {venv_path}/bin/activate")
            
    except subprocess.CalledProcessError as e:
        print(f"Error during setup: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
