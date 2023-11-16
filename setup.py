"""
This script sets up a virtual environment, installs project requirements if not already installed,
and runs the 'hilbert.py' script within the virtual environment.

Usage:
1. Ensure you are in the project directory before running this script.
2. Create a virtual environment named 'venv' if it doesn't exist.
3. Activate the virtual environment.
4. Install project requirements from 'requirements.txt' if the file exists.
5. Run 'hilbert.py' script.
6. Deactivate the virtual environment.

Note: Requires Python 3.3+ for venv support.

"""

import os
import sys
import subprocess

is_windows = sys.platform.startswith('win')
script_dir = os.getcwd()
python_executable = sys.executable
venv_dir = os.path.join(script_dir, 'venv')

if not os.path.exists(venv_dir):
    subprocess.run([python_executable, '-m', 'venv', venv_dir])

if is_windows:
    activate_script = os.path.join(venv_dir, 'Scripts', 'activate')
else:
    activate_script = os.path.join(venv_dir, 'bin', 'activate')
subprocess.run([activate_script], shell=True)

requirements_file = 'requirements.txt'
if os.path.exists(requirements_file):
    subprocess.run([python_executable, '-m', 'pip', 'install', '-r', requirements_file])

hilbert_script = 'hilbert.py'
subprocess.run([python_executable, hilbert_script])
subprocess.run(['deactivate'], shell=True)
