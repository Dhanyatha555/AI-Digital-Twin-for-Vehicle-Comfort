"""
Launch script for Streamlit Suspension Digital Twin Dashboard.
Usage:
    python run_app.py
"""

import os
import subprocess
import sys


def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    venv_streamlit = os.path.join(workspace_dir, ".venv", "Scripts", "streamlit.exe")
    if not os.path.exists(venv_streamlit):
        venv_streamlit = "streamlit"

    app_file = os.path.join(workspace_dir, "app.py")

    env = os.environ.copy()
    env["PYTHONPATH"] = workspace_dir

    print("=" * 75)
    print("Launching Physics-Guided Suspension Digital Twin Dashboard...")
    print(f"App: {app_file}")
    print("=" * 75)

    cmd = [venv_streamlit, "run", app_file, "--server.headless=false"]
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
