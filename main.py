# main.py
import subprocess
import os
import sys
from dotenv import load_dotenv

# Load .env variables. This is good practice, though your scripts also load it.
load_dotenv()

# --- Configuration ---
PYTHON_EXECUTABLE = sys.executable 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS_TO_RUN = [
    "01_build_kg_and_vector_db.py",
    "02_graph_analysis_and_community.py",
]

VISUALIZATION_SCRIPT = "03_visualize_graph.py"
STREAMLIT_APP_SCRIPT = "streamlit_app.py"


def run_script(script_name: str) -> bool:
    """Runs a Python script using subprocess and checks for errors."""
    full_script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(full_script_path):
        print(f"Error: Script not found at {full_script_path}")
        return False

    print(f"\n--- Running script: {script_name} ---")
    try:
        # `check=True` will raise CalledProcessError if the script returns a non-zero exit code
        process = subprocess.run(
            [PYTHON_EXECUTABLE, full_script_path],
            check=True,
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as text
            cwd=BASE_DIR 
        )
        print(f"--- Output from {script_name}: ---")
        print(process.stdout)
        if process.stderr: 
            print(f"--- Errors/Warnings from {script_name}: ---")
            print(process.stderr)
        print(f"--- Finished script: {script_name} ---\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {script_name}")
        print(f"Return code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print(
            f"Error: Python interpreter '{PYTHON_EXECUTABLE}' not found or script path incorrect."
        )
        return False
    except Exception as e:
        print(f"An unexpected error occurred while trying to run {script_name}: {e}")
        return False


def run_streamlit_app(script_name: str):
    """Runs the Streamlit application."""
    full_script_path = os.path.join(BASE_DIR, script_name)
    if not os.path.exists(full_script_path):
        print(f"Error: Streamlit app script not found at {full_script_path}")
        return

    print(f"\n--- Starting Streamlit application: {script_name} ---")
    print("Note: Streamlit will run in the foreground. Press Ctrl+C in this terminal to stop it.")
    try:
        subprocess.run(
            [PYTHON_EXECUTABLE, "-m", "streamlit", "run", full_script_path,
             "--server.fileWatcherType", "none"], # <--- ADDED THIS FLAG
            check=True,
            cwd=BASE_DIR
        )
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit app: {script_name}")
        print(f"Return code: {e.returncode}")
    except FileNotFoundError:
        print(
            f"Error: Python interpreter '{PYTHON_EXECUTABLE}' or 'streamlit' command not found."
        )
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred while trying to start Streamlit: {e}")

def main():
    """Main function to orchestrate the project pipeline."""
    print("Starting the GraphRAG Literary Project Pipeline...")

    # --- Docker Check (Manual Reminder) ---
    print("\nIMPORTANT: Ensure Qdrant Docker container is running before proceeding.")
    print("  Example command: docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant")

    # --- Run Data Processing Scripts ---
    for script_file in SCRIPTS_TO_RUN:
        if not run_script(script_file):
            print(f"\nPipeline aborted due to error in {script_file}.")
            return  # Stop if a script fails

    print("\nAll data processing scripts completed successfully.")

    # --- Optionally run visualization ---
    run_viz_choice = input(
        "Do you want to generate the graph visualization HTML file? (yes/no) [no]: "
    ).strip().lower()
    if run_viz_choice == 'yes':
        if not run_script(VISUALIZATION_SCRIPT):
            print("Visualization script failed, but proceeding...")
        else:
            print(f"Visualization HTML file should be at data/{os.path.basename(os.getenv('VISUALIZATION_OUTPUT_PATH', 'literary_graph_visualization.html'))}")

    # --- Start Streamlit App ---
    run_app_choice = input(
        "Do you want to start the Streamlit application now? (yes/no) [yes]: "
    ).strip().lower()
    
    if run_app_choice == '' or run_app_choice == 'yes':
        run_streamlit_app(STREAMLIT_APP_SCRIPT)
    else:
        print("Streamlit application will not be started. Pipeline finished.")

    print("\nGraphRAG Literary Project Pipeline Finished.")


if __name__ == "__main__":
    main()