import os
import subprocess
import socket
from threading import Thread
from dotenv import load_dotenv
from Core.Logger import CustomLogger

def is_port_in_use(port):
    """Check if a given port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def log_process_output(process, logger, name):
    """Log the output of a subprocess."""
    for line in iter(process.stdout.readline, ''):
        if line:
            logger.info(f"{name}: {line.strip()}")
    for line in iter(process.stderr.readline, ''):
        if line:
            logger.error(f"{name}: {line.strip()}")

def start_process(command, logger, name):
    """Start a subprocess with the given command and log output."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    thread = Thread(target=log_process_output, args=(process, logger, name))
    thread.start()
    return process

def main():
    load_dotenv()

    api_logger = CustomLogger('api', 'Logs/api.log').get_logger()
    streamlit_logger = CustomLogger('streamlit', 'Logs/streamlit.log').get_logger()
    simulation_logger = CustomLogger('simulation', 'Logs/simulation.log').get_logger()

    fastapi_port = 8000
    streamlit_port = 8501

    fastapi_process = None
    if not is_port_in_use(fastapi_port):
        api_logger.info(f"Starting FastAPI server on port {fastapi_port}...")
        fastapi_process = start_process(["uvicorn", "App.api:app", "--reload"], api_logger, "FastAPI")
        api_logger.info(f"FastAPI server is running at http://127.0.0.1:{fastapi_port}")
    else:
        api_logger.info(f"FastAPI server is already running on port {fastapi_port}.")

    streamlit_process = None
    if not is_port_in_use(streamlit_port):
        streamlit_logger.info(f"Starting Streamlit app on port {streamlit_port}...")
        streamlit_process = start_process(["python", "-m", "streamlit", "run", "Streamlit_App/main.py"], streamlit_logger, "Streamlit")
        streamlit_logger.info(f"Streamlit app is running at http://127.0.0.1:{streamlit_port}")
    else:
        streamlit_logger.info(f"Streamlit app is already running on port {streamlit_port}.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        simulation_logger.info("Shutting down servers...")
        if fastapi_process:
            fastapi_process.terminate()
        if streamlit_process:
            streamlit_process.terminate()

if __name__ == "__main__":
    main()


#Test commit 