import sys
import os
import webbrowser
import threading
import time
import uvicorn

def open_browser(host, port):
    time.sleep(1.5)
    url = f"http://localhost:{port}" if host in ("127.0.0.1", "0.0.0.0") else f"http://{host}:{port}"
    try:
        webbrowser.open(url)
    except Exception:
        pass

def main():
    # Read dynamic PORT from environment (standard across Render, Railway, Heroku, Hugging Face)
    env_port = os.environ.get("PORT")
    if env_port and env_port.isdigit():
        port = int(env_port)
    elif len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    else:
        port = 8000
        
    # Bind to 0.0.0.0 by default for container and cloud ingress accessibility
    host = os.environ.get("HOST", "0.0.0.0")
        
    print(f"==================================================")
    print(f"  CODEHOLICS Synthetic Data Generator & Validator")
    print(f"==================================================")
    print(f"Starting server at: http://{host}:{port}")
    print(f"Press CTRL+C to stop.")
    
    # Launch browser only in desktop local environments, not in cloud containers
    if not os.environ.get("PORT") and not os.environ.get("RENDER") and not os.environ.get("SPACE_ID"):
        threading.Thread(target=open_browser, args=(host, port), daemon=True).start()
    
    uvicorn.run("app.api_service:app", host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()

