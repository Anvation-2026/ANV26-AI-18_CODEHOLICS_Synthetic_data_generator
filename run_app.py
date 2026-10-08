import sys
import os
import webbrowser
import threading
import time
import uvicorn

def open_browser(port):
    time.sleep(1.5)
    webbrowser.open(f"http://127.0.0.1:{port}")

def main():
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
        
    print(f"==================================================")
    print(f"  CODEHOLICS Synthetic Data Generator & Validator")
    print(f"==================================================")
    print(f"Starting server at: http://127.0.0.1:{port}")
    print(f"Press CTRL+C to stop.")
    
    # Launch browser in a background thread if not headless
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    
    uvicorn.run("app.api_service:app", host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()
