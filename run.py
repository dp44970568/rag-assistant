"""
RAG AI Assistant Launcher.
Starts the FastAPI application and automatically launches your web browser.
"""

import sys
import time
import webbrowser
import threading
import uvicorn

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def open_browser():
    """Wait for server socket to listen, then open browser."""
    time.sleep(1.2)
    print(f"\n✨ Opening RAG AI Assistant UI at: {URL}\n")
    try:
        webbrowser.open(URL)
    except Exception:
        pass


def main():
    print("=" * 65)
    print(" 🚀  RAG AI ASSISTANT  (FastAPI + High-Speed Vector Retrieval)")
    print("     Pure Modern Web Architecture • Zero Streamlit")
    print("=" * 65)
    print(f" • Local Address:  {URL}")
    print(f" • API Docs:       {URL}/docs")
    print(f" • Press Ctrl + C to stop the server\n")

    # Start browser opener in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch uvicorn
    uvicorn.run("app:app", host=HOST, port=PORT, reload=False, log_level="info")


if __name__ == "__main__":
    main()
