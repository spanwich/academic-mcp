"""
Ollama utilities - auto-start and health checking.
"""

import subprocess
import time
import shutil
from typing import Optional


def is_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    import urllib.request
    import urllib.error
    
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def start_ollama(timeout: int = 30) -> bool:
    """
    Start Ollama server if not running.
    
    Args:
        timeout: Seconds to wait for Ollama to start
        
    Returns:
        True if Ollama is running (was running or successfully started)
    """
    # Already running?
    if is_ollama_running():
        return True
    
    # Check if ollama command exists
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("Warning: Ollama not found in PATH")
        return False
    
    print("Starting Ollama server...")
    
    try:
        # Start ollama serve in background
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True  # Detach from parent
        )
        
        # Wait for it to be ready
        for _ in range(timeout):
            if is_ollama_running():
                print(f"Ollama started (PID: {process.pid})")
                return True
            time.sleep(1)
        
        print("Timeout waiting for Ollama to start")
        return False
        
    except Exception as e:
        print(f"Failed to start Ollama: {e}")
        return False


def ensure_model_available(model: str, host: str = "http://localhost:11434") -> bool:
    """
    Check if model is available, pull if not.
    
    Args:
        model: Model name (e.g., "qwen2.5:14b")
        host: Ollama host URL
        
    Returns:
        True if model is available
    """
    import urllib.request
    import json
    
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = [m.get("name", "") for m in data.get("models", [])]
            
            # Check if model exists (handle tag variations)
            model_base = model.split(":")[0]
            if any(model_base in m for m in models):
                return True
            
            print(f"Model {model} not found. Available: {models}")
            print(f"Pull it with: ollama pull {model}")
            return False
            
    except Exception as e:
        print(f"Error checking models: {e}")
        return False


def ensure_ollama_ready(model: Optional[str] = None, auto_start: bool = True) -> bool:
    """
    Ensure Ollama is running and model is available.
    
    Args:
        model: Model name to check (optional)
        auto_start: Whether to auto-start Ollama if not running
        
    Returns:
        True if ready
    """
    # Start if needed
    if auto_start:
        if not start_ollama():
            return False
    elif not is_ollama_running():
        print("Ollama is not running. Start it with: ollama serve")
        return False
    
    # Check model if specified
    if model:
        return ensure_model_available(model)
    
    return True
