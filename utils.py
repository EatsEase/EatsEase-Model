import psutil, os

def log_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    print(f"🧠 {note} Memory usage: {mem:.2f} MB")