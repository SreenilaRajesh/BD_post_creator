import sys
import asyncio
from pathlib import Path
import uvicorn

# Ensure project root is importable even if run from another directory
sys.path.insert(0, str(Path(__file__).parent.resolve()))


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run("controller:app", host="127.0.0.1", port=5454, reload=False, log_level="info")


