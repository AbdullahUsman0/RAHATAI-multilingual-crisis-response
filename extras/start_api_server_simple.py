"""
Start API Server - Simplified (skips problematic transformer pre-loading)
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("RAHATAI API SERVER - Simplified Startup (Models load on-demand)")
print("=" * 70)
print()

# Skip transformer pre-loading due to network issues
# All models will be loaded on first API request

print("✓ Models will load on-demand on first request")
print("  This may delay the first request by 30-60 seconds")
print()

print("=" * 70)
print("Starting API server...")
print("Server will be available at:")
print("  - Local: http://localhost:8081")
print("  - Android Emulator: http://10.0.2.2:8081")
print("=" * 70)
print()

# Now start the server
from api_server import app
import uvicorn

uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
