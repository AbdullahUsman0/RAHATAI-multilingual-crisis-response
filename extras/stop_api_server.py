"""
Stop API Server running on port 8080
"""
import subprocess
import sys

print("Stopping API server on port 8080...")

try:
    # Find process using port 8080
    result = subprocess.run(
        ["netstat", "-ano"],
        capture_output=True,
        text=True
    )
    
    lines = result.stdout.split('\n')
    pid = None
    
    for line in lines:
        if ':8080' in line and 'LISTENING' in line:
            parts = line.split()
            if len(parts) >= 5:
                pid = parts[-1]
                break
    
    if pid:
        print(f"Found process {pid} using port 8080")
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], check=True)
            print(f"✅ Stopped process {pid}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Could not stop process {pid}. Try manually:")
            print(f"   taskkill /F /PID {pid}")
    else:
        print("No process found on port 8080")
        
except Exception as e:
    print(f"Error: {e}")
    print("\nManual steps:")
    print("1. Find process: netstat -ano | findstr :8080")
    print("2. Stop process: taskkill /F /PID <PID>")


