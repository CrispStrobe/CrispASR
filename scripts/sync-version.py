import sys
import re
import os

# Use the provided python path or default to current sys.executable
# (The agent will run this script using the specific miniconda path)

def update_file(file_path, patterns, version):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (not found)")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    new_content = content
    for pattern, replacement in patterns:
        new_content = re.sub(pattern, replacement.replace("{version}", version), new_content, flags=re.MULTILINE)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {file_path}")
    else:
        print(f"No changes for {file_path}")

if __name__ == "__main__":
    version_file = 'VERSION'
    if not os.path.exists(version_file):
        print(f"Error: {version_file} file not found")
        sys.exit(1)
        
    with open(version_file, 'r', encoding='utf-8') as f:
        version = f.read().strip()
    
    print(f"Synchronizing version to {version}...")

    # Rust
    update_file('crispasr/Cargo.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"'),
        (r'crispasr-sys = \{ path = "\.\./crispasr-sys", version = "[^"]+" \}', 'crispasr-sys = { path = "../crispasr-sys", version = "{version}" }')
    ], version)
    update_file('crispasr-sys/Cargo.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"')
    ], version)
    
    # Python
    update_file('python/pyproject.toml', [
        (r'^version = "[^"]+"', 'version = "{version}"')
    ], version)
    
    # Dart/Flutter
    update_file('flutter/crispasr/pubspec.yaml', [
        (r'^version: [^\n]+', 'version: {version}')
    ], version)

    # JavaScript / Bindings
    update_file('bindings/javascript/package.json', [
        (r'^  "version": "[^"]+"', '  "version": "{version}"')
    ], version)

    # CMake (though now it reads from VERSION, we might want to sync hardcoded strings in sub-CMakes if any)
    # Actually, we should check if any other files need it.
    
    print("Version synchronization complete.")
