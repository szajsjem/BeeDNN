import os
import subprocess
import sys

EMSDK_ENV_PS1 = r"C:\Synchronised\emsdk\emsdk_env.ps1"
EMSDK_ENV_BAT = r"C:\Synchronised\emsdk\emsdk_env.bat"
SRC_FILES_LIST = "files_utf8.txt"
WASM_SRC = os.path.abspath("wasm_binding/src/BeeDNNWasm.cpp")
OUTPUT_DIR = os.path.abspath("wasm_binding/build_wasm")
TEMP_DIR = os.path.abspath("wasm_binding/build_wasm/obj")

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

def run_command_ps(cmd):
    full_cmd = f"powershell -Command \"& '{EMSDK_ENV_PS1}'; {cmd}\"".replace('\x00', '')
    print(f"Executing (PS): {cmd}")
    process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return process.returncode == 0, stdout, stderr

def run_command_cmd(cmd):
    full_cmd = f"cmd /c \"{EMSDK_ENV_BAT} && {cmd}\""
    print(f"Executing (CMD): {cmd}")
    process = subprocess.Popen(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    return process.returncode == 0, stdout, stderr

def main():
    if not os.path.exists(SRC_FILES_LIST):
        print(f"Error: {SRC_FILES_LIST} not found.")
        return

    with open(SRC_FILES_LIST, 'r', encoding='utf-8') as f:
        files = [line.strip().lstrip('\ufeff') for line in f if line.strip()]

    obj_files = []
    
    # Add WASM binding source
    files.append(WASM_SRC)
    total = len(files)

    print(f"Starting compilation of {total} files...")

    for i, file_path in enumerate(files):
        base_name = os.path.basename(file_path)
        obj_name = os.path.splitext(base_name)[0] + ".o"
        obj_path = os.path.join(TEMP_DIR, obj_name)
        
        file_path = os.path.abspath(file_path)
        
        # Check if we can skip
        if os.path.exists(obj_path):
             # print(f"[{i+1}/{total}] Skipping {base_name} (already exists)")
             obj_files.append(obj_path)
             continue

        print(f"[{i+1}/{total}] Compiling {base_name}...")
        
        # Using CMD for consistency as it seems more stable for Emscripten on this machine
        cmd = f"em++ -O3 -fexceptions -lembind -Isrc -Iwasm_binding/src -c \"{file_path}\" -o \"{obj_path}\""
        success, stdout, stderr = run_command_cmd(cmd)
        if not success:
            print(f"Error compiling {base_name}")
            print(f"Stdout: {stdout}")
            print(f"Stderr: {stderr}")
            sys.exit(1)
        
        obj_files.append(obj_path)

    print("Compilation finished. Starting linking...")
    
    # Create linker response file with absolute paths
    response_file = os.path.abspath("linker_files.txt")
    with open(response_file, "w") as f:
        for obj in obj_files:
            f.write(f'"{obj}"\n')

    # Linking command
    js_output = os.path.join(OUTPUT_DIR, "beednn.js")
    link_cmd = f"em++ -O3 -fexceptions -lembind @\"{response_file}\" -o \"{js_output}\" -sMODULARIZE=1 -sEXPORT_NAME=BeeDNNModule -sALLOW_MEMORY_GROWTH=1 -sENVIRONMENT=web,worker"
    
    success, stdout, stderr = run_command_cmd(link_cmd)
    if success:
        print(f"Build successful! Output in {OUTPUT_DIR}")
    else:
        print("Linking failed.")
        print(f"Stdout: {stdout}")
        print(f"Stderr: {stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
