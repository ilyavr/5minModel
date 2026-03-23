import os
import pefile

TARGET = "libgomp-1.dll"

def has_dependency(path):
    try:
        pe = pefile.PE(path, fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']]
        )

        if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            return False

        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll = entry.dll.decode().lower()
            if dll == TARGET:
                return True
    except Exception:
        pass

    return False


for root, dirs, files in os.walk("."):
    for name in files:
        if name.lower().endswith((".dll", ".exe")):
            path = os.path.join(root, name)

            if has_dependency(path):
                print("FOUND:", os.path.abspath(path))