import os
import zipfile

# Папка с zip-файлами
folder = r"C:\Users\Volkov-iv\Desktop\ксвшки"

# Проходим по всем файлам
for filename in os.listdir(folder):

    if not filename.lower().endswith(".zip"):
        continue

    zip_path = os.path.join(folder, filename)

    try:
        # Распаковка
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

        # Удаление zip после успешной распаковки
        os.remove(zip_path)

        print(f"OK: {filename}")

    except Exception as e:
        print(f"ERROR: {filename} -> {e}")

print("\nГотово")