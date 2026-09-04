import sys
from pathlib import Path

import pandas as pd

CSV_DIR = Path(__file__).resolve().parent / "csv"


def choose_csv_file(csv_dir):
    csv_files = sorted(csv_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No hay archivos CSV en la carpeta: {csv_dir}")

    if len(csv_files) == 1:
        return csv_files[0]

    if len(sys.argv) > 1:
        nombre = sys.argv[1].strip()
        for file in csv_files:
            if file.name.lower() == nombre.lower() or file.stem.lower() == nombre.lower():
                return file
        raise FileNotFoundError(f"No se encontró el archivo '{nombre}' en la carpeta {csv_dir}")

    print("Archivos CSV disponibles:")
    for i, file in enumerate(csv_files, start=1):
        print(f"{i}. {file.name}")

    while True:
        try:
            opcion = input("Escribe el número del archivo que quieres usar: ").strip()
            indice = int(opcion)
            if 1 <= indice <= len(csv_files):
                return csv_files[indice - 1]
            print("Opción inválida. Intenta otra vez.")
        except ValueError:
            print("Debes escribir un número válido.")


def clean_dataset(path):
    df = pd.read_csv(path)
    return df


def main():
    csv_path = choose_csv_file(CSV_DIR)
    df = clean_dataset(csv_path)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.max_rows", None)

    print(f"\nArchivo seleccionado: {csv_path.name}")
    print(df.to_string(index=False))
    print("\nNúmero de filas:", len(df))
    print("Columnas:", list(df.columns))


if __name__ == "__main__":
    main()