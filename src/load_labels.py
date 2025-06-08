import pandas as pd
import numpy as np
from pathlib import Path

# Lista nagrań
recordings = ["wall1", "wall2", "wall3"]

for rec in recordings:
    # Ścieżki do plików
    input_csv = Path(f"../points_lab/{rec}_lab.csv")  # Zmień na swoje nazwy plików
    output_npy = Path(f"../points_lab/{rec}_traj.npy")

    # Wczytaj CSV
    df = pd.read_csv(input_csv, sep=',', encoding='utf-8')

    # Wybierz kolumny z współrzędnymi
    coords = df[['x_środka', 'y_środka']].to_numpy()

    # Sprawdź, czy dane są poprawne
    print(f"Przetwarzanie {rec}: Kształt danych: {coords.shape}")
    print(f"Pierwsze 5 wierszy:\n{coords[:5]}")

    # Zapisz jako .npy
    np.save(output_npy, coords)
    print(f"Zapisano dane do {output_npy}")