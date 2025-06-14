import pandas as pd
import numpy as np
from pathlib import Path

recordings = ["wall1", "wall2", "wall3"]

for rec in recordings:
    input_csv = Path(f"../matlab_files/labels/{rec}_labels.csv")
    output_npy = Path(f"../points_lab/og/{rec}_traj.npy")

    df = pd.read_csv(input_csv, sep=',', encoding='utf-8')

    coords = df[['x', 'y']].to_numpy()

    print(f"Przetwarzanie {rec}: Kształt danych: {coords.shape}")
    print(f"Pierwsze 5 wierszy:\n{coords[:5]}")

    # Zapisz jako .npy
    np.save(output_npy, coords)
    print(f"Zapisano dane do {output_npy}")