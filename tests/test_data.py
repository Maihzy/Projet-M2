#!/usr/bin/env python3
# inspect_data.py
"""
Inspection rapide d'un fichier de données (.parquet, .pkl, .csv)
- Affiche : noms de colonnes, types, 5 premières lignes
- Montre les valeurs uniques (ou seulement leur nombre) par colonne
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_any(path: str):
    ext = Path(path).suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext in [".csv", ".txt"]:
        return pd.read_csv(path)
    # fallback: try pandas
    try:
        return pd.read_parquet(path)
    except Exception:
        pass
    raise ValueError(f"Extension non gérée: {ext}")

def show_uniques(df: pd.DataFrame, max_show: int = 20):
    print("\n=== 3) Valeurs uniques par colonne ===")
    for col in df.columns:
        uniques = df[col].unique()
        n_unique = len(uniques)
        print(f"\n- {col} -> {n_unique} valeur(s) unique(s)")
        if n_unique <= max_show:
            print(uniques)
        else:
            print(f"(> {max_show}, affichage tronqué) -> {uniques[:max_show]}")

def main():
    parser = argparse.ArgumentParser(description="Inspecter un fichier de données (parquet/pickle/csv).")
    parser.add_argument("path", help="Chemin du fichier à inspecter")
    parser.add_argument("--max-unique", type=int, default=20,
                        help="Nombre max de valeurs uniques affichées entièrement (défaut: 20)")
    args = parser.parse_args()

    obj = load_any(args.path)
    print(f"\n=== Type du contenu : {type(obj)} ===")

    if isinstance(obj, pd.DataFrame):
        print("\n=== 1) Colonnes & types ===")
        print(obj.dtypes.sort_index())

        print("\n=== 2) Aperçu (head) ===")
        print(obj.head())

        show_uniques(obj, max_show=args.max_unique)

    elif isinstance(obj, pd.Series):
        print("Objet Series détecté. Conversion en DataFrame pour inspection.")
        df = obj.to_frame(name=obj.name if obj.name else "value")
        print(df.head())
        show_uniques(df, max_show=args.max_unique)

    elif isinstance(obj, (list, tuple, np.ndarray)):
        print(f"Nombre d'éléments : {len(obj)}")
        print("Contenu :")
        print(obj)
    else:
        print("Contenu brut :")
        print(obj)

if __name__ == "__main__":
    main()
