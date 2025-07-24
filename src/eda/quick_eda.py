import sys, pathlib, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[2]  # .../Projet_IA
sys.path.insert(0, str(ROOT))
from modules.data_analysis import DataAnalysis 

if __name__ == "__main__":

    print("=== LANCEMENT EDA ===")

    df = pd.read_csv("DATA/raw/Classeur1.csv", sep=";")
    DataAnalysis(df).full_eda()

    print("=== FIN EDA ===")
