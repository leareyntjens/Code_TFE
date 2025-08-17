# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:22:04 2025

@author: reynt
"""

import os
import pandas as pd

# === Paramètres ===

base_path = r"C:\Users\reynt\LMECA2675\EnergyScope-EnergyScope.py\case_studies\Scénarios de base"  # À adapter
run1 = "10MGT"
run2 = "GWP_tot_10MGT_corr"
filename = "year_balance.txt"

def convert_to_csv(run_name):
    txt_path = os.path.join(base_path, run_name, 'output', filename)
    csv_path = txt_path.replace('.txt', '.csv')

    if not os.path.exists(txt_path):
        print(f"[❌] Fichier introuvable : {txt_path}")
        return None

    if not os.path.exists(csv_path):
        try:
            df = pd.read_csv(txt_path, sep=None, engine='python')
            df.columns = df.columns.str.strip()
            df.to_csv(csv_path, index=False)
            print(f"✅ Fichier converti : {csv_path}")
        except Exception as e:
            print(f"[❌] Erreur conversion {txt_path}: {e}")
            return None
    return csv_path

def charger_dataframe(csv_path):
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python')
        df.columns = df.columns.str.strip()
        df = df.dropna(how='all')
        df = df[df.iloc[:, 0].notna()]
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        df.set_index(df.columns[0], inplace=True)
        return df
    except Exception as e:
        print(f"[❌] Erreur lecture {csv_path}: {e}")
        return None

def comparer_dataframes(df1, df2):
    differences = []

    lignes = df1.index.intersection(df2.index)
    colonnes = df1.columns.intersection(df2.columns)

    for ligne in lignes:
        for colonne in colonnes:
            val1 = df1.at[ligne, colonne]
            val2 = df2.at[ligne, colonne]
            if pd.isnull(val1) and pd.isnull(val2):
                continue
            if abs(val1 - val2) > 1:
                differences.append(f"{ligne} - {colonne} - {val1} ≠ {val2}")
    
    if differences:
        print("\n🟠 Différences trouvées :")
        for diff in differences:
            print(diff)
    else:
        print("\n✅ Aucun écart détecté entre les deux fichiers.")

# === Exécution ===
csv1 = convert_to_csv(run1)
csv2 = convert_to_csv(run2)

if csv1 and csv2:
    df1 = charger_dataframe(csv1)
    df2 = charger_dataframe(csv2)

    if df1 is not None and df2 is not None:
        comparer_dataframes(df1, df2)
