import pandas as pd
import os
import subprocess
import yaml

# Chemins vers les fichiers
TECHNOLOGIES_FILE = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/Data/2050/Technologies.csv"
CONFIG_FILE = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/config_ref.yaml"
OUTPUT_FILE_DIR = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies"
LOG_FILE = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/scenario_analysis.txt"
ENERGYSCOPE_RUNNER = "C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/run_energyscope.py"

# Paramètres de la boucle
COLUMN_X = "f_min"
ROWS_TO_MODIFY = [5, 8, 9, 10, 13, 17, 20, 21, 22, 33]
INTERVAL = (50, 150)
STEP = 0.1
LIMIT_IMPORT = 30

def update_config(run_id):
    """Met à jour le fichier config_ref.yaml avec le nouveau case study."""
    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)
    config['case_study'] = f"Run_{run_id}"
    with open(CONFIG_FILE, "w") as file:
        yaml.dump(config, file)

import time

def run_energyscope():
    """Exécute le script EnergyScope en utilisant le config mis à jour."""
    try:
        subprocess.run(["python", ENERGYSCOPE_RUNNER], check=True)
        time.sleep(5)  # Attendre 5 secondes pour que les fichiers de sortie soient écrits
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution d'EnergyScope: {e}")

def calculate_import_percentage(run_id):
    import time
    output_file = os.path.join(OUTPUT_FILE_DIR, f"Run_{run_id}", "output.csv")
    
    # Attendre que le fichier soit généré
    for _ in range(10):  # Attendre jusqu'à 10 secondes
        if os.path.exists(output_file):
            break
        time.sleep(1)
    else:
        print(f"Fichier {output_file} introuvable après 10 secondes.")
        return None
    """Calcule le pourcentage de ressources importées à partir des résultats."""
    output_file = os.path.join(OUTPUT_FILE_DIR, f"Run_{run_id}", "output.csv")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Le fichier {output_file} n'existe pas.")
    df = pd.read_csv(output_file)
    imported_values = df.iloc[[2, 5, 11, 13, 14, 43], 2].sum()
    total_values = df.iloc[:, 2].sum()
    imported_resources = (imported_values / total_values) * 100
    return imported_resources

def modify_and_run():
    """Boucle sur les valeurs, exécute EnergyScope et enregistre les résultats."""
    original_df = pd.read_csv(TECHNOLOGIES_FILE)
    
    with open(LOG_FILE, "w") as log:
        value = INTERVAL[0]
        run_id = 6
        while value <= INTERVAL[1]:
            df = original_df.copy()
            df.loc[ROWS_TO_MODIFY, COLUMN_X] = value
            df.to_csv(TECHNOLOGIES_FILE, index=False)
            
            # Mettre à jour le fichier de configuration
            update_config(run_id)
            
            # Exécuter EnergyScope
            try:
                run_energyscope()
                import_percentage = calculate_import_percentage(run_id)
            except Exception as e:
                log.write(f"Erreur lors de l'exécution avec valeur {value}: {e}\n")
                continue
            
            # Écriture des résultats
            status = "OK" if import_percentage <= LIMIT_IMPORT else "PAS OK"
            log.write(f"Valeurs modifiées: {COLUMN_X} -> {value}\n")
            log.write(f"Scénario {status} avec {import_percentage:.2f}% d'importations\n\n")
            
            value *= 1.1
            run_id += 1

if __name__ == "__main__":
    modify_and_run()
