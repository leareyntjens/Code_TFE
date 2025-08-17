import os
import pandas as pd
import matplotlib.pyplot as plt

def convert_to_csv(run_folder, base_path, num_runs, file_name):
    """
    Convertit les fichiers texte en fichiers CSV conformes.

    :param run_folder: Nom du dossier de la série de runs (par ex. 'run_7').
    :param base_path: Chemin de base vers les case studies.
    :param num_runs: Nombre total de runs (par ex. 72).
    :param file_name: Nom du fichier texte à convertir.
    """
    for run in range(num_runs):
        txt_path = os.path.join(base_path, run_folder, f'Run_{run}', 'output', file_name)
        csv_path = txt_path.replace('.txt', '.csv')

        if not os.path.exists(txt_path):
            print(f"Fichier non trouvé : {txt_path}")
            continue
        
        try:
            # Tenter de deviner le séparateur (tabulation ou autre)
            df = pd.read_csv(txt_path, sep=None, engine='python')
            
            # Nettoyer les colonnes
            df.columns = df.columns.str.strip()
            
            # Sauvegarder en CSV
            df.to_csv(csv_path, index=False)
            print(f"Fichier converti en CSV : {csv_path}")
        except Exception as e:
            print(f"Erreur lors de la conversion de {txt_path}: {e}")

def read_column_by_key(run_folder, base_path, num_runs, file_name, column_name, key_column):
    """
    Lit les fichiers CSV générés et extrait les valeurs pour une colonne donnée, organisées par une clé.

    :param run_folder: Nom du dossier de la série de runs (par ex. 'run_7').
    :param base_path: Chemin de base vers les case studies.
    :param num_runs: Nombre total de runs (par ex. 72).
    :param file_name: Nom du fichier texte initial.
    :param column_name: Nom de la colonne à extraire.
    :param key_column: Colonne utilisée comme clé pour regrouper les données (par ex. 'TECHNOLOGIES').
    :return: Dictionnaire des valeurs regroupées par clé pour la colonne spécifiée.
    """
    data_by_key = {}

    for run in range(num_runs):
        csv_path = os.path.join(base_path, run_folder, f'Run_{run}', 'output', file_name.replace('.txt', '.csv'))
        print(f"Lecture du fichier : {csv_path}")

        if not os.path.exists(csv_path):
            print(f"Fichier non trouvé : {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Nettoyer les colonnes
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"Erreur lors de la lecture de {csv_path}: {e}")
            continue

        # Vérifier que les colonnes spécifiées existent
        if key_column not in df.columns or column_name not in df.columns:
            print(f"Colonnes '{key_column}' ou '{column_name}' non trouvées dans {csv_path}. Colonnes disponibles : {df.columns.tolist()}")
            continue

        for _, row in df.iterrows():
            key = row[key_column]
            value = row[column_name]

            if key not in data_by_key:
                data_by_key[key] = []

            data_by_key[key].append(value)

    return data_by_key

def plot_box_plots_by_group(data_by_key, column_name, group_size=10):
    """
    Trace plusieurs box plots pour les données extraites, divisés en groupes pour une meilleure lisibilité.

    :param data_by_key: Dictionnaire des données organisées par clé.
    :param column_name: Nom de la colonne utilisée pour les valeurs.
    :param group_size: Nombre de clés par graphique.
    """
    # Filtrer et convertir les données pour exclure les boîtes vides ou non numériques
    filtered_data = {}
    for key, values in data_by_key.items():
        try:
            # Convertir toutes les valeurs en float
            numeric_values = [float(value) for value in values]
            if len(numeric_values) > 1 and len(set(numeric_values)) > 1:  # Exclure les boîtes vides/inutiles
                filtered_data[key] = numeric_values
        except ValueError:
            print(f"Valeurs non numériques trouvées pour la clé '{key}', données ignorées.")

    if not filtered_data:
        print("Aucune donnée valide pour le box plot.")
        return

    keys = list(filtered_data.keys())
    data = [filtered_data[key] for key in keys]

    # Diviser les données en groupes
    num_groups = (len(keys) + group_size - 1) // group_size  # Calculer le nombre de groupes
    for i in range(num_groups):
        start = i * group_size
        end = start + group_size
        group_keys = keys[start:end]
        group_data = data[start:end]

        plt.figure(figsize=(10, 6))

        # Créer le box plot pour ce groupe
        plt.boxplot(group_data, labels=group_keys)

        # Ajouter des titres et labels
        plt.title(f"Box Plot pour {column_name} (Groupe {i + 1}/{num_groups})")
        plt.xlabel("Clé")
        plt.ylabel(column_name)

        # Afficher le graphique
        plt.xticks(rotation=45)  # Rotation des labels pour plus de lisibilité
        plt.tight_layout()
        plt.show()




if __name__ == '__main__':
    base_path = 'C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py/case_studies'
    run_folder = 'run_40'
    num_runs = 42
    file_name = 'resources_breakdown.txt'  # Peut être remplacé par d'autres fichiers texte
    column_name = 'Used'  # Colonne cible
    key_column = 'Name'  # Clé pour regrouper les valeurs

    # Étape 1 : Convertir les fichiers texte en CSV
    convert_to_csv(run_folder, base_path, num_runs, file_name)

    # Étape 2 : Lire les fichiers CSV et organiser les données
    data_by_key = read_column_by_key(run_folder, base_path, num_runs, file_name, column_name, key_column)

    # Étape 3 : Tracer plusieurs box plots
    plot_box_plots_by_group(data_by_key, column_name, group_size=15)

    
    for key, values in data_by_key.items():
      print(f"{column_name}_{key} : {values}")

        


