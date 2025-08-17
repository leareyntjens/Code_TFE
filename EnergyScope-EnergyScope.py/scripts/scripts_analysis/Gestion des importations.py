import pandas as pd
import subprocess
import os
import yaml
import csv
from pathlib import Path

# === CONFIGURATION ===
base_path = Path("C:/Users/reynt/LMECA2675/EnergyScope-EnergyScope.py")
resources_csv_path = base_path / "Data/2050/Resources.csv"
yaml_path = base_path / "scripts/config_ref.yaml"
script_path = base_path / "scripts/run_energyscope.py"
case_study_dir = base_path / "case_studies"

target_resources = ["BIOETHANOL", "BIODIESEL", "GAS_RE", "H2_RE", "AMMONIA_RE", "METHANOL_RE"]
delta = 0.01
i = 0

# === Fonction pourcentage d'importation
def get_import_share(run_name):
    csv_path = os.path.join(case_study_dir, run_name, "output", "sankey", "input2sankey.csv")
    import_sum = 0.0
    filtered_sum = 0.0
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = list(csv.reader(csvfile))
            data_rows = reader[1:]
            all_targets = set(row[1].strip() for row in data_rows if len(row) > 1)
            for row in data_rows:
                try:
                    source = row[0].strip()
                    value = float(row[2])
                    if source.startswith("Imp."):
                        import_sum += value
                    if source not in all_targets:
                        filtered_sum += value
                except (IndexError, ValueError):
                    print(f"[{run_name}] ⚠️ Erreur de lecture d'une ligne.")
        if filtered_sum > 0:
            return (import_sum / filtered_sum) * 100
        else:
            print(f"[{run_name}] ❌ Dénominateur nul.")
            return None
    except FileNotFoundError:
        print(f"[{run_name}] ❌ Fichier introuvable : {csv_path}")
        return None

# === Sauvegarde initiale
backup_path = resources_csv_path.with_name("Resources_backup.csv")
if not backup_path.exists():
    with open(resources_csv_path, "r", encoding="utf-8") as f_in:
        with open(backup_path, "w", encoding="utf-8") as f_out:
            f_out.write(f_in.read())
    print(f"📦 Fichier sauvegardé sous : {backup_path.name}")

# === Boucle principale
while True:
    with open(resources_csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Identifier l'en-tête et charger dans un DataFrame temporaire
    header_line = 2  # Ligne 3 (index 2)
    header = lines[header_line].strip().split(';')
    data_lines = lines[header_line + 1:]
    df = pd.DataFrame([l.strip().split(';') for l in data_lines], columns=header)

    # Enregistrer les valeurs initiales uniquement à la première itération
    if i == 0:
        initial_values = df[df['parameter name'].str.strip().isin(target_resources)].copy()

    # Modifier uniquement la colonne 'gwp_op' pour les ressources ciblées
    for res in target_resources:
        mask = df['parameter name'].str.strip() == res
        if i == 0:
            continue  # on n'applique rien la première fois
        if not initial_values.empty:
            initial_val = float(initial_values.loc[initial_values['parameter name'] == res, 'gwp_op'].values[0])
            df.loc[mask, 'gwp_op'] = f"{initial_val + i * delta:.5f}"

    # Réécriture du fichier complet (2 lignes d'en-tête + données modifiées)
    with open(resources_csv_path, 'w', encoding='utf-8') as f:
        f.writelines(lines[:header_line + 1])  # lignes 0 à 2 incluses
        for _, row in df.iterrows():
            f.write(';'.join(str(val) for val in row.values) + '\n')

    print(f"\n🔁 Simulation {i} | GWP op += {i * delta:.2f} pour {target_resources}")

    # Modifier le fichier YAML
    run_name = f"GWP_op_iter_{i:02d}"
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    config['case_study'] = run_name
    config['cs_path'] = 'case_studies'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Lancer EnergyScope
    result = subprocess.run(["python", str(script_path)], shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Échec de l'exécution d'EnergyScope.")
        print("📄 STDERR :\n", result.stderr)
        print("📄 STDOUT :\n", result.stdout)
        break

    # Lire le % d’importation
    import_percentage = get_import_share(run_name)
    if import_percentage is not None:
        print(f"📊 Importation : {import_percentage:.2f} %")
        if import_percentage < 50:
            print("✅ Moins de 50 % d'importation atteints. Fin.")
            break
    else:
        print("⚠️ Importation non calculable. Arrêt.")
        break

    i += 1
