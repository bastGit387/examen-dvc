import pandas as pd
import os
import requests
from sklearn.model_selection import train_test_split

def download_data(url: str, output_path: str):
    # Télécharger les données depuis le lien
    response = requests.get(url)
    
    # Vérifier si le téléchargement a réussi
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Data successfully downloaded to {output_path}")
    else:
        print(f"Failed to download data. Status code: {response.status_code}")

def split_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    # Charger les données
    df = pd.read_csv(input_path)
    
    # Séparer les features et la cible
    X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1]   # Dernière colonne (silica_concentrate)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Création du dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde des datasets
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print("Data splitting completed. Processed files saved in", output_dir)

if __name__ == "__main__":
    # Télécharger les données depuis le lien S3
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    raw_data_path = "data/raw.csv"
    
    # Si les données ne sont pas déjà téléchargées, on les télécharge
    if not os.path.exists(raw_data_path):
        download_data(url, raw_data_path)
    
    # Séparer les données et sauvegarder les fichiers traités
    split_data(raw_data_path, "data/processed")

