import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import requests

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

def normalize_data(input_dir: str, output_dir: str):
    # Charger les données
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    
    # Exclure les colonnes de type datetime ou string
    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])
    
    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convertir en DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Création du dossier de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde des datasets normalisés
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
    
    print("Data normalization completed. Processed files saved in", output_dir)

if __name__ == "__main__":
    # Télécharger les données depuis le lien S3 si elles ne sont pas déjà présentes
    url = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
    raw_data_path = "data/raw.csv"
    
    if not os.path.exists(raw_data_path):
        download_data(url, raw_data_path)
    
    # Normalisation des données
    normalize_data("data/processed", "data/processed")
