import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor

def train_model(input_dir: str, param_file: str, output_dir: str):
    # Charger les données
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    
    # Charger les meilleurs hyperparamètres
    with open(param_file, 'rb') as f:
        best_params = pickle.load(f)
    
    # Entraînement du modèle
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train.values.ravel())
    
    # Sauvegarde du modèle entraîné
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'trained_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print("Model training completed. Trained model saved in", model_path)

if __name__ == "__main__":
    train_model("data/processed", "models/best_params.pkl", "models")
