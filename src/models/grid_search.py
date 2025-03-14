import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def perform_grid_search(input_dir: str, output_dir: str):
    # Charger les données
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))
    
    # Définir le modèle et les hyperparamètres à tester
    model = RandomForestRegressor()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    # GridSearch avec validation croisée
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train.values.ravel())
    
    # Sauvegarder les meilleurs paramètres
    os.makedirs(output_dir, exist_ok=True)
    best_params_path = os.path.join(output_dir, 'best_params.pkl')
    with open(best_params_path, 'wb') as f:
        pickle.dump(grid_search.best_params_, f)
    
    print("Grid search completed. Best parameters saved in", best_params_path)

if __name__ == "__main__":
    perform_grid_search("data/processed", "models")
