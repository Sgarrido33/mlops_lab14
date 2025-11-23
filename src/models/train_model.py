import argparse
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import yaml
import logging
from mlflow.tracking import MlflowClient
import os

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model from config.")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    # Por defecto es None, lo que evita que intente guardar en C:/ si no se le indica
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()

def get_model_instance(name, params):
    model_map = {
        'LinearRegression': LinearRegression,
        'RandomForest': RandomForestRegressor,
        'GradientBoosting': GradientBoostingRegressor,
        'HistGradientBoosting': HistGradientBoostingRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)

def load_and_split_data(data_path, config):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    selected_features = config['model']['feature_sets']['rfe']
    available_columns = data.columns.tolist()
    final_features = []
    
    # Lógica para encontrar columnas aunque tengan prefijos
    for feat in selected_features:
        if feat in available_columns:
            final_features.append(feat)
        else:
            found = False
            for col in available_columns:
                if col.endswith(f"__{feat}") or col == feat:
                    final_features.append(col)
                    found = True
                    break
            if not found:
                logger.warning(f"Feature '{feat}' not found in dataset. Ignored.")
            
    logger.info(f"Using {len(final_features)} selected features")

    X = data[final_features]
    y = data[config['model']['target_variable']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    logger.info("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    metrics = {'mae': mae, 'r2': r2, 'rmse': rmse}
    logger.info(f"Model performance - R²: {r2:.4f}, MAE: {mae:.2f}")
    return model, metrics

def save_model_artifacts(model, config, metrics, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(models_dir, 'trained', 'house_price_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Actualizar config
    config['model']['final_metrics'] = {
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'rmse': float(metrics['rmse'])
    }
    
    config_path = os.path.join(models_dir, 'trained', 'house_price_model.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    logger.info(f"✅ Model saved to {model_path}")
    return model_path, config_path

def register_model_in_mlflow(model, config, metrics, mlflow_tracking_uri):
    # PROTECCIÓN CRÍTICA: Si no hay URI, no hacemos nada de MLflow
    if not mlflow_tracking_uri:
        logger.info("No MLflow tracking URI provided, skipping MLflow logging (Local/CI mode)")
        return None

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("House Price Prediction - Production")
        
        with mlflow.start_run(run_name="production_training"):
            mlflow.log_params(config['model']['parameters'])
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
        logger.info("✅ Model logged to MLflow")
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")

def main():
    args = parse_args()

    # 1. Cargar Configuración
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Preparar Datos
    X_train, X_test, y_train, y_test = load_and_split_data(args.data, config)

    # 3. Entrenar
    model = get_model_instance(config['model']['best_model'], config['model']['parameters'])
    trained_model, metrics = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    # 4. Guardar Localmente
    save_model_artifacts(trained_model, config, metrics, args.models_dir)

    # 5. Registrar en MLflow (Solo si se pasó la URI, en GitHub Actions no se pasará)
    register_model_in_mlflow(trained_model, config, metrics, args.mlflow_tracking_uri)

    logger.info("🚀 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()