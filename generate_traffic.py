import requests
import time
import random

# Configuración
API_URL = "http://localhost:8000"
NUM_REQUESTS = 5000  # Número total de consultas a generar

# Datos de ejemplo
LOCATIONS = ["Rural", "Suburb", "Urban", "Downtown", "Waterfront", "Mountain"]
CONDITIONS = ["Poor", "Fair", "Good", "Excellent"]


def generate_random_prediction():
    """Genera datos aleatorios para una predicción."""
    return {
        "sqft": random.uniform(1000, 4500),
        "bedrooms": random.randint(1, 6),
        "bathrooms": random.uniform(1, 4),
        "location": random.choice(LOCATIONS),
        "year_built": random.randint(1950, 2024),
        "condition": random.choice(CONDITIONS),
        "price_per_sqft": random.uniform(150, 600)
    }


def make_prediction(data):
    """Realiza una predicción."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=10
        )
        return response.status_code == 200
    except Exception:
        return False


def check_health():
    """Verifica que el API esté disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def main():
    """Función principal."""
    print("=" * 60)
    print("🚀 Generador de Tráfico para Grafana")
    print("=" * 60)
    print()
    
    # Verificar disponibilidad del API
    print("🔍 Verificando API...")
    if not check_health():
        print("❌ API no disponible en http://localhost:8000")
        print("⚠️  Ejecuta: docker-compose -f deployment/mlflow/docker-compose.yaml up -d")
        return
    
    print("✅ API disponible")
    print(f"📊 Generando {NUM_REQUESTS} consultas...\n")
    
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i in range(1, NUM_REQUESTS + 1):
        data = generate_random_prediction()
        success = make_prediction(data)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Mostrar progreso cada 500 consultas
        if i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (NUM_REQUESTS - i) / rate if rate > 0 else 0
            print(f"✅ {i}/{NUM_REQUESTS} consultas ({successful} OK, {failed} ERROR) - {rate:.1f} req/s - Quedan ~{remaining:.0f}s")
        
        # Pequeña pausa para no saturar
        time.sleep(0.01)
    
    elapsed_time = time.time() - start_time
    
    # Resumen final
    print()
    print("=" * 60)
    print("✨ ¡COMPLETADO!")
    print("=" * 60)
    print(f"✅ Exitosas: {successful}")
    print(f"❌ Fallidas: {failed}")
    print(f"📊 Total: {NUM_REQUESTS}")
    print(f"⏱️  Tiempo: {elapsed_time:.1f}s")
    print(f"📈 Velocidad: {NUM_REQUESTS/elapsed_time:.1f} req/s")
    print()
    print(f"📌 Grafana: http://localhost:3000")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelado por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
