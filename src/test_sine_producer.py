import time
import json
import numpy as np
import pandas as pd
from kafka import KafkaProducer
import os

# --- CONFIGURACIÓN ---
KAFKA_TOPIC = "ecg-raw-signal"
KAFKA_SERVER = "localhost:9092"

# RUTA A TU ARCHIVO
PARQUET_FILE = "src/100.parquet" 

# CONFIGURACIÓN DE STREAMING PUNTO A PUNTO
FS = 360.0        # Frecuencia de muestreo (Hz)
SEND_RATE = 1.0 / FS  # Intervalo entre envíos (aprox 0.0027s)

def load_data(filepath):
    """Carga el parquet y prepara los arrays de señal y etiquetas."""
    if not os.path.exists(filepath):
        print(f"ERROR: No se encuentra el archivo: {filepath}")
        return None, None
        
    print(f"Cargando archivo: {filepath} ...")
    df = pd.read_parquet(filepath)
    
    # Asumimos que la PRIMERA columna es la señal
    signal_col = df.columns[0]
    
    if 'annotation' in df.columns:
        df['annotation'] = df['annotation'].fillna('None')
        labels = df['annotation'].values
    else:
        labels = np.full(len(df), 'None')
        
    signal = df[signal_col].values
    
    print(f"Datos cargados: {len(signal)} muestras. Frecuencia simulada: {FS}Hz")
    return signal, labels

def start_ecg_producer():
    print(f"--- ECG SINGLE-POINT PRODUCER ---")
    print(f"Transmitiendo a {FS} Hz (1 dato a la vez) hacia: {KAFKA_TOPIC}")
    
    # 1. Cargar Datos
    signal, labels = load_data(PARQUET_FILE)
    if signal is None: return

    # 2. Configurar Kafka
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

    sequence = 0
    idx = 0
    total_samples = len(signal)

    try:
        while True:
            # 3. Reinicio de bucle si llegamos al final
            if idx >= total_samples:
                print(">>> Fin del archivo. Reiniciando bucle... <<<")
                idx = 0
            
            # 4. Extraer UN solo dato
            val = float(signal[idx])
            label_val = labels[idx]
            
            # Etiqueta (si es 'None', mandamos 'Normal' para visualizar)
            label_to_send = label_val if label_val != 'None' else "Normal"
            
            # IMPORTANTE: Enviamos como lista [val] para mantener compatibilidad
            # con esquemas que esperen Array, aunque sea de longitud 1.
            raw_point = [val]

            # 5. Crear Mensaje
            message = {
                "sensor_id": "bed_100_real",
                "timestamp": time.time(),
                "sequence_id": sequence,
                "hidden_label": label_to_send,
                "raw_signal_chunk": raw_point # Ahora es un "chunk" de 1 elemento
            }

            # 6. Enviar
            producer.send(KAFKA_TOPIC, value=message)
            
            # Solo imprimimos info cada 1 segundo (360 muestras) para no saturar la consola
            if sequence % 360 == 0:
                print(f"Stream {FS}Hz | Seq {sequence}: Dato {val:.3f} enviado. (Etiqueta: {label_to_send})")
            
            # Avanzar
            idx += 1
            sequence += 1
            
            # Esperar el tiempo exacto de muestreo
            time.sleep(SEND_RATE)

    except KeyboardInterrupt:
        print("\nTransmisión detenida.")
    finally:
        producer.close()

if __name__ == "__main__":
    start_ecg_producer()