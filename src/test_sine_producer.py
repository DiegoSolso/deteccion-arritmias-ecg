"""
ECG Data Producer - Simulador de ECG en Tiempo Real

Este módulo lee un archivo de señal ECG (en formato Parquet) y lo transmite
a través de Apache Kafka, simulando un dispositivo de ECG en tiempo real.

La señal se transmite punto por punto a la frecuencia de muestreo original (360Hz),
permitiendo que el consumidor (plotter) procese y visualice la señal como si
fuera un stream en tiempo real.

Autor: Proyecto Big Data - Universidad
"""

import time
import json
import numpy as np
import pandas as pd
from kafka import KafkaProducer
import os

# --- CONFIGURACIÓN ---
KAFKA_TOPIC = "ecg-raw-signal"
KAFKA_SERVER = "localhost:9092"

# RUTA AL ARCHIVO DE DATOS ECG
# Este archivo debe ser generado por el notebook de entrenamiento
# La ruta se resuelve relativa al directorio del script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_FILE = os.path.join(SCRIPT_DIR, "data", "individual_records", "patient_200.parquet")

# CONFIGURACIÓN DE STREAMING PUNTO A PUNTO
FS = 360.0        # Frecuencia de muestreo (Hz) - MIT-BIH estándar
SEND_RATE = 1.0 / FS  # Intervalo entre envíos (aprox 0.0027s)

def load_data(filepath):
    """
    Carga el archivo Parquet y prepara los arrays de señal y etiquetas.
    
    Args:
        filepath: Ruta al archivo .parquet con la señal ECG
        
    Returns:
        signal: Array numpy con los valores de la señal
        labels: Array con las anotaciones/etiquetas de cada punto
    """
    if not os.path.exists(filepath):
        print(f"❌ ERROR: No se encuentra el archivo: {filepath}")
        print(f"   Asegúrate de ejecutar primero el notebook de entrenamiento")
        print(f"   para generar los archivos de datos.")
        return None, None
        
    print(f"📂 Cargando archivo: {filepath} ...")
    df = pd.read_parquet(filepath)
    
    # Asumimos que la PRIMERA columna es la señal (o buscar 'signal')
    if 'signal' in df.columns:
        signal_col = 'signal'
    else:
        signal_col = df.columns[0]
    
    if 'annotation' in df.columns:
        df['annotation'] = df['annotation'].fillna('None')
        labels = df['annotation'].values
    else:
        labels = np.full(len(df), 'None')
        
    signal = df[signal_col].values
    
    print(f"✅ Datos cargados: {len(signal):,} muestras")
    print(f"   Frecuencia de transmisión: {FS}Hz")
    print(f"   Duración total: {len(signal)/FS:.1f} segundos")
    return signal, labels

def start_ecg_producer():
    """
    Inicia el productor de ECG que transmite la señal punto por punto.
    
    El productor:
    1. Lee un archivo de señal ECG (Parquet)
    2. Transmite cada punto a través de Kafka a 360Hz
    3. Incluye las etiquetas reales (ocultas) para validación
    4. Reinicia automáticamente al llegar al final del archivo
    """
    print("\n" + "="*60)
    print("🫀 ECG SINGLE-POINT PRODUCER")
    print("="*60)
    print(f"\n📡 Transmitiendo a {FS} Hz hacia: {KAFKA_TOPIC}")
    
    # 1. Cargar Datos
    signal, labels = load_data(PARQUET_FILE)
    if signal is None: 
        return

    # 2. Configurar Kafka
    print(f"\n🔌 Conectando a Kafka: {KAFKA_SERVER}...")
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
    print("✅ Conectado a Kafka")

    sequence = 0
    idx = 0
    total_samples = len(signal)
    start_time = time.time()

    print(f"\n🚀 Iniciando transmisión... (Ctrl+C para detener)\n")
    
    try:
        while True:
            # 3. Reinicio de bucle si llegamos al final
            if idx >= total_samples:
                print("\n>>> Fin del archivo. Reiniciando bucle... <<<\n")
                idx = 0
            
            # 4. Extraer UN solo dato
            val = float(signal[idx])
            label_val = labels[idx]
            
            # Etiqueta (si es 'None', mandamos 'Normal' para visualizar)
            label_to_send = label_val
            
            # Enviamos como lista [val] para mantener compatibilidad
            raw_point = [val]

            # 5. Crear Mensaje
            message = {
                "sensor_id": "bed_100_real",
                "timestamp": time.time(),
                "sequence_id": sequence,
                "hidden_label": label_to_send,
                "raw_signal_chunk": raw_point
            }

            # 6. Enviar
            producer.send(KAFKA_TOPIC, value=message)
            
            # 7. Solo imprimimos si etiqueta != 'None' (para no saturar la consola)
            if label_val != 'None':
                print(f"📨 Enviado punto #{sequence:,}: Value={val:.3f}, Label={label_val}"
                      f" (Index: {idx}/{total_samples})"
                    )
            # Avanzar
            idx += 1
            sequence += 1
            
            # Esperar el tiempo exacto de muestreo
            time.sleep(SEND_RATE)

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\n📊 RESUMEN DE TRANSMISIÓN:")
        print(f"   Tiempo total: {elapsed:.1f} segundos")
        print(f"   Puntos transmitidos: {sequence:,}")
        print(f"   Tasa efectiva: {sequence/elapsed:.1f} Hz")
        print("\n✅ Transmisión detenida.")
    finally:
        producer.close()

if __name__ == "__main__":
    start_ecg_producer()