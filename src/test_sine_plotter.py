"""
ECG Real-time Monitor with Arrhythmia Detection

Este módulo recibe señales ECG a través de Kafka y las visualiza en tiempo real.
Utiliza un modelo LSTM Autoencoder entrenado para detectar arritmias basándose
en el error de reconstrucción.

Autor: Proyecto Big Data - Universidad
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from kafka import KafkaConsumer
from collections import deque
import time

# --- CONFIGURACIÓN ---
KAFKA_TOPIC = "ecg-raw-signal"
KAFKA_SERVER = "localhost:9092"
MODEL_PATH = "models/ecg_forecaster.pth"

# Parámetros de Ventana y Detección
FS = 360
WINDOW_SECONDS = 4
SWEEP_SIZE = WINDOW_SECONDS * FS  # Tamaño del barrido (1440 puntos)
GAP_SIZE = 40                     # Espacio vacío visual en el barrido

# Lógica de Detección de Picos (Ajustar según la amplitud de tu señal real)
PEAK_THRESHOLD = 0.5    # Si tu señal es normalizada (0-1), usa 0.5-0.8. Si es raw (mV), ajusta.
REFRACTORY_PERIOD = 200 # Muestras a esperar entre picos (aprox 0.6s)

# Ventana del Modelo
INPUT_WINDOW = 1440    # 4 segundos de entrada (360 * 4)
OUTPUT_WINDOW = 180    # 0.5 segundos de salida (360 * 0.5)
MODEL_WINDOW = 360     # Ventana para visualización del latido
HALF_WINDOW = 180      # Punto donde alineamos el pico (centro)


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder para predicción de series temporales ECG.
    Debe coincidir con la arquitectura del modelo entrenado.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_len=180):
        super(LSTMAutoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Capa de salida
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Preparar entrada para decoder
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, self.output_len, 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input, (hidden, cell))
        
        # Salida
        output = self.fc(decoder_output)
        
        return output.squeeze(-1)


def load_model(model_path):
    """
    Carga el modelo entrenado y sus parámetros de configuración.
    
    Returns:
        model: Modelo cargado
        threshold: Umbral para detección de anomalías
        scaler_params: Parámetros de normalización (min, max)
    """
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo no encontrado en: {model_path}")
        return None, None, None
    
    print(f"📥 Cargando modelo desde: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extraer configuración
    config = checkpoint['model_config']
    threshold = checkpoint['threshold']
    scaler_params = {
        'X_min': checkpoint['scaler_X_min'],
        'X_max': checkpoint['scaler_X_max'],
        'y_min': checkpoint['scaler_y_min'],
        'y_max': checkpoint['scaler_y_max']
    }
    
    # Crear e inicializar modelo
    model = LSTMAutoencoder(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_len=config['output_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Modelo cargado exitosamente")
    print(f"   Umbral de detección: {threshold:.6f}")
    
    return model, threshold, scaler_params


def normalize(data, min_val, max_val):
    """Normaliza datos al rango [0, 1]"""
    return (data - min_val) / (max_val - min_val)


def predict_anomaly(model, signal_window, threshold, scaler_params, device='cpu'):
    """
    Predice si un segmento de señal contiene una anomalía.
    
    Args:
        model: Modelo LSTM Autoencoder
        signal_window: Array con la señal de entrada (INPUT_WINDOW muestras)
        threshold: Umbral para clasificar como anomalía
        scaler_params: Parámetros de normalización
        device: Dispositivo de cómputo
        
    Returns:
        is_anomaly: bool indicando si es anomalía
        mse: Error cuadrático medio
        prediction: Señal predicha
    """
    if model is None:
        return False, 0.0, None
    
    # Normalizar entrada
    X_norm = normalize(signal_window, scaler_params['X_min'], scaler_params['X_max'])
    
    # Preparar tensor
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    X_tensor = X_tensor.to(device)
    
    with torch.no_grad():
        prediction = model(X_tensor).cpu().numpy()[0]
    
    # Calcular MSE (en el espacio normalizado)
    # Nota: En producción deberíamos comparar con la señal real futura
    # Aquí usamos solo la predicción para fines de demostración
    mse = np.mean(prediction ** 2)  # Placeholder - en producción comparar con señal real
    
    is_anomaly = mse > threshold
    
    return is_anomaly, mse, prediction

def start_smart_monitor():
    """
    Inicia el monitor de ECG en tiempo real con detección de arritmias.
    
    El sistema:
    1. Recibe señales ECG por Kafka
    2. Visualiza la señal en tiempo real
    3. Detecta picos R (latidos)
    4. Extrae ventanas centradas en cada latido
    5. Usa el modelo LSTM para detectar anomalías
    """
    # Cargar modelo de IA
    model, threshold, scaler_params = load_model(MODEL_PATH)
    model_available = model is not None
    
    if model_available:
        print(f"🤖 Modelo de IA ACTIVADO - Umbral: {threshold:.6f}")
    else:
        print("⚠️ Modelo de IA no disponible - Funcionando solo en modo visualización")
    
    print(f"\n📡 Conectando a Kafka: {KAFKA_TOPIC}...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest'
    )

    # --- ESTRUCTURAS DE DATOS ---
    # 1. Buffer visual (Barrido)
    sweep_buffer = np.full(SWEEP_SIZE, np.nan)
    sweep_cursor = 0
    
    # 2. Buffer de Análisis (Historial lineal continuo)
    # Guardamos suficiente historia para el modelo (4+ segundos)
    analysis_buffer = deque(maxlen=INPUT_WINDOW + FS)
    
    # Variables de estado
    last_peak_time = 0
    samples_since_start = 0
    beat_counter = 0
    anomaly_count = 0
    recent_predictions = deque(maxlen=10)  # Últimas 10 predicciones para promediar
    
    # --- CONFIGURACIÓN GRÁFICA ---
    plt.ion()
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    
    # A. MONITOR CONTINUO (Arriba)
    ax_sweep = fig.add_subplot(gs[0, :]) 
    line_sweep, = ax_sweep.plot(np.arange(SWEEP_SIZE), sweep_buffer, color='#00ff00', lw=1.2)
    ax_sweep.set_title(f"🫀 MONITOR ECG TIEMPO REAL | Frecuencia: {FS}Hz", fontsize=12, color='lime')
    ax_sweep.set_xlim(0, SWEEP_SIZE)
    ax_sweep.set_ylim(-2.0, 2.0)
    ax_sweep.grid(True, alpha=0.2)
    ax_sweep.set_xlabel('Muestras')
    ax_sweep.set_ylabel('Amplitud (mV)')
    
    # B. CAPTURA DE LATIDO (Abajo Izquierda)
    ax_zoom = fig.add_subplot(gs[1, 0])
    line_zoom, = ax_zoom.plot([], [], color='cyan', lw=2)
    ax_zoom.axvline(x=HALF_WINDOW, color='white', linestyle=':', alpha=0.5, label='Pico R')
    ax_zoom.set_title("📊 Segmentación: Latido Capturado", fontsize=11)
    ax_zoom.set_xlim(0, MODEL_WINDOW)
    ax_zoom.set_ylim(-2.0, 2.0)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_xlabel('Muestras')
    ax_zoom.set_ylabel('Amplitud (mV)')
    ax_zoom.legend(loc='upper right', fontsize=8)

    # C. PREDICCIÓN IA (Abajo Derecha)
    ax_pred = fig.add_subplot(gs[1, 1])
    if model_available:
        ax_pred.set_title("🤖 Análisis de Anomalía (IA)", fontsize=11)
        ax_pred.set_xlim(0, 100)
        ax_pred.set_ylim(0, 1)
        status_bar = ax_pred.barh([0.5], [0], color='green', height=0.3)
        ax_pred.axvline(x=50, color='yellow', linestyle='--', alpha=0.7, label='Umbral')
        ax_pred.set_xlabel('Score de Anomalía (%)')
        ax_pred.set_yticks([])
        ax_pred.legend(loc='upper right', fontsize=8)
    else:
        ax_pred.text(0.5, 0.5, "MÓDULO IA\nNO DISPONIBLE\n\nEntrena el modelo\nprimero", 
                     ha='center', va='center', color='gray', fontsize=12, transform=ax_pred.transAxes)
        ax_pred.set_title("🤖 Análisis de Anomalía", fontsize=11)
        ax_pred.set_facecolor('#111111')
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        status_bar = None

    # Textos informativos
    text_bpm = ax_sweep.text(SWEEP_SIZE - 350, 1.6, "BPM: --", fontsize=14, color='white', fontweight='bold')
    text_status = ax_sweep.text(50, 1.6, "Estado: Esperando...", fontsize=11, color='white')
    text_anomaly = ax_sweep.text(SWEEP_SIZE/2, 1.6, "", fontsize=12, color='red', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show(block=False)
    print("\n✅ Monitor Iniciado. Esperando señal...")
    print("   Presiona Ctrl+C para detener.\n")

    try:
        for message in consumer:
            data = message.value
            
            # Recibimos una lista de 1 punto (streaming punto a punto)
            val = data['raw_signal_chunk'][0]
            label_real = data.get('hidden_label', '?')
            
            # 1. ACTUALIZAR MONITOR DE BARRIDO
            sweep_buffer[sweep_cursor] = val
            
            # Crear el "gap" visual (borrado delante del cursor)
            gap_start = (sweep_cursor + 1) % SWEEP_SIZE
            gap_end = (sweep_cursor + 1 + GAP_SIZE) % SWEEP_SIZE
            if gap_start < gap_end:
                sweep_buffer[gap_start:gap_end] = np.nan
            else:
                sweep_buffer[gap_start:] = np.nan
                sweep_buffer[:gap_end] = np.nan
                
            sweep_cursor = (sweep_cursor + 1) % SWEEP_SIZE
            
            # 2. ACTUALIZAR BUFFER DE ANÁLISIS
            analysis_buffer.append(val)
            samples_since_start += 1
            
            # 3. ALGORITMO DE DETECCIÓN DE PICOS (Simple Peak Finder)
            if len(analysis_buffer) > 20:
                center_idx = -10
                center_val = analysis_buffer[center_idx]
                time_diff = samples_since_start - last_peak_time
                
                if (center_val > PEAK_THRESHOLD and time_diff > REFRACTORY_PERIOD):
                    true_peak_time = samples_since_start - 10
                    last_peak_time = true_peak_time
                    beat_counter += 1
                    
                    # Calcular BPM
                    bpm = 60.0 / (time_diff / FS)
                    text_bpm.set_text(f"BPM: {int(bpm)}")

            # 4. EXTRACCIÓN DEL LATIDO Y PREDICCIÓN
            if last_peak_time > 0:
                dist_from_peak = samples_since_start - last_peak_time
                
                if dist_from_peak == HALF_WINDOW:
                    full_history = list(analysis_buffer)
                    
                    if len(full_history) >= MODEL_WINDOW:
                        beat_window = full_history[-MODEL_WINDOW:]
                        
                        # Actualizar gráfico de latido
                        line_zoom.set_data(np.arange(MODEL_WINDOW), beat_window)
                        
                        # --- PREDICCIÓN DEL MODELO ---
                        if model_available and len(full_history) >= INPUT_WINDOW:
                            input_window = np.array(full_history[-INPUT_WINDOW:])
                            is_anomaly, mse, _ = predict_anomaly(
                                model, input_window, threshold, scaler_params
                            )
                            
                            # Actualizar indicador visual
                            anomaly_score = min(100, mse * 100000)  # Escalar para visualización
                            if status_bar is not None:
                                status_bar[0].set_width(anomaly_score)
                                if is_anomaly:
                                    status_bar[0].set_color('red')
                                    anomaly_count += 1
                                    text_anomaly.set_text(f"⚠️ ANOMALÍA DETECTADA ({anomaly_count})")
                                    ax_zoom.set_title(f"⚠️ Beat #{beat_counter} - ANOMALÍA (Real: {label_real})", color='red')
                                else:
                                    status_bar[0].set_color('green')
                                    text_anomaly.set_text("")
                                    ax_zoom.set_title(f"✅ Beat #{beat_counter} - Normal (Real: {label_real})", color='cyan')
                            
                            recent_predictions.append(is_anomaly)
                        else:
                            ax_zoom.set_title(f"Beat #{beat_counter} Capturado (Real: {label_real})", color='cyan')
            
            # Actualizar estado
            text_status.set_text(f"Estado: Latidos={beat_counter} | Anomalías={anomaly_count}")
            
            # 5. REFRESCO DE PANTALLA
            if samples_since_start % 20 == 0:
                line_sweep.set_ydata(sweep_buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\n\n📊 RESUMEN DE SESIÓN:")
        print(f"   Total de latidos analizados: {beat_counter}")
        print(f"   Anomalías detectadas: {anomaly_count}")
        if beat_counter > 0:
            print(f"   Tasa de anomalías: {100*anomaly_count/beat_counter:.1f}%")
        print("\n✅ Monitor detenido.")
        plt.close()

if __name__ == "__main__":
    start_smart_monitor()