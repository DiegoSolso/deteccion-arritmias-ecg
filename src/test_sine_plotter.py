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
import torch.optim as optim
from kafka import KafkaConsumer
from collections import deque
import time
import joblib

# --- AUXILIARY LOSS AND METRIC (from aux.py) ---
class SplitMSELoss(nn.Module):
    def __init__(self, w1=5.0, w2=1.0, threshold=0.4):
        """Pesos optimizados para FADE: w1 alto para aprender la forma base."""
        super(SplitMSELoss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.threshold = threshold
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        loss = self.mse(pred, target)
        mask_outer = (torch.abs(target) > self.threshold).float()
        mask_inner = (torch.abs(target) <= self.threshold).float()
        weighted_loss = (self.w1 * loss * mask_inner) + (self.w2 * loss * mask_outer)
        return torch.mean(weighted_loss)


def calculate_nmae(y_true, y_pred):
    """Métrica NMAE para calcular el umbral y detectar anomalías."""
    # Evitar división por cero
    eps = 1e-8

    mae = np.mean(np.abs(y_true - y_pred), axis=1)

    # Normalización por instancia (row-wise)
    min_true = y_true.min(axis=1, keepdims=True)
    max_true = y_true.max(axis=1, keepdims=True)
    range_true = max_true - min_true + eps

    min_pred = y_pred.min(axis=1, keepdims=True)
    max_pred = y_pred.max(axis=1, keepdims=True)
    range_pred = max_pred - min_pred + eps

    norm_true = (y_true - min_true) / range_true
    norm_pred = (y_pred - min_pred) / range_pred

    norm_abs_diff = np.mean(np.abs(norm_true - norm_pred), axis=1)

    return mae * norm_abs_diff

# --- CLASE PRINCIPAL DE STREAMING ---


# --- CONFIGURACIÓN ---
KAFKA_TOPIC = "ecg-raw-signal"
KAFKA_SERVER = "localhost:9092"


# --- MODELO Y SCALERS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "fade_ecg_model_base.pth")
SCALER_X_PATH = os.path.join(PROJECT_DIR, "models", "scaler_x.pkl")
SCALER_Y_PATH = os.path.join(PROJECT_DIR, "models", "scaler_y.pkl")

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

# Visualización de anomalías
# Factor de escala para convertir MSE en porcentaje visual (0-100)
# Calculado como 100 / umbral típico para que el umbral esté al 50% de la barra
ANOMALY_VISUAL_SCALE = 50.0  # Se ajustará dinámicamente basado en el umbral del modelo



# --- MODELO BASE: Debe coincidir con fade_ecg_model_base.pth ---

# --- MODELO (Encoder-Decoder 1D) ---
class ECGForecaster(nn.Module):
    def __init__(self, input_len, output_len):
        super(ECGForecaster, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Cálculo dinámico del tamaño tras las convoluciones/pooling
        reduced_len = input_len // 4
        self.flatten_dim = 128 * reduced_len

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * (output_len // 4))
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1)
        )
        self.output_len = output_len

    def forward(self, x):
        # x: (batch, seq_len) or (batch, 1, seq_len)
        if x.ndim == 3 and x.shape[1] == 1:
            x = x
        elif x.ndim == 2:
            x = x.unsqueeze(1)
        else:
            raise ValueError("Input shape must be (batch, seq_len) or (batch, 1, seq_len)")
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = x.view(x.size(0), 128, self.output_len // 4)
        x = self.decoder(x)
        # Ajuste fino del tamaño de salida si es necesario
        if x.shape[2] != self.output_len:
            x = nn.functional.interpolate(x, size=self.output_len, mode='linear')
        return x.squeeze(1)



# --- LOAD SCALERS ---
def load_scalers(scaler_x_path, scaler_y_path):
    scaler_x = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return scaler_x, scaler_y



# --- NORMALIZE USING SCALERS ---
def normalize(data, scaler):
    # scaler must have transform method (sklearn MinMaxScaler or similar)
    return scaler.transform(data.reshape(-1, 1)).flatten()



# --- CALIBRATION AND DETECTION LOGIC ---
class StreamingECGMonitor:
    def __init__(self, base_model_path, scaler_x, scaler_y, input_len, output_len, device='cpu', calibration_beats=50):
        self.device = torch.device(device)
        self.calibration_limit = calibration_beats
        self.model = ECGForecaster(input_len, output_len)
        self.model.load_state_dict(torch.load(base_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.buffer_X = []
        self.buffer_Y = []
        self.is_calibrated = False
        self.dynamic_threshold = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.criterion = SplitMSELoss(w1=5.0, w2=1.0, threshold=0.4)
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
    def process_beat(self, x_past, y_future):
        # Normalize
        x_past_scaled = normalize(np.array(x_past), self.scaler_x)
        y_future_scaled = normalize(np.array(y_future), self.scaler_y)
        if not self.is_calibrated:
            self.buffer_X.append(x_past_scaled)
            self.buffer_Y.append(y_future_scaled)
            count = len(self.buffer_X)
            if count >= self.calibration_limit:
                self._run_adaptation()
                self.is_calibrated = True
                return {"status": "READY", "msg": "Sistema calibrado y activo."}
            return {"status": "CALIBRATING", "progress": f"{count}/{self.calibration_limit}"}
        else:
            return self._detect(x_past_scaled, y_future_scaled)
    def _run_adaptation(self):
        self.model.train()
        X_arr = np.array(self.buffer_X)
        Y_arr = np.array(self.buffer_Y)
        X_tensor = torch.FloatTensor(X_arr).unsqueeze(1).to(self.device)
        Y_tensor = torch.FloatTensor(Y_arr).unsqueeze(1).to(self.device)
        epochs = 15
        for _ in range(epochs):
            self.optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = self.criterion(preds, Y_tensor)
            loss.backward()
            self.optimizer.step()
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor).cpu().numpy().squeeze()
            errors = calculate_nmae(Y_arr, reconstructions)
            self.dynamic_threshold = np.percentile(errors, 90.0)
        self.buffer_X = []
        self.buffer_Y = []
    def _detect(self, x_past, y_future):
        input_tensor = torch.FloatTensor(x_past).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor).cpu().numpy().squeeze()
        error = calculate_nmae(y_future.reshape(1, -1), prediction.reshape(1, -1))[0]
        is_anomaly = error > self.dynamic_threshold
        return {
            "status": "MONITORING",
            "is_anomaly": bool(is_anomaly),
            "error_value": float(error),
            "threshold": float(self.dynamic_threshold)
        }


def start_smart_monitor():
    # Load scalers
    scaler_x, scaler_y = load_scalers(SCALER_X_PATH, SCALER_Y_PATH)
    monitor = StreamingECGMonitor(
        base_model_path=MODEL_PATH,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        input_len=INPUT_WINDOW,
        output_len=OUTPUT_WINDOW,
        device='cpu',
        calibration_beats=50
    )
    print(f"\n📡 Conectando a Kafka: {KAFKA_TOPIC}...")
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_SERVER],
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        auto_offset_reset='latest'
    )
    sweep_buffer = np.full(SWEEP_SIZE, np.nan)
    sweep_cursor = 0
    analysis_buffer = deque(maxlen=INPUT_WINDOW + FS)
    last_peak_time = 0
    samples_since_start = 0
    beat_counter = 0
    anomaly_count = 0
    plt.ion()
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax_sweep = fig.add_subplot(gs[0, :])
    line_sweep, = ax_sweep.plot(np.arange(SWEEP_SIZE), sweep_buffer, color='#00ff00', lw=1.2)
    ax_sweep.set_title(f"MONITOR ECG TIEMPO REAL | Frecuencia: {FS}Hz", fontsize=12, color='lime')
    ax_sweep.set_xlim(0, SWEEP_SIZE)
    ax_sweep.set_ylim(-2.0, 2.0)
    ax_sweep.grid(True, alpha=0.2)
    ax_sweep.set_xlabel('Muestras')
    ax_sweep.set_ylabel('Amplitud (mV)')
    ax_zoom = fig.add_subplot(gs[1, 0])
    line_zoom, = ax_zoom.plot([], [], color='cyan', lw=2)
    ax_zoom.axvline(x=HALF_WINDOW, color='white', linestyle=':', alpha=0.5, label='Pico R')
    ax_zoom.set_title("Segmentación: Latido Capturado", fontsize=11)
    ax_zoom.set_xlim(0, MODEL_WINDOW)
    ax_zoom.set_ylim(-2.0, 2.0)
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_xlabel('Muestras')
    ax_zoom.set_ylabel('Amplitud (mV)')
    ax_zoom.legend(loc='upper right', fontsize=8)
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.set_title("Análisis de Arritmia (IA)", fontsize=11)
    ax_pred.set_xlim(0, 100)
    ax_pred.set_ylim(0, 1)
    status_bar = ax_pred.barh([0.5], [0], color='green', height=0.3)
    ax_pred.axvline(x=50, color='yellow', linestyle='--', alpha=0.7, label='Umbral')
    ax_pred.set_xlabel('Score de Anomalía (%)')
    ax_pred.set_yticks([])
    ax_pred.legend(loc='upper right', fontsize=8)
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
            val = data['raw_signal_chunk'][0]
            label_real = data.get('hidden_label', '?')
            sweep_buffer[sweep_cursor] = val
            gap_start = (sweep_cursor + 1) % SWEEP_SIZE
            gap_end = (sweep_cursor + 1 + GAP_SIZE) % SWEEP_SIZE
            if gap_start < gap_end:
                sweep_buffer[gap_start:gap_end] = np.nan
            else:
                sweep_buffer[gap_start:] = np.nan
                sweep_buffer[:gap_end] = np.nan
            sweep_cursor = (sweep_cursor + 1) % SWEEP_SIZE
            analysis_buffer.append(val)
            samples_since_start += 1
            if len(analysis_buffer) > 20:
                center_idx = -10
                center_val = analysis_buffer[center_idx]
                time_diff = samples_since_start - last_peak_time
                if (center_val > PEAK_THRESHOLD and time_diff > REFRACTORY_PERIOD):
                    true_peak_time = samples_since_start - 10
                    last_peak_time = true_peak_time
                    beat_counter += 1
                    bpm = 60.0 / (time_diff / FS)
                    text_bpm.set_text(f"BPM: {int(bpm)}")
            if last_peak_time > 0:
                dist_from_peak = samples_since_start - last_peak_time
                if dist_from_peak == HALF_WINDOW:
                    full_history = list(analysis_buffer)
                    if len(full_history) >= MODEL_WINDOW:
                        beat_window = full_history[-MODEL_WINDOW:]
                        if len(full_history) >= INPUT_WINDOW + OUTPUT_WINDOW:
                            input_window = np.array(full_history[-(INPUT_WINDOW + OUTPUT_WINDOW):-OUTPUT_WINDOW])
                            future_window = np.array(full_history[-OUTPUT_WINDOW:])
                            result = monitor.process_beat(input_window, future_window)
                            if result["status"] == "CALIBRATING":
                                ax_zoom.set_title(f"Calibrando... {result['progress']}", color='yellow')
                                text_anomaly.set_text(f"Calibrando modelo... {result['progress']}")
                                status_bar[0].set_color('yellow')
                                status_bar[0].set_width(0)
                                line_zoom.set_data([], [])
                                if hasattr(ax_zoom, 'lines') and len(ax_zoom.lines) > 1:
                                    ax_zoom.lines[1].set_data([], [])
                            elif result["status"] == "READY":
                                ax_zoom.set_title(f"Calibración completada", color='cyan')
                                text_anomaly.set_text(f"Modelo calibrado. ¡Listo!")
                                status_bar[0].set_color('green')
                                status_bar[0].set_width(0)
                                line_zoom.set_data([], [])
                                if hasattr(ax_zoom, 'lines') and len(ax_zoom.lines) > 1:
                                    ax_zoom.lines[1].set_data([], [])
                            elif result["status"] == "MONITORING":
                                anomaly_score = min(100, (result["error_value"] / result["threshold"]) * ANOMALY_VISUAL_SCALE)
                                status_bar[0].set_width(anomaly_score)
                                if result["is_anomaly"]:
                                    status_bar[0].set_color('red')
                                    anomaly_count += 1
                                    text_anomaly.set_text(f"ARRITMIA DETECTADA ({anomaly_count})")
                                    ax_zoom.set_title(f"Beat #{beat_counter} - ARRITMIA", color='red') # (Real: {label_real})
                                    # Plot real arrhythmic beat and predicted one, aligned to R peak
                                    line_zoom.set_data(np.arange(MODEL_WINDOW), beat_window)
                                    input_tensor = torch.FloatTensor(normalize(np.array(input_window), monitor.scaler_x)).unsqueeze(0).unsqueeze(1).to(monitor.device)
                                    with torch.no_grad():
                                        pred = monitor.model(input_tensor).cpu().numpy().squeeze()
                                    if pred.ndim == 2:
                                        pred = pred[0]
                                    pred_start = HALF_WINDOW
                                    pred_x = np.arange(pred_start, pred_start + len(pred))
                                    if hasattr(ax_zoom, 'lines') and len(ax_zoom.lines) > 1:
                                        ax_zoom.lines[1].set_data(pred_x, pred)
                                        ax_zoom.lines[1].set_color('magenta')
                                        ax_zoom.lines[1].set_label('Predicción')
                                    else:
                                        line_pred, = ax_zoom.plot(pred_x, pred, color='magenta', lw=2, label='Predicción')
                                        ax_zoom.legend(loc='upper right', fontsize=8)
                                    # Persist arrhythmic beat and its title until next arrhythmic
                                    last_arrhythmic_beat = (beat_window.copy(), pred.copy(), pred_x.copy(), label_real, beat_counter)
                                    last_arrhythmic_title = f"Beat #{beat_counter} - ARRITMIA" # (Real: {label_real})
                                else:
                                    status_bar[0].set_color('green')
                                    text_anomaly.set_text("")
                                    # Only update plot, not title, for normal beats
                                    if 'last_arrhythmic_beat' in locals():
                                        beat_window, pred, pred_x, label_real, beat_counter = last_arrhythmic_beat
                                        line_zoom.set_data(np.arange(MODEL_WINDOW), beat_window)
                                        if hasattr(ax_zoom, 'lines') and len(ax_zoom.lines) > 1:
                                            ax_zoom.lines[1].set_data(pred_x, pred)
                                            ax_zoom.lines[1].set_color('magenta')
                                            ax_zoom.lines[1].set_label('Predicción')
                                        # Restore last arrhythmic title
                                        if 'last_arrhythmic_title' in locals():
                                            ax_zoom.set_title(last_arrhythmic_title, color='red')
                                    else:
                                        line_zoom.set_data([], [])
                                        if hasattr(ax_zoom, 'lines') and len(ax_zoom.lines) > 1:
                                            ax_zoom.lines[1].set_data([], [])
            text_status.set_text(f"Estado: Latidos={beat_counter} | Arritmias={anomaly_count}")
            if samples_since_start % 20 == 0:
                line_sweep.set_ydata(sweep_buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("\n\nRESUMEN DE SESIÓN:")
        print(f"   Total de latidos analizados: {beat_counter}")
        print(f"   Arritmias detectadas: {anomaly_count}")
        if beat_counter > 0:
            print(f"   Tasa de arritmias: {100*anomaly_count/beat_counter:.1f}%")
        print("\nMonitor detenido.")
        plt.close()

if __name__ == "__main__":
    start_smart_monitor()