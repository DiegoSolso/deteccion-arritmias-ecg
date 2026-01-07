import json
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf  <-- COMENTADO PARA ESTA PRUEBA
from kafka import KafkaConsumer
from collections import deque
import time

# --- CONFIGURACIÓN ---
KAFKA_TOPIC = "ecg-raw-signal"
KAFKA_SERVER = "localhost:9092"
# MODEL_PATH = "models/ecg_forecaster.h5" 

# Parámetros de Ventana y Detección
FS = 360
WINDOW_SECONDS = 4
SWEEP_SIZE = WINDOW_SECONDS * FS  # Tamaño del barrido (1440 puntos)
GAP_SIZE = 40                     # Espacio vacío visual en el barrido

# Lógica de Detección de Picos (Ajustar según la amplitud de tu señal real)
PEAK_THRESHOLD = 0.5    # Si tu señal es normalizada (0-1), usa 0.5-0.8. Si es raw (mV), ajusta.
REFRACTORY_PERIOD = 200 # Muestras a esperar entre picos (aprox 0.6s)

# Ventana del Latido (Lo que extraeremos)
MODEL_WINDOW = 360     # 360 muestras total (1 segundo)
HALF_WINDOW = 180      # Punto donde alineamos el pico (centro)

def start_smart_monitor():
    # model = load_ai_model() <-- SALTAMOS ESTO
    model = None
    
    print(f"Conectando a Kafka: {KAFKA_TOPIC}...")
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
    # Guardamos 2 segundos de historia para poder mirar "hacia atrás" y encontrar picos
    analysis_buffer = deque(maxlen=FS * 2) 
    
    # Variables de estado
    last_peak_time = 0
    samples_since_start = 0
    beat_counter = 0
    
    # --- CONFIGURACIÓN GRÁFICA ---
    plt.ion()
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    
    # A. MONITOR CONTINUO (Arriba)
    ax_sweep = fig.add_subplot(gs[0, :]) 
    line_sweep, = ax_sweep.plot(np.arange(SWEEP_SIZE), sweep_buffer, color='#00ff00', lw=1.2)
    ax_sweep.set_title(f"MONITOR ECG TIEMPO REAL | Frecuencia: {FS}Hz", fontsize=10, color='lime')
    ax_sweep.set_xlim(0, SWEEP_SIZE)
    ax_sweep.set_ylim(-2.0, 2.0) # Ajusta esto si tu señal se sale
    ax_sweep.grid(True, alpha=0.2)
    
    # B. CAPTURA DE LATIDO (Abajo Izquierda)
    ax_zoom = fig.add_subplot(gs[1, 0])
    line_zoom, = ax_zoom.plot([], [], color='cyan', lw=2)
    # Dibujamos una línea vertical donde debería estar el pico R (en el centro)
    ax_zoom.axvline(x=HALF_WINDOW, color='white', linestyle=':', alpha=0.5)
    ax_zoom.set_title("Segmentación: Latido Capturado (Centrado en R)", fontsize=10)
    ax_zoom.set_xlim(0, MODEL_WINDOW)
    ax_zoom.set_ylim(-2.0, 2.0)
    ax_zoom.grid(True, alpha=0.3)

    # C. PREDICCIÓN IA (Abajo Derecha) - DESACTIVADO POR AHORA
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_pred.text(0.5, 0.5, "MÓDULO IA\nDESACTIVADO", 
                 ha='center', va='center', color='gray', fontsize=14, transform=ax_pred.transAxes)
    ax_pred.set_title("Análisis de Anomalía", fontsize=10)
    ax_pred.set_facecolor('#111111')
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    # Textos
    text_bpm = ax_sweep.text(SWEEP_SIZE - 200, 1.5, "BPM: --", fontsize=14, color='white', fontweight='bold')
    
    plt.show(block=False)
    print("Monitor Iniciado. Esperando señal...")

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
            # Necesitamos tener suficiente historia (al menos 20 muestras)
            if len(analysis_buffer) > 20:
                # Miramos el valor hace 10 muestras (center_val)
                # para ver si es mayor que sus vecinos (pico local)
                center_idx = -10
                center_val = analysis_buffer[center_idx]
                
                # Tiempo desde el último pico detectado
                time_diff = samples_since_start - last_peak_time
                
                # CONDICIÓN DE PICO:
                # a) Amplitud > Umbral
                # b) Pasó el periodo refractario (no detectar picos dobles muy pegados)
                # c) Es un máximo local (mayor que el anterior y el siguiente inmediato) - opcional pero recomendado
                if (center_val > PEAK_THRESHOLD and 
                    time_diff > REFRACTORY_PERIOD):
                    
                    # ¡PICO CONFIRMADO!
                    # Nota: El pico ocurrió realmente hace 10 muestras (center_idx)
                    true_peak_time = samples_since_start - 10
                    last_peak_time = true_peak_time
                    beat_counter += 1
                    
                    # Calcular BPM
                    bpm = 60.0 / (time_diff / FS)
                    text_bpm.set_text(f"BPM: {int(bpm)}")
                    
                    # Visual: Marcar pico en el sweep (opcional, complejo de dibujar dinámicamente)
                    # print(f"Pico R detectado #{beat_counter} (BPM: {int(bpm)})")

            # 4. EXTRACCIÓN DEL LATIDO COMPLETO
            # Queremos extraer una ventana de 360 muestras CENTRADA en el último pico.
            # 180 antes <--- PICO ---> 180 después.
            # Para hacer esto, tenemos que esperar a que el 'samples_since_start'
            # sea igual a (last_peak_time + 180). Es decir, ya han llegado 180 datos nuevos tras el pico.
            
            if last_peak_time > 0:
                dist_from_peak = samples_since_start - last_peak_time
                
                if dist_from_peak == HALF_WINDOW: # Justo ahora tenemos la mitad derecha completa
                    
                    # Convertir deque a lista para slicing
                    full_history = list(analysis_buffer)
                    
                    # El pico está en el índice: len(history) - 181
                    # Queremos: [pico_idx - 180 : pico_idx + 180]
                    # Dado que acabamos de llegar a pico + 180, simplemente tomamos los últimos 360
                    if len(full_history) >= MODEL_WINDOW:
                        beat_window = full_history[-MODEL_WINDOW:]
                        
                        # --- ACTUALIZAR GRÁFICO ZOOM ---
                        line_zoom.set_data(np.arange(MODEL_WINDOW), beat_window)
                        ax_zoom.set_title(f"Beat #{beat_counter} Capturado (Real: {label_real})", color='cyan')
                        
                        # --- AQUÍ IRÍA LA PREDICCIÓN DEL MODELO ---
                        # X_input = np.array(beat_window)[:HALF_WINDOW].reshape(1, HALF_WINDOW, 1)
                        # prediction = model(X_input)
                        # ...
            
            # 5. REFRESCO DE PANTALLA (Downsampling para rendimiento)
            if samples_since_start % 20 == 0:
                line_sweep.set_ydata(sweep_buffer)
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        print("\nMonitor detenido.")
        plt.close()

if __name__ == "__main__":
    start_smart_monitor()