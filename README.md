# Detección de arritmias en tiempo real a partir del ECG

Proyecto final de la asignatura de Big Data del Máster Universitario en Sistemas Inteligentes (Universitat Jaume I, 2026), hecho en pareja con [gtognola](https://github.com/gtognola).

Un productor envía por Kafka la señal de un electrocardiograma como si llegara de un monitor, a 360 muestras por segundo. Un consumidor la recibe, predice cómo debería continuar cada latido y marca como posible arritmia el que se aleja demasiado de la predicción.

## Cómo funciona

- **Modelo:** una red convolucional de tipo codificador-decodificador, en PyTorch, que predice los 0,5 segundos siguientes de señal a partir de los 4 anteriores. Sigue el enfoque de [FADE](https://arxiv.org/abs/2502.07389) (Ruiz-Barroso et al., 2025).
- **Entrenamiento:** el modelo base se entrenó solo con latidos normales y con pacientes distintos en entrenamiento y en prueba, para evitar fugas de datos.
- **Ajuste fino por paciente:** con los primeros 50 latidos de cada paciente, el modelo se ajusta a él durante 15 épocas y el umbral se fija en el percentil 90 de su error.
- **Detección:** a partir de ahí, un latido es anómalo si su error de predicción (MAE normalizado) supera ese umbral.
- **Visualización:** un monitor hecho con matplotlib muestra la señal en tiempo real, el último latido y las anomalías detectadas.

## Datos

[MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), de PhysioNet, con licencia Open Data Commons Attribution v1.0. Tiene 48 registros de ECG de 30 minutos, con cada latido anotado por cardiólogos. `src/100.parquet` es el registro 100 y se incluye como ejemplo.

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology* 20(3):45-50 (2001).

## Estructura

```
deteccion-arritmias-ecg/
├── src/
│   ├── test_sine_producer.py    # productor: envía la señal por Kafka
│   ├── test_sine_plotter.py     # consumidor: detecta y muestra las anomalías
│   └── 100.parquet              # registro de ejemplo
└── models/
    ├── fade_ecg_model_base.pth  # modelo base entrenado
    ├── scaler_x.pkl             # normalización de la entrada
    └── scaler_y.pkl             # normalización de la salida
```

## Cómo ejecutarlo

1. Instala las dependencias:

   ```bash
   pip install numpy pandas pyarrow matplotlib torch scikit-learn joblib kafka-python
   ```

2. Arranca Kafka en `localhost:9092` y crea el topic:

   ```bash
   bin/kafka-topics.sh --create --topic ecg-raw-signal --bootstrap-server localhost:9092
   ```

3. Desde la carpeta `src/`, en dos terminales, arranca primero el monitor y después el productor, porque el monitor solo lee los mensajes nuevos:

   ```bash
   python test_sine_plotter.py    # terminal 1: detecta y muestra las anomalías
   python test_sine_producer.py   # terminal 2: envía la señal
   ```

## Autores

Diego Solsona y [gtognola](https://github.com/gtognola).
