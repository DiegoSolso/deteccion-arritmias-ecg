# 🫀 Sistema de Detección de Arritmias en ECG

## Proyecto Final - Big Data

Este proyecto implementa un sistema de detección de arritmias cardíacas en tiempo real utilizando:

- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Modelo**: LSTM Autoencoder para detección de anomalías
- **Streaming**: Apache Kafka para transmisión en tiempo real
- **Visualización**: Monitor en tiempo real con matplotlib

## 📁 Estructura del Proyecto

```
big-data-final/
├── exploration_and_training.ipynb  # Notebook principal de exploración y entrenamiento
├── src/
│   ├── test_sine_producer.py       # Productor: simula ECG en tiempo real via Kafka
│   ├── test_sine_plotter.py        # Consumidor: visualiza y detecta arritmias
│   └── 100.parquet                 # Datos de ejemplo (generado por el notebook)
├── models/
│   ├── ecg_forecaster.pth          # Modelo entrenado (generado por el notebook)
│   └── scalers.pkl                 # Parámetros de normalización
└── README.md
```

## 🚀 Guía de Uso

### 1. Ejecutar el Notebook

Abre `exploration_and_training.ipynb` en Google Colab o Jupyter y ejecuta todas las celdas para:

1. Descargar el dataset MIT-BIH de PhysioNet
2. Explorar y visualizar las señales ECG
3. Preprocesar y segmentar los datos
4. Entrenar el modelo LSTM Autoencoder
5. Evaluar el rendimiento
6. Exportar el modelo para uso en streaming

### 2. Configurar Kafka (Local)

```bash
# Iniciar Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Iniciar Kafka
bin/kafka-server-start.sh config/server.properties

# Crear topic
bin/kafka-topics.sh --create --topic ecg-raw-signal --bootstrap-server localhost:9092
```

### 3. Ejecutar el Sistema

Terminal 1 - Productor:
```bash
cd src
python test_sine_producer.py
```

Terminal 2 - Monitor/Plotter:
```bash
cd src
python test_sine_plotter.py
```

## 📊 Metodología

### Prevención de Data Leakage

El proyecto implementa una división a **nivel de paciente** para evitar data leakage:

- Los pacientes de entrenamiento son completamente diferentes de los de test
- El modelo nunca ve latidos del mismo paciente en train y test
- Esto refleja mejor el rendimiento real en pacientes nuevos

### Detección de Anomalías

El sistema utiliza un enfoque de **reconstrucción**:

1. El autoencoder se entrena solo con latidos normales
2. Aprende a predecir el patrón futuro del ECG
3. Las anomalías producen mayor error de reconstrucción
4. Un umbral calibrado clasifica normal vs arritmia

## 📚 Dependencias

```
numpy
pandas
matplotlib
seaborn
wfdb
torch
scikit-learn
kafka-python
pyarrow
```

## 👥 Autores

Proyecto desarrollado para el curso de Big Data - Universidad

## 📝 Licencia

Este proyecto es solo para fines educativos.
