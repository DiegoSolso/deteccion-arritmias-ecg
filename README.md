# 🫀 Real-Time ECG Arrhythmia Detection System

## Final Project - Big Data

This project implements a real-time cardiac arrhythmia detection system using:

- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet)
- **Model**: LSTM Autoencoder for anomaly detection
- **Streaming**: Apache Kafka for real-time data transmission
- **Visualization**: Real-time monitor with matplotlib

## 📁 Project Structure

```
big-data-final/
├── src/
│   ├── test_sine_producer.py       # Producer: simulates real-time ECG via Kafka
│   ├── test_sine_plotter.py        # Monitor: visualizes and detects arrhythmias in realtime
│   ├── 100.parquet                 # Example data
│   └── ... (other scripts or auxiliary files)
├── models/
│   ├── fade_ecg_model_base.pth     # Trained model
│   ├── scaler_x.pkl                # Normalization parameters X
│   └── scaler_y.pkl                # Normalization parameters Y
└── README.md
```

## 🚀 Usage Guide

### 1. Configure Kafka (Local)

```bash
# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# Start Kafka
bin/kafka-server-start.sh config/server.properties

# Create topic
bin/kafka-topics.sh --create --topic ecg-raw-signal --bootstrap-server localhost:9092
```

### 2. Run the System Scripts

Open two terminals in the project root folder and run the following commands:

**Terminal 1 - Signal Producer (ECG simulation):**
```bash
cd src
python test_sine_producer.py
```

**Terminal 2 - Monitor/Plotter (arrhythmia detection and visualization):**
```bash
cd src
python test_sine_plotter.py
```

**Other scripts in `src/`**

If you add new scripts for processing, analysis, or utilities, run them in the same way:
```bash
cd src
python your_script_name.py
```

**Notes:**
- Make sure you have the models and scalers in the `models/` folder with the correct names (`fade_ecg_model_base.pth`, `scaler_x.pkl`, `scaler_y.pkl`).
- If you use additional data files, place them in the appropriate folder and adjust the paths in the scripts as needed.

## 📊 Methodology

### Data Leakage Prevention

The project implements a **patient-level split** to avoid data leakage:

- Training and test patients are completely different
- The model never sees beats from the same patient in both train and test
- This better reflects real-world performance on new patients

### Anomaly Detection

The system uses a **reconstruction-based** approach:

1. The autoencoder is trained only with normal beats
2. It learns to predict the future ECG pattern
3. Anomalies produce higher reconstruction error
4. A calibrated threshold classifies normal vs arrhythmia

## 📚 Dependencies

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

## 👥 Authors

Developed for the Big Data course - University

## 📝 License

This project is for educational purposes only.