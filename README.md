# TFG: Estudio Empírico de Importancia de Features en DRL para Finanzas

Este repositorio contiene el código fuente para el Trabajo de Fin de Grado (TFG) centrado en la evaluación de diferentes familias de características (features) financieras utilizando algoritmos de Deep Reinforcement Learning (DRL) para el trading cuantitativo.

## 📁 Estructura del Proyecto

```
TFG/
├── config/                                 # Archivos de configuración (JSON de features, CSV de escenarios, hiperparámetros)
│   ├── algorithms_config.json              # Configuracion de los hiperparámetros de los algoritmos
│   ├── feature_family.json                 # Json con las configuraciones de los features
│   ├── scenarios_config_1d_ablation.csv    # CSV con todos los escenarios para el experimento
├── experiments/                            # Scripts principales de los algoritmos e inferencia
│   ├── DQN.py                              # Entrenamiento con DQN
│   ├── PPO.py                              # Entrenamiento con PPO
│   ├── A2C.py                              # Entrenamiento con A2C
│   ├── smoke_test.py                       # Pruebas de integración rápida (pipeline completo)
│   ├── analyze_results.py                  # Análisis estadístico y tablas académicas
│   └── equity_curves.py                    # Gráficos de curvas de capital (Equity Curves)
└── src/                                    # Código fuente, utilidades y entorno
    ├── data_utils.py                       # Carga y preprocesamiento de datos
    ├── download_ds.py                      # Descarga automatizada de activos financieros
    ├── experiment_utils.py                 # Funciones comunes extraídas de los algoritmos
    ├── features.py                         # Lógica matemática de los indicadores técnicos
    └── trading_env_improved.py             # Entorno de Gym customizado con soporte de comisiones
```

## ⚙️ Requisitos e Instalación

Para ejecutar este proyecto, es recomendable utilizar un entorno virtual (venv o conda).
Las dependencias principales son `stable-baselines3`, `gymnasium`, `torch` y librerías de análisis de datos.

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Mac/Linux
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Uso y Ejecución

1. **Descarga de Datos:** Si la carpeta `data/raw/` está vacía, puedes descargar los datos ejecutando:

   ```bash
   python src/download_ds.py
   ```

2. **Entrenamiento y Experimentación:** Puedes ejecutar cualquiera de los algoritmos desde la raíz del proyecto o desde dentro de la carpeta `experiments/`:

   ```bash
   python experiments/DQN.py
   ```

   _Nota: Cada script leerá automáticamente las configuraciones desde `config/`._

3. **Análisis de Resultados:** Una vez ejecutados los modelos, genera las métricas y comparativas:

   ```bash
   python experiments/analyze_results.py
   ```

4. **Curvas de Capital:** Para visualizar el rendimiento:
   ```bash
   python experiments/equity_curves.py --asset Gold --timesteps 5000 --seed 42
   ```

## 🧠 Arquitectura de Features y Ablación

El estudio evalúa el impacto de la información agrupada por familias:

- **Trend (Tendencia):** SMA, EMA, MACD
- **Momentum:** RSI, Stochastic Oscillator
- **Volatility (Volatilidad):** Bollinger Bands, ATR, Realized Volatility
- **Statistical (Estadística Avanzada):** Variables rezagadas, Diferencias, Descomposición Temporal

Se aplican estudios de **ablación** eliminando selectivamente una o varias de estas familias para comprobar su peso real en la toma de decisiones del agente DRL frente a estrategias _Baseline_.
