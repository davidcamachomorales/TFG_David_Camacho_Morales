# TFG: Estudio empírico de familias de características financieras en agentes de aprendizaje por refuerzo profundo para trading algorítmico

Este repositorio contiene el código fuente para el Trabajo de Fin de Grado (TFG) centrado en la evaluación de diferentes familias de features financieras utilizando algoritmos de Deep Reinforcement Learning (DRL) para el trading algorítmico.

## Estructura del Proyecto

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
│   ├── equity_curves.py                    # Gráficos de curvas de capital (Equity Curves)
│   ├── combinatorial_ablation.py           # Experimento complementario: ablación combinatoria de features
│   └── generate_ablation_heatmap.py        # Heatmap de los escenarios de ablación estructurada
└── src/                                    # Código fuente, utilidades y entorno
    ├── data_utils.py                       # Carga y preprocesamiento de datos
    ├── download_ds.py                      # Descarga automatizada de activos financieros
    ├── experiment_utils.py                 # Funciones comunes extraídas de los algoritmos
    ├── features.py                         # Lógica matemática de los indicadores técnicos
    └── trading_env_improved.py             # Entorno de Gym customizado con soporte de comisiones
```

## Requisitos e Instalación

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

## Uso y Ejecución

1. **Descarga de Datos:** Se pueden descargar los datos ejecutando:

   ```bash
   python src/download_ds.py
   ```

   Al ejecutarlo se creará la carpeta `data/raw/` con los datos de los activos financieros.
   Los datos corresponden a los siguientes activos financieros:
   - Gold
   - Silver
   - Nvidia
   - Apple
   - Google
   - Inditex
   - Bitcoin
   - Ethereum
   - TetherUSDT
   - S&P_500_Vanguard

2. **Entrenamiento y Experimentación:** Se pueden ejecutar los algoritmos desde la raíz del proyecto o desde dentro de la carpeta `experiments/`:

   ```bash
   # Test usado para verificar el pipeline
   python experiments/smoke_test.py

   # Si se quiere ejecutar más de un algoritmo hacerlo de forma simultanea en distintos terminales

   # Ejecución del algoritmo DQN
   python experiments/DQN.py

   # Ejecución del algoritmo PPO
   python experiments/PPO.py

   # Ejecución del algoritmo A2C
   python experiments/A2C.py
   ```

Nota: Cada script leerá automáticamente las configuraciones desde `config/`. Por otra parte se creará una carpeta de `results/` al ejecutar cualquier algoritmo.

3. **Análisis de Resultados:** Una vez ejecutados los modelos, se pueden generar las métricas y comparativas:

   ```bash
   python experiments/analyze_results.py
   ```

   Al ejecutarlo el análisis de los resultados se guardarán en la carpeta `results/`.

4. **Curvas de Capital:** Para visualizar el rendimiento:

   ```bash
   python experiments/equity_curves.py --asset Gold --timesteps 5000 --seed 42
   ```

   Al ser ejecutado se guardará un gráfico en la carpeta `results/`.

## Arquitectura de Features y Ablación

El estudio evalúa el impacto de la información agrupada por familias:

- **Trend (Tendencia):** SMA, EMA, MACD
- **Momentum:** RSI, Stochastic Oscillator
- **Volatility (Volatilidad):** Bollinger Bands, ATR, Realized Volatility
- **Statistical (Estadística Avanzada):** Variables rezagadas, Diferencias, Descomposición Temporal

Se aplican estudios de **ablación** eliminando selectivamente una o varias de estas familias para comprobar su peso real en la toma de decisiones del agente DRL frente a estrategias _Baseline_.

## Experimento Complementario: Ablación Combinatoria

`experiments/combinatorial_ablation.py` es un análisis **complementario e independiente** del benchmark principal. No modifica ni sustituye los experimentos de `DQN.py`, `PPO.py` ni `A2C.py`.

**¿Qué hace?**
En lugar de evaluar familias de features individualmente (como hace el benchmark principal), genera **combinaciones** de familias (pares, tríos, etc.) y evalúa cuáles composiciones multi-familia obtienen mejores resultados. El objetivo es explorar si combinar varias familias supera a una sola de forma aislada.

**¿Por qué Gold por defecto?**
El script acepta cualquier asset mediante `--asset`.

**Cómo ejecutarlo:**

```bash

# Ejecución rápida — familias individuales, 1 semilla, DQN
# Top-k sirve para obtener las k mejores combinaciones. Si no se pone, se obtendrán todas
python experiments/combinatorial_ablation.py --asset Gold --max-combination-size 1 --timesteps 5000 --seeds 42 --algorithms DQN --top-k 10

# Análisis de pares y tríos — 3 algoritmos, 3 semillas (tarda más)
python experiments/combinatorial_ablation.py --asset Gold --max-combination-size 3 --timesteps 50000 --seeds 42 123 456 --algorithms DQN PPO A2C --top-k 10

# Mismo experimento sobre Bitcoin
python experiments/combinatorial_ablation.py --asset Bitcoin --max-combination-size 2 --timesteps 5000 --seeds 42 --algorithms DQN PPO A2C --top-k 10
```

**Assets disponibles:** Gold, Silver, Nvidia, Apple, Google, Inditex, Bitcoin, Ethereum, TetherUSDT, S&P_500_Vanguard

**Dónde se guardan los resultados:**

Todos los resultados se guardarán en la carpeta `results/`, en un directorio específico para la ablación combinatoria creado al ejecutar el script.

**Cómo interpretar los resultados:**

```bash
# Heatmap de todos los assets (promediado)
python experiments/generate_ablation_heatmap.py

# Solo para un asset
python experiments/generate_ablation_heatmap.py --asset Gold

# Con otra métrica
python experiments/generate_ablation_heatmap.py --asset Gold --metric total_return_mean
```
