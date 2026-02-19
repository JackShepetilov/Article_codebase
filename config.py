from pathlib import Path

# --- Пути ---
PROJECT_DIR = Path(__file__).parent
CSV_DIR = PROJECT_DIR / "csv_files"
OUTPUT_DIR = PROJECT_DIR / "output"

# --- Параметры осциллографа ---
DT = 50e-9                  # шаг дискретизации, секунды (50 нс)
N_SAMPLES = 16384           # число отсчётов в одной развёртке
CENTER_INDEX = N_SAMPLES // 2  # центр развёртки = начало помехи (8192)

# --- Параметры поиска пересечений нуля ---
N_FIT_POINTS = 7            # число точек для МНК-аппроксимации прямой

# --- Файлы данных ---
REFERENCE_FILE = 0          # номер файла без помехи
SIGNAL_FILES = list(range(1, 101))  # номера файлов с помехой

# --- LTSpice ---
LTSPICE_DIR = PROJECT_DIR / "LTSpice"

# Трассы: напряжение на катушке (дифференциальное V(A) - V(B))
LTSPICE_TRACES = {
    "bjt": ("V(N002)", "V(N003)"),      # V(L2), Colpitsa_2C___experiment_v3
    "mosfet": ("V(N003)", "V(N004)"),   # V(L7), Colpits_IRF640_common_emitter_simple_scheme
}

# Параметры симуляции BJT phase scan
LTSPICE_WARMUP_TIME = 1.0e-3       # время прогрева для обрезки (1 мс)
LTSPICE_TBASE = 6.0e-3             # базовая задержка помехи (6 мс)
LTSPICE_TEND = 11.0e-3             # конец симуляции (11 мс)
# Полезные данные: от WARMUP_TIME до TEND, помеха в TBASE = ровно посередине
