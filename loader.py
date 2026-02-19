import numpy as np
from pathlib import Path

import config


def load_signal(file_number: int) -> np.ndarray:
    """Загружает CSV осциллографа и возвращает одномерный массив напряжения.

    Файл содержит 3 столбца через запятую; используем только первый.
    """
    filepath = config.CSV_DIR / f"{file_number}.csv"
    data = np.loadtxt(filepath, delimiter=",", usecols=(0,))
    return data


# --- Отладка: запуск напрямую ---
if __name__ == "__main__":
    sig = load_signal(0)
    print(f"Загружен файл 0.csv: {len(sig)} отсчётов")
    print(f"  мин = {sig.min():.4f}, макс = {sig.max():.4f}")
    print(f"  первые 5: {sig[:5]}")
    print(f"  последние 5: {sig[-5:]}")
