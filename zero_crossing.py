"""Поиск пересечений нуля сигнала методом МНК-аппроксимации прямой.

Алгоритм:
1. Находим все пары соседних отсчётов, между которыми сигнал меняет знак
   (или проходит через ноль).
2. Для каждой такой пары берём окно из N_FIT_POINTS точек, центрированное
   на переходе, и строим прямую y = a*x + b по МНК.
3. Корень прямой x0 = -b/a даёт дробный индекс пересечения нуля.
"""

import numpy as np

import config


def _find_raw_crossing_indices(signal: np.ndarray) -> list[int]:
    """Находит индексы i, где происходит смена знака между signal[i] и signal[i+1].

    Учитываем случаи:
    - signal[i] и signal[i+1] имеют разный знак
    - signal[i] == 0 (точное попадание на ноль)
    """
    crossings = []
    n = len(signal)
    i = 0
    while i < n - 1:
        if signal[i] == 0:
            # Точное попадание: ищём конец цепочки нулей
            j = i
            while j < n and signal[j] == 0:
                j += 1
            # Если до и после цепочки нулей знаки разные — это пересечение
            # Если дошли до края — не считаем
            if i > 0 and j < n:
                if signal[i - 1] * signal[j] < 0:
                    # Центр цепочки нулей
                    crossings.append((i + j - 1) // 2)
            i = j
        elif signal[i] * signal[i + 1] < 0:
            # Классическая смена знака
            crossings.append(i)
            i += 1
        else:
            i += 1
    return crossings


def _refine_crossing_lsq(signal: np.ndarray, raw_idx: int,
                          n_fit: int = config.N_FIT_POINTS) -> float:
    """Уточняет позицию пересечения нуля по МНК-прямой.

    Берёт n_fit точек с центром около raw_idx (3-я и 4-я точки окна
    охватывают переход). Строит прямую y = a*x + b, возвращает x0 = -b/a.

    Параметры:
        signal: массив напряжения
        raw_idx: индекс i, где signal[i] и signal[i+1] — по разные стороны нуля
        n_fit: число точек для аппроксимации (по умолчанию 7)

    Возвращает:
        Дробный индекс пересечения нуля (в единицах отсчётов).
    """
    half = n_fit // 2  # для n_fit=7: half=3
    # Центрируем окно так, чтобы raw_idx попал на позицию half-1 (3-ю из 7)
    # Тогда точки [raw_idx-2, raw_idx-1, raw_idx, raw_idx+1, ..., raw_idx+4]
    # и переход raw_idx -> raw_idx+1 оказывается между 3-й и 4-й точками окна
    start = raw_idx - (half - 1)
    end = start + n_fit

    # Обрезка по краям массива
    if start < 0:
        start = 0
        end = n_fit
    if end > len(signal):
        end = len(signal)
        start = end - n_fit

    indices = np.arange(start, end, dtype=np.float64)
    values = signal[start:end]

    # МНК: y = a*x + b
    # polyfit возвращает [a, b] для полинома степени 1
    a, b = np.polyfit(indices, values, 1)

    if a == 0:
        # Вырожденный случай: прямая горизонтальна
        return float(raw_idx) + 0.5

    x0 = -b / a
    return float(x0)


def find_zero_crossings(signal: np.ndarray,
                        n_fit: int = config.N_FIT_POINTS) -> np.ndarray:
    """Находит все пересечения нуля сигнала с уточнением по МНК.

    Возвращает:
        np.ndarray дробных индексов пересечений нуля (в отсчётах),
        отсортированный по возрастанию.
    """
    raw_indices = _find_raw_crossing_indices(signal)
    refined = [_refine_crossing_lsq(signal, idx, n_fit) for idx in raw_indices]
    return np.array(refined)


def crossing_directions(signal: np.ndarray,
                        crossings: np.ndarray) -> np.ndarray:
    """Определяет направление каждого пересечения нуля.

    Возвращает:
        np.ndarray из +1 (восходящее: минус→плюс) и -1 (нисходящее: плюс→минус).
    """
    directions = np.empty(len(crossings), dtype=np.int8)
    for i, c in enumerate(crossings):
        idx = int(np.floor(c))
        idx = max(0, min(idx, len(signal) - 2))
        if signal[idx + 1] > signal[idx]:
            directions[i] = +1
        else:
            directions[i] = -1
    return directions


# --- Отладка: запуск напрямую ---
if __name__ == "__main__":
    from loader import load_signal

    sig = load_signal(0)
    crossings = find_zero_crossings(sig)
    dirs = crossing_directions(sig, crossings)

    print(f"Файл 0.csv (без помехи): найдено {len(crossings)} пересечений нуля")
    print(f"  первые 5 позиций: {crossings[:5]}")
    print(f"  первые 5 направлений: {dirs[:5]}  (+1=вверх, -1=вниз)")

    # Проверка: расстояния между соседними пересечениями (полупериоды)
    half_periods = np.diff(crossings)
    print(f"  средний полупериод: {half_periods.mean():.2f} отсчётов")
    print(f"  std полупериода: {half_periods.std():.4f} отсчётов")
    print(f"  мин/макс полупериод: {half_periods.min():.2f} / {half_periods.max():.2f}")
