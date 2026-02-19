"""Вычисление периодов и фазового сдвига из пересечений нуля.

Алгоритм фазового сдвига:
1. Период[i] = crossing[i+2] - crossing[i]  (каждое 2-е пересечение)
2. Делим периоды на «до помехи» (первая половина) и «после» (вторая).
3. T_avg = среднее по периодам первой половины.
4. N_total = полное число периодов (обе половины).
5. sum_all = сумма всех периодов.
6. delta_phase = (sum_all - T_avg * N_total) / T_avg

Определение фазы начала помехи:
- Находим два пересечения нуля, между которыми лежит CENTER_INDEX.
- Интерполируем фазу внутри полупериода и переводим в 0–2π.
"""

import numpy as np

import config
from zero_crossing import crossing_directions


def compute_periods(crossings: np.ndarray) -> np.ndarray:
    """Из массива пересечений нуля вычисляет массив периодов.

    Период[i] = crossing[i+2] - crossing[i].
    Длина результата: len(crossings) - 2.
    """
    if len(crossings) < 3:
        raise ValueError(f"Недостаточно пересечений нуля: {len(crossings)}, нужно >= 3")
    return crossings[2:] - crossings[:-2]


def split_periods_at_center(crossings: np.ndarray,
                            periods: np.ndarray,
                            center: float = config.CENTER_INDEX
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет периоды на «до помехи» и «после помехи».

    Период[i] покрывает интервал от crossing[i] до crossing[i+2].
    Если помеха попала внутрь этого интервала, период «заражён» и не должен
    попадать в «чистую» первую половину для усреднения.

    Первая половина (clean before): crossing[i+2] < center
        — весь период гарантированно до помехи.
    Вторая половина: crossing[i] >= center
        — весь период гарантированно после начала помехи.

    Периоды, перекрывающие center, не попадают ни в одну из групп.
    """
    period_starts = crossings[:-2]   # crossing[i]
    period_ends = crossings[2:]      # crossing[i+2]

    mask_before = period_ends < center
    mask_after = period_starts >= center
    return periods[mask_before], periods[mask_after]


def compute_phase_shift(crossings: np.ndarray,
                        center: float = config.CENTER_INDEX) -> float:
    """Вычисляет фазовый сдвиг, вносимый помехой.

    Возвращает:
        delta_phase — разность фаз в единицах периода.
        Положительное значение = фаза убежала вперёд.
    """
    periods = compute_periods(crossings)
    before, after = split_periods_at_center(crossings, periods, center)

    if len(before) == 0:
        raise ValueError("Нет периодов до помехи — невозможно вычислить среднее")

    T_avg = before.mean()
    N_total = len(periods)
    sum_all = periods.sum()

    delta_phase = (sum_all - T_avg * N_total) / T_avg
    return delta_phase


def compute_perturbation_phase(signal: np.ndarray,
                               crossings: np.ndarray,
                               center: float = config.CENTER_INDEX
                               ) -> float:
    """Определяет фазу колебания в момент начала помехи (0–2π).

    1. Находит два соседних пересечения нуля, между которыми лежит center.
    2. Линейно интерполирует долю полупериода.
    3. Учитывает направление (восходящее/нисходящее) для перевода в 0–2π.
    """
    # Находим k: crossings[k] <= center < crossings[k+1]
    idx = np.searchsorted(crossings, center, side="right") - 1
    idx = max(0, min(idx, len(crossings) - 2))

    c_left = crossings[idx]
    c_right = crossings[idx + 1]

    # Доля внутри полупериода (0..1)
    frac = (center - c_left) / (c_right - c_left)

    # Направление: +1 = восходящее (минус→плюс), -1 = нисходящее
    dirs = crossing_directions(signal, crossings)
    direction = dirs[idx]

    if direction == +1:
        # Восходящее пересечение → фаза идёт от 0 до π
        phase = frac * np.pi
    else:
        # Нисходящее пересечение → фаза идёт от π до 2π
        phase = np.pi + frac * np.pi

    return phase


# --- Отладка: запуск напрямую ---
if __name__ == "__main__":
    from loader import load_signal
    from zero_crossing import find_zero_crossings

    # Файл без помехи — сдвиг фазы должен быть ~0
    sig0 = load_signal(0)
    cr0 = find_zero_crossings(sig0)
    periods0 = compute_periods(cr0)
    shift0 = compute_phase_shift(cr0)
    phase0 = compute_perturbation_phase(sig0, cr0)

    print("=== Файл 0.csv (без помехи) ===")
    print(f"  Периодов: {len(periods0)}")
    print(f"  Средний период: {periods0.mean():.4f} отсчётов")
    print(f"  Сдвиг фазы: {shift0:.6f} (должен быть ~0)")
    print(f"  Фаза в центре: {phase0:.4f} рад ({np.degrees(phase0):.2f} град)")

    # Файл с помехой
    sig1 = load_signal(1)
    cr1 = find_zero_crossings(sig1)
    periods1 = compute_periods(cr1)
    shift1 = compute_phase_shift(cr1)
    phase1 = compute_perturbation_phase(sig1, cr1)

    print("\n=== Файл 1.csv (с помехой) ===")
    print(f"  Периодов: {len(periods1)}")
    print(f"  Средний период: {periods1.mean():.4f} отсчётов")
    print(f"  Сдвиг фазы: {shift1:.6f}")
    print(f"  Фаза помехи: {phase1:.4f} рад ({np.degrees(phase1):.2f} град)")
