"""Оркестрация обработки: загрузка → нули → периоды → фаза.

Единственный модуль, который знает обо всех остальных.
Остальные модули независимы друг от друга (кроме config).
"""

import numpy as np
from dataclasses import dataclass

import config
from loader import load_signal
from zero_crossing import find_zero_crossings, crossing_directions
from period_analysis import (
    compute_periods,
    split_periods_at_center,
    compute_phase_shift,
    compute_perturbation_phase,
)


@dataclass
class FileResult:
    """Результат обработки одного файла."""
    file_number: int
    n_crossings: int
    n_periods: int
    mean_period_before: float   # средний период до помехи (отсчёты)
    phase_shift: float          # сдвиг фазы (в периодах)
    perturbation_phase: float   # фаза начала помехи (0–2π рад)


@dataclass
class ExperimentDiagnostics:
    """Все промежуточные данные одного файла — для диагностических графиков."""
    file_number: int
    signal: np.ndarray                # исходный сигнал (16384 отсчёта)
    crossings_idx: np.ndarray         # дробные индексы пересечений нуля
    crossings_dirs: np.ndarray        # +1 (вверх) / -1 (вниз)
    periods: np.ndarray               # period[i] = crossing[i+2] - crossing[i]
    periods_start: np.ndarray         # crossings_idx[:-2]
    mask_before: np.ndarray           # bool: period_end < CENTER_INDEX
    mask_after: np.ndarray            # bool: period_start >= CENTER_INDEX
    T_avg: float                      # средний период до помехи (отсчёты)
    phase_shift: float                # сдвиг фазы (в периодах)
    perturbation_phase: float         # фаза помехи (0–2π рад)


def process_single_file(file_number: int) -> FileResult:
    """Полная обработка одного CSV-файла."""
    signal = load_signal(file_number)
    crossings = find_zero_crossings(signal)
    periods = compute_periods(crossings)

    # Средний период первой половины
    period_starts = crossings[:-2]
    mask_before = period_starts < config.CENTER_INDEX
    periods_before = periods[mask_before]
    mean_T = periods_before.mean() if len(periods_before) > 0 else np.nan

    phase_shift = compute_phase_shift(crossings)
    pert_phase = compute_perturbation_phase(signal, crossings)

    return FileResult(
        file_number=file_number,
        n_crossings=len(crossings),
        n_periods=len(periods),
        mean_period_before=mean_T,
        phase_shift=phase_shift,
        perturbation_phase=pert_phase,
    )


def process_single_file_diagnostics(file_number: int) -> ExperimentDiagnostics:
    """Полная обработка одного CSV-файла с сохранением ВСЕХ промежуточных данных.

    Используется для построения диагностических графиков.
    """
    signal = load_signal(file_number)
    crossings = find_zero_crossings(signal)
    dirs = crossing_directions(signal, crossings)
    periods = compute_periods(crossings)

    period_starts = crossings[:-2]
    period_ends = crossings[2:]
    mask_before = period_ends < config.CENTER_INDEX
    mask_after = period_starts >= config.CENTER_INDEX

    before = periods[mask_before]
    T_avg = before.mean() if len(before) > 0 else np.nan

    phase_shift = compute_phase_shift(crossings)
    pert_phase = compute_perturbation_phase(signal, crossings)

    return ExperimentDiagnostics(
        file_number=file_number,
        signal=signal,
        crossings_idx=crossings,
        crossings_dirs=dirs,
        periods=periods,
        periods_start=period_starts,
        mask_before=mask_before,
        mask_after=mask_after,
        T_avg=T_avg,
        phase_shift=phase_shift,
        perturbation_phase=pert_phase,
    )


def process_all_files(file_numbers: list[int] | None = None) -> list[FileResult]:
    """Обрабатывает все файлы с помехой. Возвращает список результатов."""
    if file_numbers is None:
        file_numbers = config.SIGNAL_FILES

    results = []
    for num in file_numbers:
        try:
            result = process_single_file(num)
            results.append(result)
        except Exception as e:
            print(f"[!] Ошибка при обработке файла {num}.csv: {e}")
    return results


def results_to_arrays(results: list[FileResult]) -> dict[str, np.ndarray]:
    """Преобразует список результатов в словарь numpy-массивов для удобства."""
    return {
        "file_numbers": np.array([r.file_number for r in results]),
        "perturbation_phase": np.array([r.perturbation_phase for r in results]),
        "phase_shift": np.array([r.phase_shift for r in results]),
        "mean_period_before": np.array([r.mean_period_before for r in results]),
    }


# --- Отладка: запуск напрямую ---
if __name__ == "__main__":
    print("Обработка файла 0 (эталон)...")
    ref = process_single_file(0)
    print(f"  Пересечений: {ref.n_crossings}, периодов: {ref.n_periods}")
    print(f"  Средний период (до центра): {ref.mean_period_before:.4f}")
    print(f"  Сдвиг фазы: {ref.phase_shift:.6f}")

    print("\nОбработка файлов 1-5 (с помехой)...")
    results = process_all_files([1, 2, 3, 4, 5])
    for r in results:
        print(f"  Файл {r.file_number}: "
              f"фаза помехи = {np.degrees(r.perturbation_phase):.1f} град, "
              f"сдвиг = {r.phase_shift:.6f}")
