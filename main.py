"""Точка входа: обрабатывает все файлы и строит итоговый график."""

import numpy as np

import config
from pipeline import process_single_file, process_all_files, results_to_arrays
from plotting import (
    plot_signal_with_crossings,
    plot_signal_zoom_center,
    plot_periods,
    plot_phase_vs_shift,
)


def main():
    # 1. Эталон (без помехи) — проверяем, что сдвиг ~0
    print("=" * 60)
    print("Обработка эталона (файл 0, без помехи)")
    print("=" * 60)
    ref = process_single_file(config.REFERENCE_FILE)
    print(f"  Пересечений нуля: {ref.n_crossings}")
    print(f"  Периодов: {ref.n_periods}")
    print(f"  Средний период (до центра): {ref.mean_period_before:.4f} отсчётов")
    print(f"  Сдвиг фазы: {ref.phase_shift:.6f} (ожидаем ~0)")

    # 2. Обработка всех файлов с помехой
    print(f"\n{'=' * 60}")
    print(f"Обработка файлов 1–100 (с помехой)")
    print(f"{'=' * 60}")
    results = process_all_files()
    print(f"  Успешно обработано: {len(results)} из {len(config.SIGNAL_FILES)}")

    # 3. Сводка
    data = results_to_arrays(results)
    phases_deg = np.degrees(data["perturbation_phase"])
    shifts = data["phase_shift"]

    print(f"\n  Фаза помехи: {phases_deg.min():.1f}° ... {phases_deg.max():.1f}°")
    print(f"  Сдвиг фазы:  мин={shifts.min():.6f}, макс={shifts.max():.6f}")
    print(f"  Сдвиг фазы:  среднее={shifts.mean():.6f}, std={shifts.std():.6f}")

    # 4. Итоговый график
    plot_phase_vs_shift(data)


if __name__ == "__main__":
    main()
