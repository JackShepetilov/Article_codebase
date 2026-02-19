"""Визуализация для отладки и для статьи.

Каждая функция — самостоятельный график. Можно вызывать по отдельности.
"""

import numpy as np
import matplotlib.pyplot as plt

import config
from loader import load_signal
from zero_crossing import find_zero_crossings, crossing_directions
from period_analysis import compute_periods


def plot_signal_with_crossings(file_number: int, save: bool = False):
    """Сигнал + отмеченные пересечения нуля + вертикальная линия центра.

    Главный отладочный график: видно, правильно ли найдены нули.
    """
    signal = load_signal(file_number)
    crossings = find_zero_crossings(signal)
    dirs = crossing_directions(signal, crossings)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(signal, linewidth=0.5, color="steelblue", label="Сигнал")

    # Пересечения нуля: вверх — зелёный, вниз — красный
    for c, d in zip(crossings, dirs):
        color = "green" if d == +1 else "red"
        ax.axvline(c, color=color, alpha=0.3, linewidth=0.5)

    ax.axvline(config.CENTER_INDEX, color="black", linestyle="--",
               linewidth=1.5, label=f"Центр ({config.CENTER_INDEX})")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Отсчёт")
    ax.set_ylabel("Напряжение, В")
    ax.set_title(f"Файл {file_number}.csv — {len(crossings)} пересечений нуля")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(config.OUTPUT_DIR / f"signal_{file_number}.png", dpi=150)
    plt.show()


def plot_signal_zoom_center(file_number: int, window: int = 200,
                            save: bool = False):
    """Увеличенный вид вокруг центра развёртки (начало помехи).

    Позволяет визуально проверить определение фазы помехи.
    """
    signal = load_signal(file_number)
    crossings = find_zero_crossings(signal)
    dirs = crossing_directions(signal, crossings)

    c = config.CENTER_INDEX
    sl = slice(max(0, c - window), min(len(signal), c + window))
    x = np.arange(sl.start, sl.stop)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, signal[sl], linewidth=1, color="steelblue", label="Сигнал")

    # Пересечения в видимом окне
    for cr, d in zip(crossings, dirs):
        if sl.start <= cr <= sl.stop:
            color = "green" if d == +1 else "red"
            marker = "^" if d == +1 else "v"
            ax.plot(cr, 0, marker, color=color, markersize=8)
            ax.axvline(cr, color=color, alpha=0.3, linewidth=0.8)

    ax.axvline(c, color="black", linestyle="--", linewidth=1.5,
               label="Начало помехи")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Отсчёт")
    ax.set_ylabel("Напряжение, В")
    ax.set_title(f"Файл {file_number}.csv — окрестность помехи")
    ax.legend()
    plt.tight_layout()

    if save:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(config.OUTPUT_DIR / f"zoom_center_{file_number}.png", dpi=150)
    plt.show()


def plot_periods(file_number: int, save: bool = False):
    """График периодов по номеру. Видно, как помеха меняет период."""
    signal = load_signal(file_number)
    crossings = find_zero_crossings(signal)
    periods = compute_periods(crossings)
    period_starts = crossings[:-2]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["steelblue" if s < config.CENTER_INDEX else "coral"
              for s in period_starts]
    ax.scatter(period_starts, periods, c=colors, s=3)
    ax.axvline(config.CENTER_INDEX, color="black", linestyle="--",
               linewidth=1.5, label="Начало помехи")

    # Среднее по первой половине
    mask = period_starts < config.CENTER_INDEX
    if mask.any():
        mean_T = periods[mask].mean()
        ax.axhline(mean_T, color="steelblue", linestyle=":",
                   label=f"Среднее до помехи: {mean_T:.2f}")

    ax.set_xlabel("Позиция начала периода (отсчёт)")
    ax.set_ylabel("Период (отсчёты)")
    ax.set_title(f"Файл {file_number}.csv — периоды")
    ax.legend()
    plt.tight_layout()

    if save:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(config.OUTPUT_DIR / f"periods_{file_number}.png", dpi=150)
    plt.show()


def plot_phase_vs_shift(results_dict: dict[str, np.ndarray],
                        save: bool = False):
    """Главный результат: фаза помехи (0–2π) vs сдвиг фазы.

    results_dict — из pipeline.results_to_arrays().
    """
    phases = results_dict["perturbation_phase"]
    shifts = results_dict["phase_shift"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(np.degrees(phases), shifts, s=15, color="steelblue", alpha=0.7)
    ax.set_xlabel("Фаза начала помехи (градусы)")
    ax.set_ylabel("Сдвиг фазы (в периодах)")
    ax.set_title("Зависимость сдвига фазы от фазы попадания помехи")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlim(0, 360)
    plt.tight_layout()

    if save:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fig.savefig(config.OUTPUT_DIR / "phase_vs_shift.png", dpi=150)
    plt.show()


# --- Отладка: запуск напрямую ---
if __name__ == "__main__":
    print("Построение отладочных графиков для файла 0 (эталон)...")
    plot_signal_with_crossings(0)
    plot_periods(0)

    print("Построение отладочных графиков для файла 1 (с помехой)...")
    plot_signal_with_crossings(1)
    plot_signal_zoom_center(1)
    plot_periods(1)
