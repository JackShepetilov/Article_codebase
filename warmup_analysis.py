"""Определение времени прогрева генератора Колпитца в LTSpice-симуляции.

Алгоритм:
1. Загружаем сигнал из .raw файла (симуляция без помехи).
2. Вычисляем огибающую амплитуды: для каждого полупериода находим
   максимум |signal|.
3. Определяем стационарную амплитуду как среднее последних 20% огибающей.
4. Время прогрева — момент, когда огибающая впервые достигает 95%
   от стационарной амплитуды (порог настраивается).

Запуск:
    python warmup_analysis.py <путь_к_raw> [trace_name]
"""

import numpy as np
from pathlib import Path

from loader_ltspice import load_raw, list_traces
from zero_crossing import find_zero_crossings


def compute_envelope(time: np.ndarray, signal: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Вычисляет огибающую амплитуды по полупериодам.

    Для каждого интервала между соседними пересечениями нуля
    находит max(|signal|) и время середины интервала.

    Возвращает:
        (env_time, env_amplitude) — массивы одинаковой длины.
    """
    crossings = find_zero_crossings(signal)

    if len(crossings) < 2:
        raise ValueError("Слишком мало пересечений нуля для вычисления огибающей")

    env_time = []
    env_amp = []

    for i in range(len(crossings) - 1):
        idx_start = int(np.floor(crossings[i]))
        idx_end = int(np.ceil(crossings[i + 1]))
        idx_start = max(0, idx_start)
        idx_end = min(len(signal), idx_end + 1)

        if idx_end <= idx_start:
            continue

        segment = signal[idx_start:idx_end]
        amp = np.max(np.abs(segment))

        # Время — середина полупериода
        mid_idx = (idx_start + idx_end) // 2
        if mid_idx < len(time):
            env_time.append(time[mid_idx])
            env_amp.append(amp)

    return np.array(env_time), np.array(env_amp)


def find_warmup_time(time: np.ndarray, signal: np.ndarray,
                     threshold: float = 0.95) -> dict:
    """Определяет время прогрева генератора.

    Параметры:
        time: массив времени
        signal: массив напряжения
        threshold: доля от стационарной амплитуды (по умолчанию 0.95)

    Возвращает dict:
        warmup_time: время прогрева в секундах
        warmup_index: индекс в исходном массиве, соответствующий warmup_time
        steady_amplitude: стационарная амплитуда
        envelope_time: массив времён огибающей
        envelope_amp: массив амплитуд огибающей
    """
    env_time, env_amp = compute_envelope(time, signal)

    # Стационарная амплитуда — среднее последних 20%
    tail_start = int(len(env_amp) * 0.8)
    steady_amp = env_amp[tail_start:].mean()

    # Порог
    target = threshold * steady_amp

    # Ищем первый момент, когда амплитуда достигла порога
    above_mask = env_amp >= target
    if not above_mask.any():
        warmup_t = env_time[-1]
    else:
        first_above = np.argmax(above_mask)
        warmup_t = env_time[first_above]

    # Индекс в исходном массиве time
    warmup_idx = np.searchsorted(time, warmup_t)

    return {
        "warmup_time": warmup_t,
        "warmup_index": int(warmup_idx),
        "steady_amplitude": steady_amp,
        "threshold": threshold,
        "envelope_time": env_time,
        "envelope_amp": env_amp,
    }


def plot_warmup(time, signal, warmup_result, save_path=None):
    """Строит график сигнала + огибающая + отметка прогрева."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env_t = warmup_result["envelope_time"]
    env_a = warmup_result["envelope_amp"]
    w_time = warmup_result["warmup_time"]
    steady = warmup_result["steady_amplitude"]
    thr = warmup_result["threshold"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Верхний график: сигнал
    ax1.plot(time * 1e3, signal, linewidth=0.3, color="steelblue")
    ax1.axvline(w_time * 1e3, color="red", linestyle="--",
                linewidth=1.5, label=f"Прогрев: {w_time*1e3:.3f} мс")
    ax1.set_ylabel("Напряжение, В")
    ax1.set_title("Сигнал генератора Колпитца (LTSpice)")
    ax1.legend()

    # Нижний график: огибающая
    ax2.plot(env_t * 1e3, env_a, ".-", markersize=2, color="steelblue",
             label="Огибающая")
    ax2.axhline(steady, color="green", linestyle=":", linewidth=1,
                label=f"Стационар: {steady:.4f} В")
    ax2.axhline(thr * steady, color="orange", linestyle=":", linewidth=1,
                label=f"Порог {thr*100:.0f}%: {thr*steady:.4f} В")
    ax2.axvline(w_time * 1e3, color="red", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Время, мс")
    ax2.set_ylabel("Амплитуда, В")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Сохранено: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# --- Отладка ---
if __name__ == "__main__":
    import sys
    import config

    if len(sys.argv) < 2:
        print("Использование: python warmup_analysis.py <путь_к_raw> [схема]")
        print("  схема: 'bjt' или 'mosfet' (для выбора дифф. трассы из config)")
        print(f"  Трассы в config: {config.LTSPICE_TRACES}")
        print("\nСначала прогони warmup_test.asc в LTSpice, затем:")
        print("  python warmup_analysis.py LTSpice/warmup_test.raw bjt")
        sys.exit(1)

    raw_path = sys.argv[1]
    scheme = sys.argv[2] if len(sys.argv) > 2 else "bjt"
    trace = config.LTSPICE_TRACES.get(scheme)

    print(f"Файл: {raw_path}")
    print(f"Трассы в файле: {list_traces(raw_path)}")
    print(f"Используемая трасса: {trace} (схема: {scheme})")

    time, signal = load_raw(raw_path, trace_name=trace)

    print(f"\nДанные: {len(time)} точек")
    print(f"  Время: {time[0]*1e3:.3f} ... {time[-1]*1e3:.3f} мс")
    print(f"  Сигнал: мин={signal.min():.4f}, макс={signal.max():.4f} В")

    result = find_warmup_time(time, signal)

    print(f"\n=== Результат ===")
    print(f"  Стационарная амплитуда: {result['steady_amplitude']:.4f} В")
    print(f"  Порог ({result['threshold']*100:.0f}%): "
          f"{result['threshold'] * result['steady_amplitude']:.4f} В")
    print(f"  Время прогрева: {result['warmup_time']*1e3:.3f} мс")
    print(f"  Индекс прогрева: {result['warmup_index']}")

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    plot_warmup(time, signal, result,
                save_path=config.OUTPUT_DIR / "warmup_analysis.png")
