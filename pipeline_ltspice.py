"""Пайплайн обработки LTSpice-данных: аналогичен экспериментальному.

ВАЖНО: LTSpice использует адаптивный шаг симуляции (неравномерную сетку),
поэтому нельзя работать в индексах — нужно пересчитывать пересечения нуля
из дробных индексов в моменты времени (секунды).

Для каждого шага .step param:
1. Загружаем (time, signal) из .raw
2. Обрезаем прогрев: берём данные от T_warmup до T_end
3. Находим пересечения нуля (в индексах), конвертируем в секунды
4. Считаем периоды, фазу помехи, сдвиг фазы — всё в секундах
"""

import numpy as np
from dataclasses import dataclass

import config
from loader_ltspice import load_raw, get_step_count, list_traces
from zero_crossing import find_zero_crossings, crossing_directions


@dataclass
class LTSpiceResult:
    """Результат обработки одного шага LTSpice-симуляции."""
    step: int
    n_crossings: int
    n_periods: int
    mean_period_before: float   # средний период до помехи (секунды)
    phase_shift: float          # сдвиг фазы (в периодах)
    perturbation_phase: float   # фаза начала помехи (0–2π рад)
    tdelay: float               # задержка помехи из .step (секунды)


@dataclass
class LTSpiceDiagnostics:
    """Все промежуточные данные одного шага — для диагностических графиков."""
    step: int
    tdelay: float
    t_perturbation: float       # LTSPICE_TBASE + tdelay (секунды)
    # Сырые данные (до обрезки)
    time_raw: np.ndarray
    signal_raw: np.ndarray
    # Обрезанные данные
    time_trimmed: np.ndarray
    signal_trimmed: np.ndarray
    # Пересечения нуля
    crossings_idx: np.ndarray       # дробные индексы в обрезанном массиве
    crossings_time: np.ndarray      # моменты времени (секунды)
    crossings_dirs: np.ndarray      # направления: +1 (вверх) / -1 (вниз)
    # Периоды
    periods: np.ndarray             # длительности периодов (секунды)
    periods_start_time: np.ndarray  # crossings_time[:-2]
    mask_before: np.ndarray         # bool-маска: «чисто до помехи»
    mask_after: np.ndarray          # bool-маска: «чисто после помехи»
    # Итоговые величины
    T_avg: float                    # средний период до помехи (секунды)
    phase_shift: float              # сдвиг фазы (в периодах)
    perturbation_phase: float       # фаза помехи (рад, 0–2π)


def _trim_to_useful(time: np.ndarray, signal: np.ndarray,
                    t_start: float = config.LTSPICE_WARMUP_TIME,
                    t_end: float = config.LTSPICE_TEND
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Обрезает сигнал до полезного окна [t_start, t_end].

    Возвращает обрезанные (time, signal).
    """
    mask = (time >= t_start) & (time <= t_end)
    return time[mask], signal[mask]


def _crossings_to_time(crossings_idx: np.ndarray,
                       time: np.ndarray) -> np.ndarray:
    """Конвертирует дробные индексы пересечений нуля в моменты времени.

    Для дробного индекса c = int_part + frac:
        t(c) = time[int_part] + frac * (time[int_part+1] - time[int_part])

    Это критически важно для LTSpice, где шаг по времени неравномерный.
    """
    int_parts = np.floor(crossings_idx).astype(int)
    fracs = crossings_idx - int_parts

    # Обеспечиваем безопасность на краях
    int_parts = np.clip(int_parts, 0, len(time) - 2)

    t_left = time[int_parts]
    t_right = time[int_parts + 1]
    return t_left + fracs * (t_right - t_left)


def _compute_periods_time(crossings_time: np.ndarray) -> np.ndarray:
    """Периоды из пересечений нуля (в секундах).

    period[i] = crossings_time[i+2] - crossings_time[i]
    """
    if len(crossings_time) < 3:
        raise ValueError(f"Недостаточно пересечений нуля: {len(crossings_time)}")
    return crossings_time[2:] - crossings_time[:-2]


def _split_periods_time(crossings_time: np.ndarray,
                        periods_time: np.ndarray,
                        t_perturbation: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Разделяет периоды на «до» и «после» помехи (работа в секундах).

    Период[i] покрывает интервал от crossing[i] до crossing[i+2].
    - «Чисто до»: crossing[i+2] < t_perturbation
    - «Чисто после»: crossing[i] >= t_perturbation
    """
    period_starts = crossings_time[:-2]
    period_ends = crossings_time[2:]

    mask_before = period_ends < t_perturbation
    mask_after = period_starts >= t_perturbation
    return periods_time[mask_before], periods_time[mask_after]


def _compute_phase_shift_time(crossings_time: np.ndarray,
                              t_perturbation: float) -> float:
    """Вычисляет фазовый сдвиг помехи (в периодах), работая в секундах.

    Алгоритм аналогичен period_analysis.compute_phase_shift,
    но все величины — в секундах, не в индексах.
    """
    periods = _compute_periods_time(crossings_time)
    before, after = _split_periods_time(crossings_time, periods, t_perturbation)

    if len(before) == 0:
        raise ValueError("Нет периодов до помехи — невозможно вычислить среднее")

    T_avg = before.mean()
    N_total = len(periods)
    sum_all = periods.sum()

    delta_phase = (sum_all - T_avg * N_total) / T_avg
    return delta_phase


def _compute_perturbation_phase_time(signal: np.ndarray,
                                     crossings_idx: np.ndarray,
                                     crossings_time: np.ndarray,
                                     t_perturbation: float) -> float:
    """Определяет фазу колебания в момент начала помехи (0–2π).

    Находит два соседних пересечения (по времени), между которыми
    лежит t_perturbation, линейно интерполирует.
    """
    idx = np.searchsorted(crossings_time, t_perturbation, side="right") - 1
    idx = max(0, min(idx, len(crossings_time) - 2))

    t_left = crossings_time[idx]
    t_right = crossings_time[idx + 1]

    frac = (t_perturbation - t_left) / (t_right - t_left)

    # Направление пересечения определяем по сигналу (индексы)
    dirs = crossing_directions(signal, crossings_idx)
    direction = dirs[idx]

    if direction == +1:
        phase = frac * np.pi
    else:
        phase = np.pi + frac * np.pi

    return phase


def process_single_step(raw_path: str,
                        trace_name: str | tuple,
                        step: int,
                        tdelay: float = 0.0) -> LTSpiceResult:
    """Полная обработка одного шага .step param."""
    time, signal = load_raw(raw_path, trace_name=trace_name, step=step)

    # Помеха в этом шаге — в момент Tbase + Tdelay
    t_perturbation = config.LTSPICE_TBASE + tdelay

    # Обрезаем прогрев
    time_tr, signal_tr = _trim_to_useful(time, signal)

    # Пересечения нуля в индексах обрезанного массива
    crossings_idx = find_zero_crossings(signal_tr)

    # КЛЮЧЕВОЕ ОТЛИЧИЕ: конвертируем индексы в моменты времени (секунды)
    crossings_time = _crossings_to_time(crossings_idx, time_tr)

    # Периоды в секундах
    periods_time = _compute_periods_time(crossings_time)

    # Средний период до помехи (секунды)
    before, _ = _split_periods_time(crossings_time, periods_time, t_perturbation)
    mean_T = before.mean() if len(before) > 0 else np.nan

    # Сдвиг фазы (в периодах) — всё в секундах
    phase_shift = _compute_phase_shift_time(crossings_time, t_perturbation)

    # Фаза попадания помехи — используем оба представления
    pert_phase = _compute_perturbation_phase_time(
        signal_tr, crossings_idx, crossings_time, t_perturbation)

    return LTSpiceResult(
        step=step,
        n_crossings=len(crossings_idx),
        n_periods=len(periods_time),
        mean_period_before=mean_T,
        phase_shift=phase_shift,
        perturbation_phase=pert_phase,
        tdelay=tdelay,
    )


def process_single_step_diagnostics(raw_path: str,
                                    trace_name: str | tuple,
                                    step: int,
                                    tdelay: float = 0.0) -> LTSpiceDiagnostics:
    """Полная обработка одного шага с сохранением ВСЕХ промежуточных данных.

    Используется для построения диагностических графиков.
    Алгоритм идентичен process_single_step(), но ничего не выбрасывается.
    """
    time_raw, signal_raw = load_raw(raw_path, trace_name=trace_name, step=step)

    t_perturbation = config.LTSPICE_TBASE + tdelay

    time_tr, signal_tr = _trim_to_useful(time_raw, signal_raw)

    crossings_idx = find_zero_crossings(signal_tr)
    crossings_time = _crossings_to_time(crossings_idx, time_tr)
    dirs = crossing_directions(signal_tr, crossings_idx)

    periods = _compute_periods_time(crossings_time)
    periods_start_time = crossings_time[:-2]

    # Маски классификации
    period_ends = crossings_time[2:]
    mask_before = period_ends < t_perturbation
    mask_after = periods_start_time >= t_perturbation

    before = periods[mask_before]
    T_avg = before.mean() if len(before) > 0 else np.nan

    phase_shift = _compute_phase_shift_time(crossings_time, t_perturbation)
    pert_phase = _compute_perturbation_phase_time(
        signal_tr, crossings_idx, crossings_time, t_perturbation)

    return LTSpiceDiagnostics(
        step=step,
        tdelay=tdelay,
        t_perturbation=t_perturbation,
        time_raw=time_raw,
        signal_raw=signal_raw,
        time_trimmed=time_tr,
        signal_trimmed=signal_tr,
        crossings_idx=crossings_idx,
        crossings_time=crossings_time,
        crossings_dirs=dirs,
        periods=periods,
        periods_start_time=periods_start_time,
        mask_before=mask_before,
        mask_after=mask_after,
        T_avg=T_avg,
        phase_shift=phase_shift,
        perturbation_phase=pert_phase,
    )


def _read_tdelay_from_raw(raw_path: str, n_steps: int) -> np.ndarray:
    """Вычисляет Tdelay для каждого шага, анализируя сигнал помехи.

    Загружает шаг 0 и последний шаг, находит момент начала помехи
    (резкий скачок I(V2) или напряжения), и по ним определяет шаг.

    Если не получается — возвращает линейную шкалу по параметрам из .asc.
    """
    from loader_ltspice import load_raw as _load

    # Пробуем определить Tdelay по реальным данным:
    # На шаге 0 помеха в Tbase, на шаге (n-1) помеха в Tbase + (n-1)*step_dt
    # Разница даёт полный диапазон.
    # Из .asc: .step param Tdelay START END STEP
    # Лучше прочитать .raw напрямую через PyLTSpice
    try:
        from PyLTSpice import RawRead
        raw = RawRead(str(raw_path))
        # PyLTSpice хранит параметры step в get_steps()
        steps = list(raw.get_steps())
        if len(steps) > 1:
            # Получаем I(V2) — ток помехи — для первого и последнего шага
            # и находим момент его начала
            tdelays = np.zeros(len(steps))
            for s in steps:
                time_s = raw.get_wave('time', step=s).real.astype(np.float64)
                # Ищем ток V2 (помеха) — он резко вырастает в момент помехи
                try:
                    iv2 = raw.get_wave('I(V2)', step=s).real.astype(np.float64)
                    # Момент первого значимого тока (> порога)
                    threshold = np.abs(iv2).max() * 0.5
                    if threshold > 0:
                        idx_start = np.argmax(np.abs(iv2) > threshold)
                        tdelays[s] = time_s[idx_start] - config.LTSPICE_TBASE
                except Exception:
                    # Если I(V2) нет, используем линейную аппроксимацию
                    pass
            # Проверяем, получилось ли
            if tdelays[-1] > tdelays[0]:
                return tdelays
    except Exception:
        pass

    # Фоллбек: линейная шкала
    # Пробуем определить из числа шагов и параметров
    return None


def process_all_steps(raw_path: str,
                      trace_name: str | tuple,
                      tdelay_start: float = 0.0,
                      tdelay_end: float | None = None,
                      ) -> list[LTSpiceResult]:
    """Обрабатывает все шаги .step param из .raw файла.

    Параметры:
        raw_path: путь к .raw файлу
        trace_name: имя трассы
        tdelay_start: начальное значение Tdelay (из .step directive)
        tdelay_end: конечное значение Tdelay (из .step directive)
    """
    n_steps = get_step_count(raw_path)

    # Пробуем определить Tdelay автоматически
    tdelays = _read_tdelay_from_raw(raw_path, n_steps)

    if tdelays is None:
        # Линейная шкала по параметрам
        if tdelay_end is None:
            tdelay_end = tdelay_start
        tdelays = np.linspace(tdelay_start, tdelay_end, n_steps)

    results = []
    for s in range(n_steps):
        tdelay = float(tdelays[s])
        try:
            result = process_single_step(raw_path, trace_name, s, tdelay)
            results.append(result)
        except Exception as e:
            print(f"[!] Ошибка на шаге {s}: {e}")
    return results


def results_to_arrays(results: list[LTSpiceResult]) -> dict[str, np.ndarray]:
    """Преобразует список результатов в словарь numpy-массивов."""
    return {
        "steps": np.array([r.step for r in results]),
        "tdelay": np.array([r.tdelay for r in results]),
        "perturbation_phase": np.array([r.perturbation_phase for r in results]),
        "phase_shift": np.array([r.phase_shift for r in results]),
        "mean_period_before": np.array([r.mean_period_before for r in results]),
    }


# --- Отладка ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python pipeline_ltspice.py <путь_к_raw> [схема]")
        print("  схема: 'bjt' (по умолчанию) или 'mosfet'")
        sys.exit(1)

    raw_path = sys.argv[1]
    scheme = sys.argv[2] if len(sys.argv) > 2 else "bjt"
    trace = config.LTSPICE_TRACES.get(scheme)

    print(f"Файл: {raw_path}")
    print(f"Трассы: {list_traces(raw_path)}")
    print(f"Используемая трасса: {trace}")

    n_steps = get_step_count(raw_path)
    print(f"Шагов: {n_steps}")

    if n_steps == 1:
        print("\nОдин шаг — обрабатываю как одиночный прогон...")
        r = process_single_step(raw_path, trace, 0, tdelay=0.0)
        print(f"  Пересечений: {r.n_crossings}, периодов: {r.n_periods}")
        print(f"  Средний период: {r.mean_period_before*1e6:.4f} мкс")
        print(f"  Фаза помехи: {np.degrees(r.perturbation_phase):.1f}°")
        print(f"  Сдвиг фазы: {r.phase_shift:.6f} периодов = {r.phase_shift*360:.1f}°")
    else:
        print(f"\nОбрабатываю {n_steps} шагов...")
        results = process_all_steps(raw_path, trace)
        data = results_to_arrays(results)

        phases = np.degrees(data["perturbation_phase"])
        shifts = data["phase_shift"] * 360

        print(f"  Успешно: {len(results)} из {n_steps}")
        print(f"  Средний период: {data['mean_period_before'].mean()*1e6:.2f} мкс")
        print(f"  Фаза помехи: {phases.min():.1f}° ... {phases.max():.1f}°")
        print(f"  Сдвиг фазы: {shifts.min():.1f}° ... {shifts.max():.1f}°")

        # Строим график
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        config.OUTPUT_DIR.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(phases, shifts, s=20, color="steelblue", alpha=0.8)
        ax.set_xlabel("Фаза начала помехи (градусы)")
        ax.set_ylabel("Сдвиг фазы (градусы)")
        ax.set_title(f"LTSpice ({scheme}): зависимость сдвига фазы от фазы помехи")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlim(0, 360)
        plt.tight_layout()
        out = config.OUTPUT_DIR / f"ltspice_{scheme}_phase_vs_shift.png"
        fig.savefig(out, dpi=150)
        print(f"  График: {out}")
        plt.close(fig)
