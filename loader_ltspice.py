"""Загрузка данных из LTSpice .raw файлов.

Использует PyLTSpice (spicelib) для чтения бинарных .raw файлов.
Поддерживает как одиночные симуляции, так и .step param прогоны.
Поддерживает дифференциальные трассы (V(A) - V(B)).
"""

import numpy as np
from pathlib import Path
from PyLTSpice import RawRead


def _read_signal(raw: RawRead, trace_spec, step: int) -> np.ndarray:
    """Читает сигнал по спецификации трассы.

    trace_spec может быть:
      - str: одиночная трасса, напр. 'V(N002)'
      - tuple[str, str]: дифференциальная, напр. ('V(N002)', 'V(N003)')
        → возвращает V(N002) - V(N003)
    """
    if isinstance(trace_spec, tuple):
        a = raw.get_wave(trace_spec[0], step=step)
        b = raw.get_wave(trace_spec[1], step=step)
        return (a - b).real.astype(np.float64)
    else:
        return raw.get_wave(trace_spec, step=step).real.astype(np.float64)


def _resolve_trace(raw: RawRead, trace_spec) -> str | tuple:
    """Если trace_spec=None, берёт первую трассу напряжения. Иначе возвращает как есть."""
    if trace_spec is not None:
        return trace_spec
    names = raw.get_trace_names()
    voltage_names = [n for n in names if n.upper().startswith('V(') and n != 'time']
    if not voltage_names:
        raise ValueError(f"Не найдено трасс напряжения. Доступные: {names}")
    return voltage_names[0]


def load_raw(raw_path: str | Path,
             trace_name: str | tuple | None = None,
             step: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Загружает один прогон из .raw файла.

    Параметры:
        raw_path: путь к .raw файлу
        trace_name: имя трассы ('V(N002)') или кортеж для дифференциальной
                    ('V(N002)', 'V(N003)'). None → первая трасса напряжения.
        step: номер шага для .step param прогонов

    Возвращает:
        (time, signal) — два numpy-массива.
    """
    raw = RawRead(str(raw_path))
    trace_spec = _resolve_trace(raw, trace_name)

    time = raw.get_wave('time', step=step).real.astype(np.float64)
    signal = _read_signal(raw, trace_spec, step)
    return time, signal


def get_step_count(raw_path: str | Path) -> int:
    """Возвращает число шагов в .raw файле (1 если без .step)."""
    raw = RawRead(str(raw_path))
    steps = raw.get_steps()
    if isinstance(steps, range):
        return len(steps)
    return len(list(steps))


def list_traces(raw_path: str | Path) -> list[str]:
    """Возвращает список имён трасс в .raw файле."""
    raw = RawRead(str(raw_path))
    return raw.get_trace_names()


def load_all_steps(raw_path: str | Path,
                   trace_name: str | tuple | None = None
                   ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Загружает все шаги из .raw файла с .step param.

    Возвращает:
        Список кортежей (time, signal) для каждого шага.
    """
    raw = RawRead(str(raw_path))
    trace_spec = _resolve_trace(raw, trace_name)
    steps = raw.get_steps()

    results = []
    for s in steps:
        time = raw.get_wave('time', step=s).real.astype(np.float64)
        signal = _read_signal(raw, trace_spec, s)
        results.append((time, signal))
    return results


# --- Отладка ---
if __name__ == "__main__":
    import sys
    import config

    if len(sys.argv) < 2:
        print("Использование: python loader_ltspice.py <путь_к_raw> [схема]")
        print("  схема: 'bjt' или 'mosfet' (для выбора дифф. трассы из config)")
        print(f"  Трассы в config: {config.LTSPICE_TRACES}")
        sys.exit(1)

    raw_path = sys.argv[1]
    scheme = sys.argv[2] if len(sys.argv) > 2 else None
    trace = config.LTSPICE_TRACES.get(scheme) if scheme else None

    print(f"Файл: {raw_path}")
    print(f"Трассы в файле: {list_traces(raw_path)}")
    print(f"Используемая трасса: {trace or 'авто (первая V(...))'}")

    n_steps = get_step_count(raw_path)
    print(f"Шагов: {n_steps}")

    time, signal = load_raw(raw_path, trace_name=trace, step=0)
    print(f"\nStep 0: {len(time)} точек")
    print(f"  Время: {time[0]:.2e} ... {time[-1]:.2e} с")
    print(f"  Сигнал: мин={signal.min():.4f}, макс={signal.max():.4f} В")
