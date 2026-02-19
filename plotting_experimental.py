"""Диагностические графики для экспериментального пайплайна (CSV осциллографа).

Визуализирует КАЖДЫЙ этап обработки: загрузку, поиск нулей,
вычисление периодов, классификацию, фазовый сдвиг и определение фазы помехи.

6 фигур:
  1. Обзор сигнала (полный сигнал + пересечения нуля)
  2. Качество пересечений нуля (окна линейной аппроксимации + полупериоды)
  3. Классификация периодов (до/после/исключённые)
  4. Разбор вычисления сдвига фазы (накопление + отклонения + фаза помехи)
  5. Валидация по всем файлам (scatter + стабильность T_avg + n_crossings + эталон)
  6. Качество линейных аппроксимаций (RSS + наихудшие случаи)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from zero_crossing import _find_raw_crossing_indices

# --- Стиль (аналогичен plotting_ltspice.py) ---
COLOR_SIGNAL = "steelblue"
COLOR_BEFORE = "steelblue"
COLOR_AFTER = "coral"
COLOR_EXCLUDED = "silver"
COLOR_UP = "green"
COLOR_DOWN = "red"
COLOR_PERTURBATION = "black"
COLOR_FIT = "darkorange"
COLOR_FIT_PTS = "gold"
DPI = 150
FIGSIZE_3P = (14, 10)
FIGSIZE_2P = (14, 7)
FIGSIZE_2x2 = (14, 10)

_us = lambda samples: samples * config.DT * 1e6  # отсчёты → мкс


def _file_label(file_number):
    """Строка идентификации: 'Файл 0.csv (эталон)' или 'Файл 42.csv'."""
    if file_number == config.REFERENCE_FILE:
        return f"Файл {file_number}.csv (эталон, без помехи)"
    return f"Файл {file_number}.csv"


def _add_source_footer(fig, sources: list[str], file_number: int):
    """Подпись внизу фигуры: источники данных + идентификация файла."""
    src_text = "Источники:  " + "  →  ".join(sources)
    id_text = f"Данные: {_file_label(file_number)}"
    fig.text(0.5, 0.002, f"{id_text}    |    {src_text}",
             ha="center", va="bottom", fontsize=6, color="gray", fontstyle="italic")
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.02)


# =====================================================================
#  Фигура 1: Обзор сигнала
# =====================================================================

def fig_signal_overview(diag, save_dir=None):
    """Полный сигнал с пересечениями нуля, гистограмма полупериодов."""
    fig, (ax_sig, ax_hist) = plt.subplots(2, 1, figsize=FIGSIZE_2P)

    signal = diag.signal
    n = len(signal)
    x = np.arange(n)
    center = config.CENTER_INDEX

    # --- Панель A: полный сигнал ---
    ax_sig.plot(x, signal, linewidth=0.3, color=COLOR_SIGNAL)

    # Пересечения нуля
    c_idx = diag.crossings_idx
    mask_up = diag.crossings_dirs == +1
    ax_sig.scatter(c_idx[mask_up], np.zeros(mask_up.sum()), marker="^",
                   s=4, c=COLOR_UP, zorder=5, label="Вверх")
    ax_sig.scatter(c_idx[~mask_up], np.zeros((~mask_up).sum()), marker="v",
                   s=4, c=COLOR_DOWN, zorder=5, label="Вниз")

    ax_sig.axvline(center, color=COLOR_PERTURBATION, ls="--", lw=1.5,
                   label=f"Помеха (отсчёт {center})")
    ax_sig.axhline(0, color="gray", lw=0.5)
    ax_sig.set_xlabel("Отсчёт")
    ax_sig.set_ylabel("Напряжение, В")
    ax_sig.set_title(f"{_file_label(diag.file_number)}: сигнал "
                     f"({n} отсчётов, dt={config.DT*1e9:.0f} нс, "
                     f"{len(c_idx)} пересечений нуля)")
    ax_sig.legend(loc="upper right", fontsize=8)

    # --- Панель B: гистограмма полупериодов ---
    half_periods = np.diff(c_idx)
    hp_us = _us(half_periods)
    mean_hp = hp_us.mean()
    std_hp = hp_us.std()
    ax_hist.hist(hp_us, bins=60, color=COLOR_SIGNAL, alpha=0.7, edgecolor="none")
    ax_hist.axvline(mean_hp, color="green", ls="--", lw=1.5,
                    label=f"Среднее = {mean_hp:.3f} мкс")
    ax_hist.axvline(mean_hp - 3*std_hp, color="red", ls=":", lw=1,
                    label=f"-3σ = {mean_hp - 3*std_hp:.3f}")
    ax_hist.axvline(mean_hp + 3*std_hp, color="red", ls=":", lw=1,
                    label=f"+3σ = {mean_hp + 3*std_hp:.3f}")
    ax_hist.set_xlabel("Полупериод, мкс")
    ax_hist.set_ylabel("Количество")
    cv = std_hp / mean_hp * 100
    ax_hist.set_title(f"Распределение полупериодов (CV = {cv:.2f}%)")
    ax_hist.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "loader.py: load_signal()",
        "zero_crossing.py: find_zero_crossings()",
        "zero_crossing.py: crossing_directions()",
    ], diag.file_number)
    if save_dir:
        fig.savefig(save_dir / f"exp_diag_file{diag.file_number:03d}_01_signal_overview.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 2: Качество пересечений нуля
# =====================================================================

def fig_zero_crossing_quality(diag, save_dir=None, n_show=3):
    """Окна линейной аппроксимации, полупериоды, гистограмма полупериодов."""
    fig, (ax_lsq, ax_hp, ax_hist) = plt.subplots(3, 1, figsize=FIGSIZE_3P)

    signal = diag.signal
    c_idx = diag.crossings_idx
    n_fit = config.N_FIT_POINTS
    center = config.CENTER_INDEX

    # --- Панель A: линейная аппроксимация на 3 пересечениях ---
    quarter = max(3, len(c_idx) // 4)
    sel = np.linspace(quarter - n_show, quarter, n_show, dtype=int)

    raw_indices = _find_raw_crossing_indices(signal)

    for k, ci in enumerate(sel):
        if ci >= len(c_idx) or ci >= len(raw_indices):
            continue
        raw_i = raw_indices[ci]
        refined = c_idx[ci]
        half = n_fit // 2

        start = raw_i - (half - 1)
        end = start + n_fit
        if start < 0:
            start = 0; end = n_fit
        if end > len(signal):
            end = len(signal); start = end - n_fit

        indices = np.arange(start, end, dtype=float)
        values = signal[start:end]

        # Линейная аппроксимация
        a, b = np.polyfit(indices, values, 1)
        x_dense = np.linspace(indices[0], indices[-1], 50)
        fit_y = a * x_dense + b

        offset = raw_i
        x_pts = indices - offset
        x_dense_off = x_dense - offset

        # Рисуем
        ax_lsq.scatter(x_pts, values, s=40, zorder=5, c=COLOR_FIT_PTS,
                       edgecolors="k", linewidths=0.5,
                       label=f"{n_fit} точек" if k == 0 else "")
        ax_lsq.plot(x_dense_off, fit_y, color=COLOR_FIT, lw=1.5,
                    label="Линейная аппроксимация" if k == 0 else "")
        ax_lsq.axvline(refined - offset, color=COLOR_PERTURBATION, ls=":", lw=1,
                       label="Уточнённый нуль" if k == 0 else "")

    ax_lsq.axhline(0, color="gray", lw=0.5)
    ax_lsq.set_xlabel("Отсчёт (относительно пересечения)")
    ax_lsq.set_ylabel("Напряжение, В")
    ax_lsq.set_title(f"{_file_label(diag.file_number)}: "
                     f"уточнение пересечений нуля линейной аппроксимацией ({n_fit} точек)")
    ax_lsq.legend(fontsize=8)

    # --- Панель B: полупериоды ---
    half_periods = np.diff(c_idx)
    hp_us = _us(half_periods)
    hp_mid = (c_idx[:-1] + c_idx[1:]) / 2
    colors_hp = np.where(hp_mid < center, COLOR_BEFORE, COLOR_AFTER)
    ax_hp.scatter(np.arange(len(hp_us)), hp_us, c=colors_hp, s=2, alpha=0.6)
    med_hp = np.median(hp_us)
    ax_hp.axhline(med_hp, color="green", ls="--", lw=1,
                  label=f"Медиана = {med_hp:.2f} мкс")
    ax_hp.set_xlabel("Номер полупериода")
    ax_hp.set_ylabel("Полупериод, мкс")
    ax_hp.set_title(f"Полупериоды ({len(hp_us)} шт.)")
    ax_hp.legend(fontsize=8)

    # --- Панель C: гистограмма полупериодов ---
    mean_hp = hp_us.mean()
    std_hp = hp_us.std()
    ax_hist.hist(hp_us, bins=80, color=COLOR_SIGNAL, alpha=0.7, edgecolor="none")
    ax_hist.axvline(mean_hp, color="green", ls="--", lw=1.5,
                    label=f"Среднее = {mean_hp:.3f} мкс")
    ax_hist.axvline(mean_hp - 3*std_hp, color="red", ls=":", lw=1,
                    label=f"-3σ = {mean_hp - 3*std_hp:.3f}")
    ax_hist.axvline(mean_hp + 3*std_hp, color="red", ls=":", lw=1,
                    label=f"+3σ = {mean_hp + 3*std_hp:.3f}")
    ax_hist.set_xlabel("Полупериод, мкс")
    ax_hist.set_ylabel("Количество")
    cv = std_hp / mean_hp * 100
    ax_hist.set_title(f"Распределение полупериодов (CV = {cv:.2f}%)")
    ax_hist.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "zero_crossing.py: _find_raw_crossing_indices()",
        f"zero_crossing.py: _refine_crossing_lsq(n_fit={n_fit})",
    ], diag.file_number)
    if save_dir:
        fig.savefig(save_dir / f"exp_diag_file{diag.file_number:03d}_02_crossing_quality.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 3: Классификация периодов
# =====================================================================

def fig_period_classification(diag, save_dir=None, zoom_n_periods=3):
    """Периоды vs позиция, увеличение вокруг помехи, гистограммы до/после."""
    fig, (ax_all, ax_zoom, ax_hist) = plt.subplots(3, 1, figsize=FIGSIZE_3P)

    periods = diag.periods
    starts = diag.periods_start
    T_avg = diag.T_avg
    center = config.CENTER_INDEX
    periods_us = _us(periods)
    T_avg_us = _us(T_avg)

    # --- Панель A: все периоды ---
    mask_excl = ~diag.mask_before & ~diag.mask_after
    ax_all.scatter(starts[diag.mask_before], periods_us[diag.mask_before],
                   s=5, c=COLOR_BEFORE, label=f"До помехи ({diag.mask_before.sum()})")
    ax_all.scatter(starts[diag.mask_after], periods_us[diag.mask_after],
                   s=5, c=COLOR_AFTER, label=f"После помехи ({diag.mask_after.sum()})")
    ax_all.scatter(starts[mask_excl], periods_us[mask_excl],
                   s=8, c=COLOR_EXCLUDED, marker="x",
                   label=f"Исключённые ({mask_excl.sum()})")
    ax_all.axhline(T_avg_us, color=COLOR_BEFORE, ls=":", lw=1.5,
                   label=f"T_avg = {T_avg_us:.3f} мкс")
    ax_all.axvline(center, color=COLOR_PERTURBATION, ls="--", lw=1.5)
    ax_all.set_xlabel("Позиция начала периода (отсчёт)")
    ax_all.set_ylabel("Период, мкс")
    ax_all.set_title(f"{_file_label(diag.file_number)}: "
                     f"классификация периодов: до / после / исключённые")
    ax_all.legend(fontsize=8, loc="upper right")

    # --- Панель B: увеличение вокруг помехи ---
    win_samples = int(zoom_n_periods * T_avg)
    i_lo = max(0, center - win_samples)
    i_hi = min(len(diag.signal), center + win_samples)
    x_win = np.arange(i_lo, i_hi)
    s_win = diag.signal[i_lo:i_hi]

    ax_zoom.plot(x_win, s_win, lw=0.8, color=COLOR_SIGNAL)

    # Пересечения в окне
    c_in = (diag.crossings_idx >= i_lo) & (diag.crossings_idx <= i_hi)
    c_win = diag.crossings_idx[c_in]
    d_win = diag.crossings_dirs[c_in]
    for ct, cd in zip(c_win, d_win):
        clr = COLOR_UP if cd == +1 else COLOR_DOWN
        mkr = "^" if cd == +1 else "v"
        ax_zoom.plot(ct, 0, mkr, color=clr, markersize=6, zorder=5)

    # Заливка исключённых периодов
    period_ends = diag.crossings_idx[2:]
    for i in np.where(mask_excl)[0]:
        ps = starts[i]
        pe = period_ends[i]
        if ps < i_hi and pe > i_lo:
            ax_zoom.axvspan(ps, pe, alpha=0.2, color=COLOR_EXCLUDED)

    ax_zoom.axvline(center, color=COLOR_PERTURBATION, ls="--", lw=2,
                    label="Момент помехи")
    ax_zoom.axhline(0, color="gray", lw=0.5)
    ax_zoom.set_xlabel("Отсчёт")
    ax_zoom.set_ylabel("Напряжение, В")
    ax_zoom.set_title(f"Окрестность помехи (±{zoom_n_periods} периодов)")
    ax_zoom.legend(fontsize=8)

    # --- Панель C: гистограммы до vs после ---
    before_us = _us(periods[diag.mask_before])
    after_us = _us(periods[diag.mask_after])
    all_us = np.concatenate([before_us, after_us])
    bins = np.linspace(all_us.min() - 0.05, all_us.max() + 0.05, 60)
    ax_hist.hist(before_us, bins=bins, alpha=0.6, color=COLOR_BEFORE,
                 label=f"До помехи (N={len(before_us)})")
    ax_hist.hist(after_us, bins=bins, alpha=0.6, color=COLOR_AFTER,
                 label=f"После помехи (N={len(after_us)})")
    ax_hist.axvline(T_avg_us, color=COLOR_PERTURBATION, ls="--", lw=1.5,
                    label=f"T_avg = {T_avg_us:.3f} мкс")
    ax_hist.set_xlabel("Период, мкс")
    ax_hist.set_ylabel("Количество")
    ax_hist.set_title("Распределение периодов до и после помехи")
    ax_hist.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "period_analysis.py: compute_periods()",
        "period_analysis.py: split_periods_at_center()",
    ], diag.file_number)
    if save_dir:
        fig.savefig(save_dir / f"exp_diag_file{diag.file_number:03d}_03_period_classification.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 4: Разбор вычисления сдвига фазы
# =====================================================================

def fig_phase_shift_breakdown(diag, save_dir=None):
    """Накопленное отклонение, отклонения по периодам, определение фазы помехи."""
    fig, (ax_cum, ax_dev, ax_interp) = plt.subplots(3, 1, figsize=FIGSIZE_3P)

    T_avg = diag.T_avg
    periods = diag.periods
    starts = diag.periods_start
    center = config.CENTER_INDEX

    # --- Панель A: накопленное отклонение фазы ---
    deviations = periods - T_avg
    cumulative = np.cumsum(deviations) / T_avg   # в периодах
    ax_cum.plot(starts, cumulative, lw=1, color=COLOR_SIGNAL)
    ax_cum.axvline(center, color=COLOR_PERTURBATION, ls="--", lw=1.5,
                   label="Момент помехи")
    ax_cum.axhline(0, color="gray", lw=0.5)
    # Аннотация итогового значения
    final = cumulative[-1]
    ax_cum.annotate(f"Итого: {final:.4f} пер. = {final*360:.1f}°",
                    xy=(starts[-1], final),
                    xytext=(starts[-1] - len(starts)*0.15, final + 0.02 * np.sign(final)),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax_cum.set_xlabel("Позиция начала периода (отсчёт)")
    ax_cum.set_ylabel("Накопленное отклонение (периоды)")
    ax_cum.set_title(f"{_file_label(diag.file_number)}: накопление фазового сдвига")
    ax_cum.legend(fontsize=8)

    # --- Панель B: отклонение каждого периода от T_avg ---
    dev_us = _us(deviations)
    colors = np.full(len(periods), COLOR_EXCLUDED, dtype=object)
    colors[diag.mask_before] = COLOR_BEFORE
    colors[diag.mask_after] = COLOR_AFTER
    ax_dev.bar(starts, dev_us, width=T_avg * 0.8, color=colors,
               edgecolor="none", alpha=0.7)
    ax_dev.axvline(center, color=COLOR_PERTURBATION, ls="--", lw=1.5)
    ax_dev.axhline(0, color="gray", lw=0.5)
    ax_dev.set_xlabel("Позиция начала периода (отсчёт)")
    ax_dev.set_ylabel("Отклонение от T_avg, мкс")
    ax_dev.set_title("Отклонение каждого периода от среднего")

    # --- Панель C: определение фазы помехи ---
    c_idx = diag.crossings_idx

    idx = np.searchsorted(c_idx, center, side="right") - 1
    idx = max(0, min(idx, len(c_idx) - 2))
    c_left = c_idx[idx]
    c_right = c_idx[idx + 1]
    frac = (center - c_left) / (c_right - c_left)
    direction = diag.crossings_dirs[idx]

    # Сигнал между двумя пересечениями (с запасом)
    i_left = int(c_left)
    i_right = min(int(c_right) + 2, len(diag.signal))
    i_start = max(i_left - 2, 0)
    x_seg = np.arange(i_start, i_right)
    s_seg = diag.signal[i_start:i_right]

    ax_interp.plot(x_seg - c_left, s_seg, lw=1.5, color=COLOR_SIGNAL)
    ax_interp.axvline(0, color=COLOR_UP if direction == +1 else COLOR_DOWN,
                      ls="-", lw=2,
                      label=f"Левое пересечение ({'вверх' if direction==+1 else 'вниз'})")
    ax_interp.axvline(c_right - c_left,
                      color=COLOR_DOWN if direction == +1 else COLOR_UP,
                      ls="-", lw=2, label="Правое пересечение")
    ax_interp.axvline(center - c_left, color=COLOR_PERTURBATION,
                      ls="--", lw=2, label="Момент помехи")
    ax_interp.axhline(0, color="gray", lw=0.5)

    phase_deg = np.degrees(diag.perturbation_phase)
    base_str = "0" if direction == +1 else "π"
    ax_interp.set_title(f"Определение фазы: frac={frac:.3f}, база={base_str}, "
                        f"фаза={phase_deg:.1f}°")
    ax_interp.set_xlabel("Отсчёт от левого пересечения")
    ax_interp.set_ylabel("Напряжение, В")
    ax_interp.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "period_analysis.py: compute_phase_shift()",
        "period_analysis.py: compute_perturbation_phase()",
    ], diag.file_number)
    if save_dir:
        fig.savefig(save_dir / f"exp_diag_file{diag.file_number:03d}_04_phase_shift_breakdown.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 5: Валидация по всем файлам
# =====================================================================

def fig_multifile_validation(results, ref_result=None, save_dir=None):
    """Scatter фаза-сдвиг, стабильность T_avg, n_crossings, проверка эталона."""
    fig, ((ax_sc, ax_tavg), (ax_nc, ax_ref)) = plt.subplots(2, 2, figsize=FIGSIZE_2x2)

    files = np.array([r.file_number for r in results])
    phases = np.degrees(np.array([r.perturbation_phase for r in results]))
    shifts = np.array([r.phase_shift for r in results]) * 360
    tavgs = np.array([r.mean_period_before for r in results])
    ncs = np.array([r.n_crossings for r in results])

    # --- A: scatter фаза→сдвиг ---
    ax_sc.scatter(phases, shifts, s=15, color=COLOR_SIGNAL, alpha=0.8)
    ax_sc.axhline(0, color="gray", lw=0.5)
    ax_sc.set_xlim(0, 360)
    ax_sc.set_xlabel("Фаза помехи, град")
    ax_sc.set_ylabel("Сдвиг фазы, град")
    ax_sc.set_title(f"Файлы 1–{files[-1]}: фаза помехи vs сдвиг фазы")

    # --- B: T_avg по файлам ---
    tavg_us = _us(tavgs)
    ax_tavg.plot(files, tavg_us, "o-", ms=3, color=COLOR_SIGNAL)
    mean_tavg = tavg_us.mean()
    std_tavg = tavg_us.std()
    ax_tavg.axhline(mean_tavg, color="green", ls="--", lw=1)
    ax_tavg.fill_between(files, mean_tavg - std_tavg, mean_tavg + std_tavg,
                         alpha=0.2, color="green")
    ax_tavg.set_xlabel("Номер файла")
    ax_tavg.set_ylabel("T_avg, мкс")
    ax_tavg.set_title(f"Стабильность среднего периода "
                      f"({mean_tavg:.3f} ± {std_tavg:.4f} мкс)")

    # --- C: число пересечений ---
    ax_nc.plot(files, ncs, "o-", ms=3, color=COLOR_SIGNAL)
    med_nc = np.median(ncs)
    ax_nc.axhline(med_nc, color="green", ls="--", lw=1,
                  label=f"Медиана = {int(med_nc)}")
    ax_nc.set_xlabel("Номер файла")
    ax_nc.set_ylabel("Число пересечений нуля")
    ax_nc.set_title("Число пересечений нуля по файлам")
    ax_nc.legend(fontsize=8)

    # --- D: проверка эталона ---
    if ref_result is not None:
        ref_shift = ref_result.phase_shift * 360
        ax_ref.bar(["Эталон\n(файл 0)"], [ref_shift],
                   color=COLOR_BEFORE if abs(ref_shift) < 5 else "red",
                   edgecolor="k", width=0.4)
        ax_ref.axhline(0, color="gray", lw=0.5)
        ax_ref.set_ylabel("Сдвиг фазы, град")
        ax_ref.set_title(f"Файл 0 (без помехи): сдвиг = {ref_shift:.2f}°")
        # Добавляем для сравнения средний сдвиг файлов с помехой
        ax_ref.bar(["Среднее\nфайлов 1–100"], [shifts.mean()],
                   color=COLOR_AFTER, edgecolor="k", width=0.4)
    else:
        ax_ref.text(0.5, 0.5, "Эталон (файл 0)\nне обработан",
                    ha="center", va="center", transform=ax_ref.transAxes, fontsize=12)
        ax_ref.set_title("Проверка эталона")

    fig.tight_layout()
    _add_source_footer(fig, [
        "pipeline.py: process_all_files()",
        "pipeline.py: process_single_file()",
    ], 0)
    if save_dir:
        fig.savefig(save_dir / "exp_diag_05_multifile_validation.png", dpi=DPI)
    return fig


# =====================================================================
#  Фигура 6: Качество линейных аппроксимаций
# =====================================================================

def fig_lsq_residuals(diag, save_dir=None, n_worst=3):
    """RSS по всем пересечениям и визуализация наихудших случаев."""
    fig, (ax_rss, ax_worst) = plt.subplots(2, 1, figsize=FIGSIZE_2P)

    signal = diag.signal
    n_fit = config.N_FIT_POINTS
    half = n_fit // 2

    raw_indices = _find_raw_crossing_indices(signal)
    n_cross = min(len(raw_indices), len(diag.crossings_idx))

    rss_arr = np.zeros(n_cross)
    for ci in range(n_cross):
        raw_i = raw_indices[ci]
        start = raw_i - (half - 1)
        end = start + n_fit
        if start < 0:
            start = 0; end = n_fit
        if end > len(signal):
            end = len(signal); start = end - n_fit
        idx_arr = np.arange(start, end, dtype=np.float64)
        vals = signal[start:end]
        a, b = np.polyfit(idx_arr, vals, 1)
        fitted = a * idx_arr + b
        rss_arr[ci] = np.sum((vals - fitted) ** 2)

    # --- Панель A: RSS по всем пересечениям ---
    center = config.CENTER_INDEX
    colors_rss = np.where(
        diag.crossings_idx[:n_cross] < center, COLOR_BEFORE, COLOR_AFTER)
    ax_rss.scatter(np.arange(n_cross), rss_arr, s=3, c=colors_rss, alpha=0.5)
    med_rss = np.median(rss_arr)
    ax_rss.axhline(med_rss, color="green", ls="--", lw=1,
                   label=f"Медиана RSS = {med_rss:.2e}")
    ax_rss.set_xlabel("Номер пересечения нуля")
    ax_rss.set_ylabel("RSS (сумма квадратов остатков)")
    ax_rss.set_title(f"{_file_label(diag.file_number)}: "
                     f"качество линейной аппроксимации ({n_fit} точек)")
    ax_rss.legend(fontsize=8)

    # --- Панель B: наихудшие случаи ---
    worst_idx = np.argsort(rss_arr)[-n_worst:]
    for k, ci in enumerate(worst_idx):
        if ci >= len(raw_indices):
            continue
        raw_i = raw_indices[ci]
        start = raw_i - (half - 1)
        end = start + n_fit
        if start < 0:
            start = 0; end = n_fit
        if end > len(signal):
            end = len(signal); start = end - n_fit

        idx_arr = np.arange(start, end, dtype=np.float64)
        vals = signal[start:end]
        a, b = np.polyfit(idx_arr, vals, 1)
        x_dense = np.linspace(idx_arr[0], idx_arr[-1], 50)
        fit_y = a * x_dense + b

        x_pts = idx_arr - raw_i
        x_dense_off = x_dense - raw_i

        ax_worst.scatter(x_pts + k * 12, vals, s=30, zorder=5,
                         c=COLOR_FIT_PTS, edgecolors="k", linewidths=0.5)
        ax_worst.plot(x_dense_off + k * 12, fit_y, color=COLOR_FIT, lw=1.5)
        ax_worst.text(x_pts.mean() + k * 12, vals.max(),
                      f"#{ci}\nRSS={rss_arr[ci]:.1e}", fontsize=7, ha="center")

    ax_worst.axhline(0, color="gray", lw=0.5)
    ax_worst.set_xlabel("Отсчёт (смещённый)")
    ax_worst.set_ylabel("Напряжение, В")
    ax_worst.set_title(f"{n_worst} наихудших линейных аппроксимаций")

    fig.tight_layout()
    _add_source_footer(fig, [
        "zero_crossing.py: _refine_crossing_lsq()",
        "numpy.polyfit(deg=1)",
    ], diag.file_number)
    if save_dir:
        fig.savefig(save_dir / f"exp_diag_file{diag.file_number:03d}_06_lsq_residuals.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Обёртка: все графики разом
# =====================================================================

def plot_all_diagnostics(diag, all_results=None, ref_result=None, save_dir=None):
    """Генерирует все диагностические фигуры.

    Параметры:
        diag: ExperimentDiagnostics для одного выбранного файла
        all_results: list[FileResult] для всех файлов (для фигуры 5)
        ref_result: FileResult для файла 0 (для фигуры 5, панель D)
        save_dir: Path — куда сохранять PNG (None = не сохранять)
    """
    figs = []
    figs.append(fig_signal_overview(diag, save_dir))
    figs.append(fig_zero_crossing_quality(diag, save_dir))
    figs.append(fig_period_classification(diag, save_dir))
    figs.append(fig_phase_shift_breakdown(diag, save_dir))
    if all_results is not None:
        figs.append(fig_multifile_validation(all_results, ref_result, save_dir))
    figs.append(fig_lsq_residuals(diag, save_dir))
    return figs


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    file_number = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    save_dir = config.OUTPUT_DIR
    save_dir.mkdir(exist_ok=True)

    from pipeline import (
        process_single_file_diagnostics, process_single_file,
        process_all_files,
    )

    print(f"Детальная диагностика: {_file_label(file_number)}")
    diag = process_single_file_diagnostics(file_number)

    print("Обрабатываю все файлы для мульти-файл валидации...")
    all_results = process_all_files()
    ref_result = process_single_file(config.REFERENCE_FILE)

    print(f"Генерирую 6 фигур в {save_dir}/ ...")
    figs = plot_all_diagnostics(diag, all_results, ref_result, save_dir)
    for f in figs:
        plt.close(f)

    print("Готово! Файлы:")
    for p in sorted(save_dir.glob("exp_diag_*")):
        print(f"  {p.name}")
