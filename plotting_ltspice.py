"""Диагностические графики для LTSpice-пайплайна.

Визуализирует КАЖДЫЙ этап обработки: загрузку, обрезку, поиск нулей,
конвертацию индексов в секунды, вычисление периодов, классификацию,
фазовый сдвиг и определение фазы помехи.

7 фигур:
  1. Обзор сигнала (сырой + обрезанный + гистограмма dt)
  2. Качество пересечений нуля (окна линейной аппроксимации + полупериоды)
  3. Классификация периодов (до/после/исключённые)
  4. Разбор вычисления сдвига фазы (накопление + отклонения + фаза помехи)
  5. Валидация по всем шагам (scatter + стабильность T_avg + n_crossings + tdelay)
  6. Конвертация индекс→время (адаптивная сетка)
  7. Качество линейных аппроксимаций (RSS + наихудшие случаи)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from zero_crossing import _find_raw_crossing_indices

# --- Стиль ---
COLOR_SIGNAL = "steelblue"
COLOR_BEFORE = "steelblue"
COLOR_AFTER = "coral"
COLOR_EXCLUDED = "silver"
COLOR_UP = "green"
COLOR_DOWN = "red"
COLOR_PERTURBATION = "black"
COLOR_WARMUP = "orange"
COLOR_FIT = "darkorange"
COLOR_FIT_PTS = "gold"
DPI = 150
FIGSIZE_3P = (14, 10)
FIGSIZE_2P = (14, 7)
FIGSIZE_2x2 = (14, 10)

_ms = lambda t: t * 1e3
_us = lambda t: t * 1e6
_ns = lambda t: t * 1e9


def _step_label(diag):
    """Строка идентификации: 'Шаг 50, Tdelay=7.10 мкс'."""
    return f"Шаг {diag.step}, Tdelay={diag.tdelay*1e6:.2f} мкс"


def _add_source_footer(fig, sources: list[str], diag=None):
    """Добавляет в нижнюю часть фигуры подпись с источниками данных.

    sources — список строк вида "файл.py: функция()".
    diag — LTSpiceDiagnostics (для идентификации шага).
    """
    src_text = "Источники:  " + "  →  ".join(sources)
    if diag is not None:
        id_text = f"Данные: {_step_label(diag)}"
        text = f"{id_text}    |    {src_text}"
    else:
        text = src_text
    fig.text(0.5, 0.002, text, ha="center", va="bottom",
             fontsize=6, color="gray", fontstyle="italic")
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + 0.02)


# =====================================================================
#  Фигура 1: Обзор сигнала
# =====================================================================

def fig_signal_overview(diag, save_dir=None):
    """Полный сигнал, обрезанный сигнал с пересечениями, гистограмма dt."""
    fig, (ax_raw, ax_trim, ax_dt) = plt.subplots(3, 1, figsize=FIGSIZE_3P)

    # --- Панель A: полный сигнал ---
    t_ms = _ms(diag.time_raw)
    ax_raw.plot(t_ms, diag.signal_raw, linewidth=0.3, color=COLOR_SIGNAL)
    ax_raw.axvspan(0, _ms(config.LTSPICE_WARMUP_TIME),
                   alpha=0.15, color=COLOR_WARMUP, label="Прогрев")
    ax_raw.axvspan(_ms(config.LTSPICE_WARMUP_TIME), _ms(config.LTSPICE_TEND),
                   alpha=0.05, color="green", label="Полезное окно")
    ax_raw.axvline(_ms(diag.t_perturbation), color=COLOR_PERTURBATION,
                   ls="--", lw=1.5, label="Помеха")
    ax_raw.set_ylabel("Напряжение, В")
    ax_raw.set_title(f"Шаг {diag.step}: полный сигнал ({len(diag.time_raw)} точек)")
    ax_raw.legend(loc="upper right", fontsize=8)

    # --- Панель B: обрезанный сигнал + пересечения ---
    t_tr_ms = _ms(diag.time_trimmed)
    ax_trim.plot(t_tr_ms, diag.signal_trimmed, linewidth=0.3, color=COLOR_SIGNAL)

    # Рисуем пересечения (не все — при >1000 будет тяжело)
    c_ms = _ms(diag.crossings_time)
    mask_up = diag.crossings_dirs == +1
    ax_trim.scatter(c_ms[mask_up], np.zeros(mask_up.sum()), marker="^",
                    s=4, c=COLOR_UP, zorder=5, label="Вверх")
    ax_trim.scatter(c_ms[~mask_up], np.zeros((~mask_up).sum()), marker="v",
                    s=4, c=COLOR_DOWN, zorder=5, label="Вниз")
    ax_trim.axvline(_ms(diag.t_perturbation), color=COLOR_PERTURBATION,
                    ls="--", lw=1.5)
    ax_trim.axhline(0, color="gray", lw=0.5)
    ax_trim.set_ylabel("Напряжение, В")
    ax_trim.set_title(f"{len(diag.crossings_idx)} пересечений нуля")
    ax_trim.legend(loc="upper right", fontsize=8)

    # Общая ось X для панелей A и B
    ax_raw.set_xlim(t_ms[0], t_ms[-1])
    ax_trim.set_xlim(t_tr_ms[0], t_tr_ms[-1])
    ax_trim.set_xlabel("Время, мс")

    # --- Панель C: гистограмма dt ---
    dt_arr = np.diff(diag.time_trimmed)
    ax_dt.hist(_ns(dt_arr), bins=100, color=COLOR_SIGNAL, alpha=0.7, edgecolor="none")
    ax_dt.set_yscale("log")
    med_dt = np.median(dt_arr)
    ax_dt.axvline(_ns(med_dt), color="green", ls="--", lw=1.5,
                  label=f"Медиана = {_ns(med_dt):.0f} нс")
    ax_dt.set_xlabel("Шаг по времени, нс")
    ax_dt.set_ylabel("Количество (log)")
    ax_dt.set_title("Распределение шагов адаптивной сетки LTSpice")
    ax_dt.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "loader_ltspice.py: load_raw()",
        "pipeline_ltspice.py: _trim_to_useful()",
        "zero_crossing.py: find_zero_crossings()",
        "pipeline_ltspice.py: _crossings_to_time()",
    ], diag)
    if save_dir:
        fig.savefig(save_dir / f"ltspice_diag_step{diag.step:02d}_01_signal_overview.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 2: Качество пересечений нуля
# =====================================================================

def fig_zero_crossing_quality(diag, save_dir=None, n_show=3):
    """Линейная аппроксимация на 3 пересечениях, полупериоды, гистограмма полупериодов."""
    fig, (ax_lsq, ax_hp, ax_hist) = plt.subplots(3, 1, figsize=FIGSIZE_3P)

    signal = diag.signal_trimmed
    time_tr = diag.time_trimmed
    c_idx = diag.crossings_idx
    c_time = diag.crossings_time
    n_fit = config.N_FIT_POINTS

    # --- Панель A: линейная аппроксимация на 3 пересечениях ---
    # Берём 3 пересечения из первой четверти (до помехи)
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

        indices = np.arange(start, end)
        values = signal[start:end]
        # Время для этих точек
        t_pts = time_tr[start:end]
        t_offset = time_tr[raw_i]
        t_pts_us = _us(t_pts - t_offset)

        # Линейная аппроксимация по времени (чтобы на графике выглядела прямой)
        a_t, b_t = np.polyfit(t_pts, values, 1)
        t_dense = np.linspace(t_pts[0], t_pts[-1], 50)
        fit_y_dense = a_t * t_dense + b_t
        t_dense_us = _us(t_dense - t_offset)

        t_refined_us = _us(time_tr[int(refined)] + (refined - int(refined)) *
                          (time_tr[min(int(refined)+1, len(time_tr)-1)] - time_tr[int(refined)])
                          - t_offset)

        # Рисуем
        ax_lsq.scatter(t_pts_us, values, s=40, zorder=5, c=COLOR_FIT_PTS,
                       edgecolors="k", linewidths=0.5,
                       label=f"{n_fit} точек" if k == 0 else "")
        ax_lsq.plot(t_dense_us, fit_y_dense, color=COLOR_FIT, lw=1.5,
                    label="Линейная аппроксимация" if k == 0 else "")
        ax_lsq.axvline(t_refined_us, color=COLOR_PERTURBATION, ls=":", lw=1,
                       label="Уточнённый нуль" if k == 0 else "")

    ax_lsq.axhline(0, color="gray", lw=0.5)
    ax_lsq.set_xlabel("Время относительно пересечения, мкс")
    ax_lsq.set_ylabel("Напряжение, В")
    ax_lsq.set_title(f"Уточнение пересечений нуля линейной аппроксимацией ({n_fit} точек)")
    ax_lsq.legend(fontsize=8)

    # --- Панель B: полупериоды ---
    half_periods_us = _us(np.diff(c_time))
    hp_mid_time = (c_time[:-1] + c_time[1:]) / 2
    colors_hp = np.where(hp_mid_time < diag.t_perturbation, COLOR_BEFORE, COLOR_AFTER)
    ax_hp.scatter(np.arange(len(half_periods_us)), half_periods_us,
                  c=colors_hp, s=2, alpha=0.6)
    med_hp = np.median(half_periods_us)
    ax_hp.axhline(med_hp, color="green", ls="--", lw=1,
                  label=f"Медиана = {med_hp:.2f} мкс")
    ax_hp.set_xlabel("Номер полупериода")
    ax_hp.set_ylabel("Полупериод, мкс")
    ax_hp.set_title(f"Полупериоды ({len(half_periods_us)} шт.)")
    ax_hp.legend(fontsize=8)

    # --- Панель C: гистограмма полупериодов ---
    mean_hp = half_periods_us.mean()
    std_hp = half_periods_us.std()
    ax_hist.hist(half_periods_us, bins=80, color=COLOR_SIGNAL, alpha=0.7, edgecolor="none")
    ax_hist.axvline(mean_hp, color="green", ls="--", lw=1.5,
                    label=f"Среднее = {mean_hp:.3f} мкс")
    ax_hist.axvline(mean_hp - 3*std_hp, color="red", ls=":", lw=1,
                    label=f"-3\u03c3 = {mean_hp - 3*std_hp:.3f}")
    ax_hist.axvline(mean_hp + 3*std_hp, color="red", ls=":", lw=1,
                    label=f"+3\u03c3 = {mean_hp + 3*std_hp:.3f}")
    ax_hist.set_xlabel("Полупериод, мкс")
    ax_hist.set_ylabel("Количество")
    cv = std_hp / mean_hp * 100
    ax_hist.set_title(f"Распределение полупериодов (CV = {cv:.2f}%)")
    ax_hist.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "zero_crossing.py: _find_raw_crossing_indices()",
        "zero_crossing.py: _refine_crossing_lsq(n_fit={})".format(n_fit),
        "pipeline_ltspice.py: _crossings_to_time()",
    ], diag)
    if save_dir:
        fig.savefig(save_dir / f"ltspice_diag_step{diag.step:02d}_02_crossing_quality.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 3: Классификация периодов
# =====================================================================

def fig_period_classification(diag, save_dir=None, zoom_n_periods=3):
    """Периоды vs время, увеличение вокруг помехи, гистограммы до/после."""
    fig, (ax_all, ax_zoom, ax_hist) = plt.subplots(3, 1, figsize=FIGSIZE_3P)

    periods_us = _us(diag.periods)
    starts_ms = _ms(diag.periods_start_time)
    T_avg_us = _us(diag.T_avg)
    t_pert_ms = _ms(diag.t_perturbation)

    # --- Панель A: все периоды ---
    mask_excl = ~diag.mask_before & ~diag.mask_after
    ax_all.scatter(starts_ms[diag.mask_before], periods_us[diag.mask_before],
                   s=5, c=COLOR_BEFORE, label=f"До помехи ({diag.mask_before.sum()})")
    ax_all.scatter(starts_ms[diag.mask_after], periods_us[diag.mask_after],
                   s=5, c=COLOR_AFTER, label=f"После помехи ({diag.mask_after.sum()})")
    ax_all.scatter(starts_ms[mask_excl], periods_us[mask_excl],
                   s=8, c=COLOR_EXCLUDED, marker="x",
                   label=f"Исключённые ({mask_excl.sum()})")
    ax_all.axhline(T_avg_us, color=COLOR_BEFORE, ls=":", lw=1.5,
                   label=f"T_avg = {T_avg_us:.3f} мкс")
    ax_all.axvline(t_pert_ms, color=COLOR_PERTURBATION, ls="--", lw=1.5)
    ax_all.set_xlabel("Время начала периода, мс")
    ax_all.set_ylabel("Период, мкс")
    ax_all.set_title("Классификация периодов: до / после / исключённые")
    ax_all.legend(fontsize=8, loc="upper right")

    # --- Панель B: увеличение вокруг помехи ---
    T_full = diag.T_avg
    t_lo = diag.t_perturbation - zoom_n_periods * T_full
    t_hi = diag.t_perturbation + zoom_n_periods * T_full
    mask_win = (diag.time_trimmed >= t_lo) & (diag.time_trimmed <= t_hi)
    t_win = diag.time_trimmed[mask_win]
    s_win = diag.signal_trimmed[mask_win]

    ax_zoom.plot(_ms(t_win), s_win, lw=0.8, color=COLOR_SIGNAL)

    # Пересечения в окне
    c_in = (diag.crossings_time >= t_lo) & (diag.crossings_time <= t_hi)
    c_t_win = diag.crossings_time[c_in]
    c_d_win = diag.crossings_dirs[c_in]
    for ct, cd in zip(c_t_win, c_d_win):
        clr = COLOR_UP if cd == +1 else COLOR_DOWN
        mkr = "^" if cd == +1 else "v"
        ax_zoom.plot(_ms(ct), 0, mkr, color=clr, markersize=6, zorder=5)

    # Заливка исключённых периодов
    for i in np.where(mask_excl)[0]:
        ps = diag.periods_start_time[i]
        pe = diag.crossings_time[i + 2]
        if ps < t_hi and pe > t_lo:
            ax_zoom.axvspan(_ms(ps), _ms(pe), alpha=0.2, color=COLOR_EXCLUDED)

    ax_zoom.axvline(t_pert_ms, color=COLOR_PERTURBATION, ls="--", lw=2,
                    label="Момент помехи")
    ax_zoom.axhline(0, color="gray", lw=0.5)
    ax_zoom.set_xlabel("Время, мс")
    ax_zoom.set_ylabel("Напряжение, В")
    ax_zoom.set_title(f"Окрестность помехи (\u00b1{zoom_n_periods} периодов)")
    ax_zoom.legend(fontsize=8)

    # --- Панель C: гистограммы до vs после ---
    before_us = _us(diag.periods[diag.mask_before])
    after_us = _us(diag.periods[diag.mask_after])
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
        "pipeline_ltspice.py: _compute_periods_time()",
        "pipeline_ltspice.py: _split_periods_time()",
    ], diag)
    if save_dir:
        fig.savefig(save_dir / f"ltspice_diag_step{diag.step:02d}_03_period_classification.png",
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
    starts_ms = _ms(diag.periods_start_time)
    t_pert_ms = _ms(diag.t_perturbation)

    # --- Панель A: накопленное отклонение фазы ---
    deviations = periods - T_avg
    cumulative = np.cumsum(deviations) / T_avg   # в периодах
    ax_cum.plot(starts_ms, cumulative, lw=1, color=COLOR_SIGNAL)
    ax_cum.axvline(t_pert_ms, color=COLOR_PERTURBATION, ls="--", lw=1.5,
                   label="Момент помехи")
    ax_cum.axhline(0, color="gray", lw=0.5)
    # Аннотация итогового значения
    final = cumulative[-1]
    ax_cum.annotate(f"Итого: {final:.4f} пер. = {final*360:.1f}\u00b0",
                    xy=(starts_ms[-1], final),
                    xytext=(starts_ms[-1] - 1.5, final + 0.02 * np.sign(final)),
                    fontsize=9, color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))
    ax_cum.set_xlabel("Время начала периода, мс")
    ax_cum.set_ylabel("Накопленное отклонение (периоды)")
    ax_cum.set_title("Накопление фазового сдвига")
    ax_cum.legend(fontsize=8)

    # --- Панель B: отклонение каждого периода от T_avg ---
    dev_ns = _ns(deviations)
    colors = np.full(len(periods), COLOR_EXCLUDED, dtype=object)
    colors[diag.mask_before] = COLOR_BEFORE
    colors[diag.mask_after] = COLOR_AFTER
    ax_dev.bar(starts_ms, dev_ns, width=_ms(T_avg) * 0.8, color=colors,
               edgecolor="none", alpha=0.7)
    ax_dev.axvline(t_pert_ms, color=COLOR_PERTURBATION, ls="--", lw=1.5)
    ax_dev.axhline(0, color="gray", lw=0.5)
    ax_dev.set_xlabel("Время начала периода, мс")
    ax_dev.set_ylabel("Отклонение от T_avg, нс")
    ax_dev.set_title("Отклонение каждого периода от среднего")

    # --- Панель C: определение фазы помехи ---
    c_time = diag.crossings_time
    t_pert = diag.t_perturbation

    idx = np.searchsorted(c_time, t_pert, side="right") - 1
    idx = max(0, min(idx, len(c_time) - 2))
    t_left = c_time[idx]
    t_right = c_time[idx + 1]
    frac = (t_pert - t_left) / (t_right - t_left)
    direction = diag.crossings_dirs[idx]

    # Сигнал между двумя пересечениями (с запасом)
    i_left = int(diag.crossings_idx[idx])
    i_right = min(int(diag.crossings_idx[idx + 1]) + 2, len(diag.signal_trimmed))
    i_start = max(i_left - 2, 0)
    t_seg = diag.time_trimmed[i_start:i_right]
    s_seg = diag.signal_trimmed[i_start:i_right]

    ax_interp.plot(_us(t_seg - t_left), s_seg, lw=1.5, color=COLOR_SIGNAL)
    ax_interp.axvline(0, color=COLOR_UP if direction == +1 else COLOR_DOWN,
                      ls="-", lw=2, label=f"Левое пересечение ({'вверх' if direction==+1 else 'вниз'})")
    ax_interp.axvline(_us(t_right - t_left), color=COLOR_DOWN if direction == +1 else COLOR_UP,
                      ls="-", lw=2, label="Правое пересечение")
    ax_interp.axvline(_us(t_pert - t_left), color=COLOR_PERTURBATION,
                      ls="--", lw=2, label="Момент помехи")
    ax_interp.axhline(0, color="gray", lw=0.5)

    phase_deg = np.degrees(diag.perturbation_phase)
    base_str = "0" if direction == +1 else "\u03c0"
    ax_interp.set_title(f"Определение фазы: frac={frac:.3f}, база={base_str}, "
                        f"фаза={phase_deg:.1f}\u00b0")
    ax_interp.set_xlabel("Время от левого пересечения, мкс")
    ax_interp.set_ylabel("Напряжение, В")
    ax_interp.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "pipeline_ltspice.py: _compute_phase_shift_time()",
        "pipeline_ltspice.py: _compute_perturbation_phase_time()",
    ], diag)
    if save_dir:
        fig.savefig(save_dir / f"ltspice_diag_step{diag.step:02d}_04_phase_shift_breakdown.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 5: Валидация по всем шагам
# =====================================================================

def fig_multistep_validation(results, save_dir=None):
    """Scatter фаза-сдвиг, стабильность T_avg, n_crossings, tdelay."""
    fig, ((ax_sc, ax_tavg), (ax_nc, ax_td)) = plt.subplots(2, 2, figsize=FIGSIZE_2x2)

    steps = np.array([r.step for r in results])
    phases = np.degrees(np.array([r.perturbation_phase for r in results]))
    shifts = np.array([r.phase_shift for r in results]) * 360
    tavgs = np.array([r.mean_period_before for r in results])
    ncs = np.array([r.n_crossings for r in results])
    tds = np.array([r.tdelay for r in results])

    # --- A: scatter фаза→сдвиг ---
    ax_sc.scatter(phases, shifts, s=15, color=COLOR_SIGNAL, alpha=0.8)
    ax_sc.axhline(0, color="gray", lw=0.5)
    ax_sc.set_xlim(0, 360)
    ax_sc.set_xlabel("Фаза помехи, град")
    ax_sc.set_ylabel("Сдвиг фазы, град")
    ax_sc.set_title("Зависимость сдвига фазы от фазы помехи")

    # --- B: T_avg по шагам ---
    tavg_us = _us(tavgs)
    ax_tavg.plot(steps, tavg_us, "o-", ms=3, color=COLOR_SIGNAL)
    mean_tavg = tavg_us.mean()
    std_tavg = tavg_us.std()
    ax_tavg.axhline(mean_tavg, color="green", ls="--", lw=1)
    ax_tavg.fill_between(steps, mean_tavg - std_tavg, mean_tavg + std_tavg,
                         alpha=0.2, color="green")
    ax_tavg.set_xlabel("Номер шага")
    ax_tavg.set_ylabel("T_avg, мкс")
    ax_tavg.set_title(f"Стабильность среднего периода "
                      f"({mean_tavg:.3f} \u00b1 {std_tavg:.4f} мкс)")

    # --- C: число пересечений ---
    ax_nc.plot(steps, ncs, "o-", ms=3, color=COLOR_SIGNAL)
    med_nc = np.median(ncs)
    ax_nc.axhline(med_nc, color="green", ls="--", lw=1,
                  label=f"Медиана = {int(med_nc)}")
    ax_nc.set_xlabel("Номер шага")
    ax_nc.set_ylabel("Число пересечений нуля")
    ax_nc.set_title("Число пересечений нуля по шагам")
    ax_nc.legend(fontsize=8)

    # --- D: tdelay ---
    ax_td.plot(steps, _us(tds), "o-", ms=3, color=COLOR_SIGNAL)
    # Линейный фит для проверки
    if len(steps) > 2:
        p = np.polyfit(steps, _us(tds), 1)
        ax_td.plot(steps, np.polyval(p, steps), "--", color=COLOR_FIT, lw=1,
                   label=f"Линейный фит: {p[0]:.3f} мкс/шаг")
    ax_td.set_xlabel("Номер шага")
    ax_td.set_ylabel("Tdelay, мкс")
    ax_td.set_title("Задержка помехи по шагам")
    ax_td.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "pipeline_ltspice.py: process_all_steps()",
        "pipeline_ltspice.py: _read_tdelay_from_raw()",
    ])
    if save_dir:
        fig.savefig(save_dir / "ltspice_diag_05_multistep_validation.png", dpi=DPI)
    return fig


# =====================================================================
#  Фигура 6: Конвертация индекс→время
# =====================================================================

def fig_index_time_mapping(diag, save_dir=None):
    """Кривая time(index) и локальный dt в точках пересечений."""
    fig, (ax_map, ax_ldt) = plt.subplots(2, 1, figsize=FIGSIZE_2P)

    n_pts = len(diag.time_trimmed)
    indices = np.arange(n_pts)

    # --- Панель A: отображение индекс→время ---
    # Прореживаем для отрисовки (каждый 10-й)
    step_draw = max(1, n_pts // 2000)
    ax_map.plot(indices[::step_draw], _ms(diag.time_trimmed[::step_draw]),
                lw=0.5, color="gray", label="time(index)")
    ax_map.scatter(diag.crossings_idx, _ms(diag.crossings_time),
                   s=3, c=COLOR_SIGNAL, zorder=5, label="Пересечения нуля")
    ax_map.set_xlabel("Индекс в обрезанном массиве")
    ax_map.set_ylabel("Время, мс")
    ax_map.set_title("Отображение индекс -> время (адаптивная сетка)")
    ax_map.legend(fontsize=8)

    # --- Панель B: локальный dt в точках пересечений ---
    int_parts = np.floor(diag.crossings_idx).astype(int)
    int_parts = np.clip(int_parts, 0, n_pts - 2)
    local_dt = diag.time_trimmed[int_parts + 1] - diag.time_trimmed[int_parts]
    ax_ldt.scatter(np.arange(len(local_dt)), _ns(local_dt),
                   s=3, c=COLOR_SIGNAL, alpha=0.5)
    med_ldt = np.median(local_dt)
    ax_ldt.axhline(_ns(med_ldt), color="green", ls="--", lw=1,
                   label=f"Медиана = {_ns(med_ldt):.0f} нс")
    ax_ldt.set_xlabel("Номер пересечения нуля")
    ax_ldt.set_ylabel("Локальный dt, нс")
    ax_ldt.set_title("Шаг сетки в точках пересечения нуля")
    ax_ldt.legend(fontsize=8)

    fig.tight_layout()
    _add_source_footer(fig, [
        "pipeline_ltspice.py: _crossings_to_time()",
        "config.py: LTSPICE_WARMUP_TIME, LTSPICE_TEND",
    ], diag)
    if save_dir:
        fig.savefig(save_dir / f"ltspice_diag_step{diag.step:02d}_06_index_time_mapping.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Фигура 7: Качество линейных аппроксимаций
# =====================================================================

def fig_lsq_residuals(diag, save_dir=None, n_worst=3):
    """RSS по всем пересечениям и визуализация наихудших случаев."""
    fig, (ax_rss, ax_worst) = plt.subplots(2, 1, figsize=FIGSIZE_2P)

    signal = diag.signal_trimmed
    time_tr = diag.time_trimmed
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
    colors_rss = np.where(np.arange(n_cross) < n_cross // 2, COLOR_BEFORE, COLOR_AFTER)
    ax_rss.scatter(np.arange(n_cross), rss_arr, s=3, c=colors_rss, alpha=0.5)
    med_rss = np.median(rss_arr)
    ax_rss.axhline(med_rss, color="green", ls="--", lw=1,
                   label=f"Медиана RSS = {med_rss:.2e}")
    ax_rss.set_xlabel("Номер пересечения нуля")
    ax_rss.set_ylabel("RSS (сумма квадратов остатков)")
    ax_rss.set_title(f"Качество линейной аппроксимации ({n_fit} точек) по всем пересечениям")
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
        fit_y = a * idx_arr + b

        t_pts = time_tr[start:end]
        t_pts_us = _us(t_pts - time_tr[raw_i])

        # Прямая по времени (а не по индексам) — чтобы на графике она выглядела прямой
        a_t, b_t = np.polyfit(t_pts, vals, 1)
        t_dense = np.linspace(t_pts[0], t_pts[-1], 50)
        fit_y_dense = a_t * t_dense + b_t
        t_dense_us = _us(t_dense - time_tr[raw_i])

        ax_worst.scatter(t_pts_us + k * 3, vals, s=30, zorder=5,
                         c=COLOR_FIT_PTS, edgecolors="k", linewidths=0.5)
        ax_worst.plot(t_dense_us + k * 3, fit_y_dense, color=COLOR_FIT, lw=1.5)
        ax_worst.text(t_pts_us.mean() + k * 3, vals.max(),
                      f"#{ci}\nRSS={rss_arr[ci]:.1e}", fontsize=7, ha="center")

    ax_worst.axhline(0, color="gray", lw=0.5)
    ax_worst.set_xlabel("Время (смещённое), мкс")
    ax_worst.set_ylabel("Напряжение, В")
    ax_worst.set_title(f"{n_worst} наихудших линейных аппроксимаций")

    fig.tight_layout()
    _add_source_footer(fig, [
        "zero_crossing.py: _refine_crossing_lsq()",
        "numpy.polyfit(deg=1)",
    ], diag)
    if save_dir:
        fig.savefig(save_dir / f"ltspice_diag_step{diag.step:02d}_07_lsq_residuals.png",
                    dpi=DPI)
    return fig


# =====================================================================
#  Обёртка: все графики разом
# =====================================================================

def plot_all_diagnostics(diag, all_results=None, save_dir=None):
    """Генерирует все диагностические фигуры.

    Параметры:
        diag: LTSpiceDiagnostics для одного выбранного шага
        all_results: list[LTSpiceResult] для всех шагов (для фигуры 5)
        save_dir: Path — куда сохранять PNG (None = не сохранять)
    """
    figs = []
    figs.append(fig_signal_overview(diag, save_dir))
    figs.append(fig_zero_crossing_quality(diag, save_dir))
    figs.append(fig_period_classification(diag, save_dir))
    figs.append(fig_phase_shift_breakdown(diag, save_dir))
    if all_results is not None:
        figs.append(fig_multistep_validation(all_results, save_dir))
    figs.append(fig_index_time_mapping(diag, save_dir))
    figs.append(fig_lsq_residuals(diag, save_dir))
    return figs


# =====================================================================
#  CLI
# =====================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Использование: python plotting_ltspice.py <путь_к_raw> [схема] [шаг]")
        print("  схема: 'bjt' (по умолчанию) или 'mosfet'")
        print("  шаг: номер шага для детальной диагностики (по умолчанию 50)")
        sys.exit(1)

    raw_path = sys.argv[1]
    scheme = sys.argv[2] if len(sys.argv) > 2 else "bjt"
    diag_step = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    trace = config.LTSPICE_TRACES.get(scheme)
    save_dir = config.OUTPUT_DIR
    save_dir.mkdir(exist_ok=True)

    from pipeline_ltspice import (
        process_single_step_diagnostics, process_all_steps,
        _read_tdelay_from_raw,
    )
    from loader_ltspice import get_step_count

    n_steps = get_step_count(raw_path)
    diag_step = min(diag_step, n_steps - 1)

    print(f"Файл: {raw_path}, схема: {scheme}, шагов: {n_steps}")
    print(f"Детальная диагностика: шаг {diag_step}")

    # Определяем tdelay для выбранного шага
    tdelays = _read_tdelay_from_raw(raw_path, n_steps)
    if tdelays is not None:
        td = float(tdelays[diag_step])
    else:
        td = 0.0

    print(f"  tdelay = {td*1e6:.2f} мкс")
    print("Загружаю диагностику одного шага...")
    diag = process_single_step_diagnostics(raw_path, trace, diag_step, td)

    print("Обрабатываю все шаги для мульти-степ валидации...")
    all_results = process_all_steps(raw_path, trace)

    print(f"Генерирую 7 фигур в {save_dir}/ ...")
    figs = plot_all_diagnostics(diag, all_results, save_dir)
    for f in figs:
        plt.close(f)

    print("Готово! Файлы:")
    for p in sorted(save_dir.glob("ltspice_diag_*")):
        print(f"  {p.name}")
