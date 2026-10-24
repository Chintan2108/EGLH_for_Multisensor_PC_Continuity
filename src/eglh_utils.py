"""
Helper functions for the EGLH phycocyanin (PC) analysis.

Kumar, Maniyar and Mishra - "A Cross-calibrated Phycocyanin Index for Landsat and Ocean Color Sensors
for Long-Term Monitoring of Cyanobacteria in Inland Waterbodies".

The notebooks in ../notebooks import this file as
    sys.path.append('../src'); import eglh_utils as utils

Reflectance convention: all PC models (Table 2, Equations 3-6) use ACOLITE surface reflectance (rho_s,
dimensionless). If you only have remote-sensing reflectance (Rrs, 1/sr), convert first: rho_s = pi * Rrs.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm

RANDOM_SEED = 42        # fixed seed so bootstrap results are reproducible
N_BOOTSTRAP = 2000      # number of bootstrap resamples used in the paper

################################################# SENSOR BANDS AND INDEX NAMES ####################################################
###################################################################################################################################

# Band centres (nm) of the bands used by the indices. OLI "orange" is the virtual orange band (Castagna et al., 2020).
BAND_CENTRES_NM = {
    "OLCI":  {"blue": 490, "green": 560, "orange": 620, "red": 665, "nir": 865},
    "MERIS": {"blue": 490, "green": 560, "orange": 620, "red": 665, "nir": 865},
    "OLI":   {"blue": 480, "green": 560, "orange": 613, "red": 655, "nir": 865},
}

# Virtual orange band of Landsat 8/9 OLI (Castagna et al., 2020): orange = 2.2861 pan - 0.9467 green - 0.1989 red
VIRTUAL_ORANGE_WEIGHTS = {"pan": 2.2861, "green": -0.9467, "red": -0.1989}

INDEX_LABELS = {
    "orange_red": "Orange:Red ratio",
    "orange_green": "Orange:Green ratio",
    "nir_orange": "NIR:Orange ratio",
    "OLH": "Orange Line Height (OLH)",
    "GLH": "Green Line Height (GLH)",
    "EGLH": "Enhanced Green Line Height (EGLH)",
}

# Okabe-Ito colour-blind-safe colours and marker shapes for the eight 2024-2025 lakes
LAKE_STYLE = {
    "Western Lake Erie": ("#000000", "o"), "Green Bay": ("#E69F00", "s"),
    "Utah Lake": ("#56B4E9", "^"), "Lake Okeechobee": ("#009E73", "D"),
    "Clear Lake": ("#F0E442", "v"), "San Luis Reservoir": ("#0072B2", "P"),
    "Lake Pontchartrain": ("#D55E00", "X"), "Lake Winnebago": ("#CC79A7", "*"),
}

################################################# BAND SIMULATION FROM HYPERSPECTRAL Rrs ##########################################
###################################################################################################################################

def get_virtual_orange(pan, green, red):
    """Landsat OLI virtual orange band from the panchromatic, green and red bands (all in the same unit)."""
    w = VIRTUAL_ORANGE_WEIGHTS
    return w["pan"] * pan + w["green"] * green + w["red"] * red


def convolve_band(spectra, spectra_wavelengths, rsr_wavelengths, rsr_values, normalise="weighted"):
    """Band-averaged reflectance of every spectrum for one sensor band.

    spectra             array (n_spectra, n_wavelengths) of Rrs at `spectra_wavelengths` (nm, 1 nm steps)
    rsr_wavelengths     wavelengths (nm) of the relative spectral response of the band
    rsr_values          relative spectral response at `rsr_wavelengths`

    normalise="weighted" (default, Methods 2.2):  sum(Rrs * RSR) / sum(RSR)
    normalise="mean":                             sum(Rrs * RSR) / number of wavelengths (NOT a weighted mean; the band
                                                  value is then scaled by sum(RSR)/N, which differs per band and sensor)
    """
    spectra = np.asarray(spectra, float)
    spectra_wavelengths = np.asarray(spectra_wavelengths)
    rsr_wavelengths = np.asarray(rsr_wavelengths)
    rsr_values = np.asarray(rsr_values, float)
    if not np.isin(rsr_wavelengths, spectra_wavelengths).all():
        raise ValueError("The spectra do not cover the whole band")
    columns = np.searchsorted(spectra_wavelengths, rsr_wavelengths)
    weighted_sum = spectra[:, columns] @ rsr_values
    if normalise == "weighted":
        return weighted_sum / rsr_values.sum()
    if normalise == "mean":
        return weighted_sum / len(rsr_values)
    raise ValueError("normalise must be 'weighted' or 'mean'")

################################################# CYANOBACTERIA INDICES ###########################################################
###################################################################################################################################

def line_height(center, minus, plus, wl_center, wl_minus, wl_plus):
    """Spectral shape / line height (Equation 1): height of the centre band above the line between its neighbours."""
    return center - minus + (minus - plus) * (wl_center - wl_minus) / (wl_plus - wl_minus)


def calculate_indices(blue, green, orange, red, nir=None, sensor="OLCI"):
    """The six cyanobacteria indices of the paper for one sensor.

    Inputs are arrays of reflectance in one unit (rho_s or Rrs). Returns a DataFrame with the columns
    orange_red, orange_green, nir_orange, OLH, GLH, green_blue_term and EGLH.
    """
    wl = BAND_CENTRES_NM[sensor]
    blue, green, orange, red = (np.asarray(v, float) for v in (blue, green, orange, red))
    indices = pd.DataFrame(index=np.arange(len(np.atleast_1d(green))))
    indices["orange_red"] = orange / red
    indices["orange_green"] = orange / green
    indices["nir_orange"] = np.asarray(nir, float) / orange if nir is not None else np.nan
    indices["OLH"] = line_height(orange, green, red, wl["orange"], wl["green"], wl["red"])
    indices["GLH"] = line_height(green, blue, orange, wl["green"], wl["blue"], wl["orange"])
    indices["green_blue_term"] = (green - blue) / (green + blue)          # (G - B) / (G + B)
    indices["EGLH"] = np.exp(indices["green_blue_term"]) * indices["GLH"]  # Equation 2
    return indices

################################################# PC MODELS (TABLE 2, EQUATIONS 3-6) ##############################################
###################################################################################################################################

# PC = f(index), calibrated with 144 western Lake Erie OLCI matchups (rho_s based indices, PC in ug/L)
PC_MODELS = {
    # index:          (form,  coefficients)
    "orange_red":   ("exp",  (19.81, -0.766)),           # PC = a * exp(b * x)
    "orange_green": ("exp",  (13.596, -1.366)),
    "nir_orange":   ("lin",  (26.06, 10.95)),            # PC = a * x + b
    "OLH":          ("quad", (186204, 183.95, 3.0803)),  # PC = a * x^2 + b * x + c    (Equation 5)
    "GLH":          ("quad", (59381, -2113, 22.604)),    # Equation 6
    "EGLH":         ("quad", (18731, -545.6, 6.338)),    # Equation 4
}

# OLI to OLCI transfer, 26 September 2017 (Equation 3 and 3a-3b): index_OLCI = slope * index_OLI + intercept
OLI_TO_OLCI_TRANSFER = {"OLH": (0.9644, -0.004893), "GLH": (1.369, -0.001247), "EGLH": (1.483, -0.003022)}

# Valid domain of Equation 4
_A, _B, _C = PC_MODELS["EGLH"][1]
EGLH_VERTEX = -_B / (2 * _A)                         # 0.01456, minimum of the parabola
PC_AT_VERTEX = _C - _B ** 2 / (4 * _A)               # 2.36 ug/L
EGLH_CALIBRATION_MAX = 0.092941                    # largest EGLH of the 100 calibration matchups (PC = 117.4 ug/L)


def apply_pc_model(x, form, coefficients):
    """PC (ug/L) from index values x. form: 'exp', 'lin', 'quad' or 'loglog' (log10 PC = a + b log10 x)."""
    x = np.asarray(x, dtype=float)
    if form == "exp":
        return coefficients[0] * np.exp(coefficients[1] * x)
    if form == "lin":
        return coefficients[0] * x + coefficients[1]
    if form == "quad":
        return coefficients[0] * x ** 2 + coefficients[1] * x + coefficients[2]
    if form == "loglog":
        pc = np.full_like(x, np.nan)
        positive = x > 0
        pc[positive] = 10 ** (coefficients[0] + coefficients[1] * np.log10(x[positive]))
        return pc
    raise ValueError(f"Unknown model form: {form}")


def fit_pc_model(x, pc, form):
    """Fit one model form. Returns (coefficients, R2). The exponential model is fitted to ln(PC), as in Excel."""
    x, pc = np.asarray(x, float), np.asarray(pc, float)
    if form in ("quad", "lin"):
        coefficients = tuple(np.polyfit(x, pc, 2 if form == "quad" else 1))
        predicted = apply_pc_model(x, form, coefficients)
        r2 = 1 - np.sum((pc - predicted) ** 2) / np.sum((pc - pc.mean()) ** 2)
    elif form == "exp":
        b, ln_a = np.polyfit(x, np.log(pc), 1)
        coefficients = (np.exp(ln_a), b)
        r2 = np.corrcoef(x, np.log(pc))[0, 1] ** 2
    elif form == "loglog":
        positive = x > 0
        b, a = np.polyfit(np.log10(x[positive]), np.log10(pc[positive]), 1)
        coefficients = (a, b)
        r2 = np.corrcoef(np.log10(x[positive]), np.log10(pc[positive]))[0, 1] ** 2
    else:
        raise ValueError(f"Unknown model form: {form}")
    return coefficients, r2


def oli_to_olci(index_name, oli_values):
    """Transfer an OLI index (OLH, GLH or EGLH) to the OLCI scale (Equation 3)."""
    slope, intercept = OLI_TO_OLCI_TRANSFER[index_name]
    return slope * np.asarray(oli_values, float) + intercept


def eglh_to_pc(eglh, rhos_blue=None, rhos_green=None, rhos_orange=None, coefficients=(_A, _B, _C)):
    """Equation 4 with the valid-domain rules. Returns (PC in ug/L, flag).

    flag = 'invalid'           rho_s <= 0 in the blue, green or orange band (PC is NaN)
           'below sensitivity' EGLH below the vertex of the parabola; PC is reported as the lower limit (2.36 ug/L)
           'valid'             vertex <= EGLH <= largest calibration EGLH
           'extrapolated'      EGLH above the calibration range (computed, but flagged)
    """
    eglh = np.asarray(eglh, float)
    pc = apply_pc_model(eglh, "quad", coefficients)
    flag = np.full(eglh.shape, "valid", dtype=object)
    flag[eglh > EGLH_CALIBRATION_MAX] = "extrapolated"
    below_vertex = eglh < EGLH_VERTEX
    pc[below_vertex] = PC_AT_VERTEX
    flag[below_vertex] = "below sensitivity"
    if rhos_blue is not None:
        invalid = (np.asarray(rhos_blue) <= 0) | (np.asarray(rhos_green) <= 0) | (np.asarray(rhos_orange) <= 0)
        pc[invalid] = np.nan
        flag[invalid] = "invalid"
    return pc, flag

################################################# ERROR METRICS ###################################################################
###################################################################################################################################

def get_validation_metrics(observed, predicted):
    """Dictionary of validation metrics. MdSA and SSPB follow Morley et al. (2018) and Seegers et al. (2018).

    bias = mean(predicted - observed); NRMSE_range = RMSE / (max - min of observed);
    MdSA = 100 (exp(median |ln(P/O)|) - 1);  SSPB = 100 sign(median ln(P/O)) (exp(|median ln(P/O)|) - 1).
    Log metrics use only the pairs with observed and predicted values > 0.
    """
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    obs, pred = observed[finite], predicted[finite]
    n = len(obs)
    if n < 3:
        return {"n": n}
    difference = pred - obs
    rmse = np.sqrt(np.mean(difference ** 2))
    slope, intercept, r_value, _, _ = stats.linregress(obs, pred)
    positive = (obs > 0) & (pred > 0)
    log_ratio = np.log(pred[positive] / obs[positive])
    return {
        "n": n, "R2": r_value ** 2, "slope": slope, "intercept": intercept,
        "bias": difference.mean(), "MAE": np.abs(difference).mean(), "RMSE": rmse,
        "NRMSE_range_%": 100 * rmse / (obs.max() - obs.min()), "NRMSE_mean_%": 100 * rmse / obs.mean(),
        "MdSA_%": 100 * (np.exp(np.median(np.abs(log_ratio))) - 1),
        "SSPB_%": 100 * np.sign(np.median(log_ratio)) * (np.exp(np.abs(np.median(log_ratio))) - 1),
        "RMSLE": np.sqrt(np.mean((np.log10(pred[positive]) - np.log10(obs[positive])) ** 2)),
        "Spearman_rho": stats.spearmanr(obs, pred).correlation, "n_log": int(positive.sum()),
    }


def bootstrap_confidence_intervals(observed, predicted, metrics=("RMSE", "MAE", "bias", "R2", "MdSA_%", "SSPB_%"),
                                   n_boot=N_BOOTSTRAP, seed=RANDOM_SEED, groups=None):
    """95% percentile bootstrap intervals of validation metrics.

    If `groups` is given (for example image dates), whole groups are resampled together (cluster bootstrap), because
    matchups from the same image are not independent.
    """
    rng = np.random.default_rng(seed)
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    all_rows = np.arange(len(observed))
    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        rows_of_group = {g: all_rows[groups == g] for g in unique_groups}
    results = {m: [] for m in metrics}
    for _ in range(n_boot):
        if groups is None:
            rows = rng.choice(all_rows, size=len(all_rows), replace=True)
        else:
            chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            rows = np.concatenate([rows_of_group[g] for g in chosen])
        resampled = get_validation_metrics(observed[rows], predicted[rows])
        for m in metrics:
            results[m].append(resampled.get(m, np.nan))
    return {m: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for m, v in results.items()}


def format_with_interval(value, interval, digits=2):
    """'value (low to high)' for tables."""
    return f"{value:.{digits}f} ({interval[0]:.{digits}f} to {interval[1]:.{digits}f})"


def get_log_metrics(observed, predicted):
    """Log-space metrics used for the multi-lake recalibration (PC spans three orders of magnitude)."""
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    log_ratio = np.log(predicted / observed)
    return {"n": len(observed), "MdSA_%": 100 * (np.exp(np.median(np.abs(log_ratio))) - 1),
            "SSPB_%": 100 * np.sign(np.median(log_ratio)) * (np.exp(np.abs(np.median(log_ratio))) - 1),
            "Spearman_rho": stats.spearmanr(observed, predicted).correlation,
            "R2_log": np.corrcoef(np.log10(observed), np.log10(predicted))[0, 1] ** 2}

################################################# BLOCKED CROSS-VALIDATION ########################################################
###################################################################################################################################

def blocked_cross_validation(df, index_name, form, block_column, pc_column="PC"):
    """Leave-one-block-out predictions: the model is refitted without each block and then applied to that block.

    block_column: 'image_date' (LODO), 'year' (LOYO) or 'station' (LOSO). Returns the rows used, with a `pred` column.
    """
    data = df[np.isfinite(df[index_name])].copy()
    if index_name in ("orange_red", "orange_green", "nir_orange"):
        data = data[data[index_name] > 0]               # ratios of negative reflectance are meaningless
    data["pred"] = np.nan
    for block in data[block_column].unique():
        training = data[data[block_column] != block]
        held_out = data[data[block_column] == block]
        coefficients, _ = fit_pc_model(training[index_name], training[pc_column], form)
        data.loc[held_out.index, "pred"] = apply_pc_model(held_out[index_name], form, coefficients)
    return data

################################################# EMBARGOED 2024-2025 DATA ########################################################
###################################################################################################################################

EMBARGOED_FILES = {
    "stations":  "new_2024_2025_stations_PC.csv",                  # station, lake, year, PC
    "bands":     "new_2024_2025_insitu_sensor_bands_Rrs.csv",      # station, sensor, blue, green, orange, red
    "matchups":  "new_2024_2025_olci_matchups_Rrs.csv",            # OLCI matchups: station ... PC, Rrs_490 ... Rrs_865
}


class StopExecution(Exception):
    """Stops 'Run All' without a traceback when the embargoed files are not in data/embargoed/."""
    def _render_traceback_(self):
        return [str(self)]


def load_embargoed_data(folder="../data/embargoed"):
    """Read the 2024-2025 field data (embargoed, not part of the repository).

    Returns a dictionary of DataFrames. If a file is missing, raises StopExecution with an explanation, so that the
    notebook stops here and everything above this cell still shows.
    """
    import os
    missing = [name for name in EMBARGOED_FILES.values() if not os.path.exists(os.path.join(folder, name))]
    if missing:
        raise StopExecution(
            "The 2024-2025 field data are under a NASA/UGA embargo and are not included in this repository.\n"
            f"Missing in {folder}/: {', '.join(missing)}\n"
            "Place the files there (see data/embargoed/README.md for the columns) and run this notebook again. "
            "Until the data are released (planned: SeaBASS), see the paper and its supplementary tables for the results.")
    data = {key: pd.read_csv(os.path.join(folder, name)) for key, name in EMBARGOED_FILES.items()}
    data["matchups"]["overpass_utc"] = pd.to_datetime(data["matchups"]["overpass_utc"])
    data["matchups"]["sampling_utc"] = pd.to_datetime(data["matchups"]["sampling_utc"])
    return data

################################################# PLOTTING ########################################################################
###################################################################################################################################

def set_plot_fontsize(fontsize):
    plt.rc("font", size=fontsize)
    plt.rc("axes", titlesize=fontsize)
    plt.rc("axes", labelsize=fontsize)
    plt.rc("xtick", labelsize=fontsize)
    plt.rc("ytick", labelsize=fontsize)
    plt.rc("legend", fontsize=fontsize)


def format_number(value):
    """Short number format for the statistics boxes in the figures."""
    if value == 0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 100:
        return f"{value:.2e}"
    return f"{value:.3f}"


def plot_comparison_panel(ax, x_by_group, y_by_group, group_colours, panel_letter, xlabel, ylabel, title,
                          clip_percentile=None, stats_corner="upper left", show_nrmse=False):
    """Scatter of y against x with 1:1 line, least-squares line and statistics box (Figures 3 to 6).

    x_by_group / y_by_group: dictionaries {group name: array}. The statistics use all groups together.
    clip_percentile: axis limits are set from this upper percentile (hides a few extreme ratio values).
    """
    for group, x in x_by_group.items():
        ax.scatter(x, y_by_group[group], s=55, alpha=0.65, facecolors="none", edgecolors=group_colours[group],
                   linewidths=1.1, rasterized=True, zorder=2)
    all_x = np.concatenate(list(x_by_group.values()))
    all_y = np.concatenate([y_by_group[g] for g in x_by_group])

    if clip_percentile is not None:
        low = min(np.percentile(all_x, 0.5), np.percentile(all_y, 0.5))
        high = max(np.percentile(all_x, clip_percentile), np.percentile(all_y, clip_percentile))
    else:
        low, high = min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())
    padding = (high - low) * 0.06
    low, high = low - padding, high + padding

    slope, intercept, r_value, _, _ = stats.linregress(all_x, all_y)
    line_x = np.array([low, high])
    ax.plot(line_x, line_x, color="black", ls="--", lw=1.5, alpha=0.85, zorder=1)                  # 1:1 line
    ax.plot(line_x, slope * line_x + intercept, color="#C0392B", lw=1.3, alpha=0.75, zorder=3)    # best fit

    rmse = np.sqrt(np.mean((all_y - all_x) ** 2))
    bias = (all_y - all_x).mean()
    sign = "+" if intercept >= 0 else "−"
    text = (f"y = {slope:.3f}x {sign} {format_number(abs(intercept))}\nR² = {r_value ** 2:.4f}\n"
            f"RMSE = {format_number(rmse)}\nbias = {format_number(bias)}")
    if show_nrmse:
        text += f"\nNRMSE = {100 * rmse / (all_x.max() - all_x.min()):.2f}%"
    horizontal, vertical = (0.04, 0.96) if stats_corner == "upper left" else (0.96, 0.04)
    ax.text(horizontal, vertical, text, transform=ax.transAxes, fontsize=8.5, style="italic",
            ha="left" if stats_corner == "upper left" else "right",
            va="top" if stats_corner == "upper left" else "bottom",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.55", alpha=0.95, lw=0.7), zorder=5)

    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=5)
    letter_x, letter_y = (0.94, 0.06) if stats_corner == "upper left" else (0.06, 0.94)
    ax.text(letter_x, letter_y, panel_letter, transform=ax.transAxes, fontsize=15, fontweight="bold",
            ha="right" if stats_corner == "upper left" else "left",
            va="bottom" if stats_corner == "upper left" else "top", zorder=5)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.locator_params(axis="both", nbins=5)
    ax.tick_params(axis="both", labelsize=9)
    return {"slope": slope, "intercept": intercept, "R2": r_value ** 2, "RMSE": rmse, "bias": bias, "n": len(all_x)}


def get_validation_plot(x, y, ax=None, metrics=("n", "R2", "slope", "bias", "MAE", "RMSE", "MdSA_%"),
                        xlabel="Observed", ylabel="Predicted", title="", log_axes=False, colour="black",
                        lower_limit=None, upper_limit=None, units="", category=None, category_colours=None, marker_size=25):
    """Observed (x) against predicted (y) with a 1:1 line and a box of validation metrics.

    category / category_colours: optional array of group labels and a {label: colour} dictionary.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x, y = np.asarray(x, float), np.asarray(y, float)
    if category is None:
        ax.scatter(x, y, s=marker_size, color=colour, edgecolor="none")
    else:
        category = np.asarray(category)
        for label in pd.unique(category):
            ax.scatter(x[category == label], y[category == label], s=marker_size, label=label,
                       color=(category_colours or {}).get(label), edgecolor="k", lw=0.3)
    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
    low = lower_limit if lower_limit is not None else np.nanmin([x, y])
    high = upper_limit if upper_limit is not None else np.nanmax([x, y])
    ax.plot([low, high], [low, high], color="0.4", lw=1, ls="--", zorder=0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    results = get_validation_metrics(x, y)
    label_formats = {"n": "n = {:.0f}", "R2": "R$^2$ = {:.2f}", "slope": "slope = {:.2f}",
                     "bias": "bias = {:.1f} " + units, "MAE": "MAE = {:.1f} " + units, "RMSE": "RMSE = {:.1f} " + units,
                     "MdSA_%": "MdSA = {:.0f}%", "SSPB_%": "SSPB = {:.0f}%", "NRMSE_range_%": "NRMSE = {:.1f}%",
                     "Spearman_rho": "Spearman ρ = {:.2f}"}
    text = "\n".join(label_formats[m].format(results[m]) for m in metrics if m in results)
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    return ax, results


def save_table(df, name, folder="../results/tables", index=False):
    """Save a table as CSV and return it."""
    import os
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, name + ".csv"), index=index)
    return df

################################################# INDEX - PC RELATIONSHIPS (TABLE S3) #############################################
###################################################################################################################################

def best_fit_summary(index_values, pc):
    """Fit linear, exponential (ln PC) and quadratic models of PC against an index; report the one with the highest R2.

    Also returns the Spearman rank correlation, which does not depend on the model form and is not dominated by the few
    very high PC values.
    """
    x, y = np.asarray(index_values, float), np.asarray(pc, float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[ok], y[ok]
    fits = {}
    linear = stats.linregress(x, y)
    fits["linear"] = (linear.rvalue ** 2, linear.pvalue, f"y = {linear.slope:.4g}x {linear.intercept:+.4g}")
    exponential = stats.linregress(x, np.log(y))
    fits["exponential"] = (exponential.rvalue ** 2, exponential.pvalue,
                           f"y = {np.exp(exponential.intercept):.4g}e^({exponential.slope:.4g}x)")
    quadratic = sm.OLS(y, sm.add_constant(np.column_stack([x, x ** 2]))).fit()
    fits["quadratic"] = (quadratic.rsquared, quadratic.f_pvalue,
                         f"y = {quadratic.params[2]:.4g}x^2 {quadratic.params[1]:+.4g}x {quadratic.params[0]:+.4g}")
    best = max(fits, key=lambda form: fits[form][0])
    rho = stats.spearmanr(x, y)
    return {"n": len(x), "best form": best, "equation": fits[best][2], "R2": fits[best][0], "p": fits[best][1],
            "Spearman rho": rho.correlation, "Spearman p": rho.pvalue}
