import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from pyqdm.core import models
from pyqdm.utils import double_norm

FREQ_LABEL = "f [GHz]"
CONTRAST_LABEL = "c [%]"


def plot_light_img(ax, data, img=None, **plt_props):
    img = update_img(
        ax,
        img,
        data,
        cmap="bone",
        interpolation="none",
        origin="lower",
        aspect="equal",
        zorder=0,
        **plt_props,
    )
    return img


def plot_fluorescence(ax, data, img=None, **plt_props):
    img = update_img(
        ax,
        img,
        data,
        cmap="inferno",
        interpolation="none",
        origin="lower",
        aspect="equal",
        zorder=0,
        **plt_props,
    )
    return img


def plot_laser_img(ax, data, img=None, **plt_props):
    img = update_img(
        ax,
        img,
        data,
        cmap="magma",
        interpolation="none",
        origin="lower",
        aspect="equal",
        zorder=0,
        **plt_props,
    )
    return img


def update_line(ax, x, y=None, line=None, **plt_props):
    if y is None:
        return
    if line is None:
        (line,) = ax.plot(x, y, **plt_props)
    else:
        if all(y == line.get_ydata()):
            return line
        line.set_ydata(y)
        # ax.draw_artist(line)
    return line


def update_marker(ax, x, y, line=None, **plt_props):
    if line is None:
        (line,) = ax.plot(x, y, **plt_props)
    else:
        line.set_data(x, y)
        ax.draw_artist(line)
    return line


def plot_data(ax, data, img=None, **plt_props):

    norm = get_color_norm(data.min(), data.max())
    # plt_props["cmap"] = ""
    plt_props["norm"] = norm
    img = update_img(ax, img, data, **plt_props)
    return img


def get_vmin_vmax(img, percentile, use_percentile):
    if img is None:
        return 0, 1

    if percentile and use_percentile:
        vmin, vmax = np.percentile(
            img.get_array(),
            [(100 - percentile) / 2, 100 - (100 - percentile) / 2],
        )
    else:
        vmin, vmax = (
            img.get_array().min(),
            img.get_array().max(),
        )
    return vmin, vmax


def get_color_norm(vmin, vmax):
    if vmin < 0 < vmax:
        return colors.CenteredNorm(halfrange=vmax, vcenter=0)
    else:
        return colors.Normalize(vmin=vmin, vmax=vmax)


def plot_overlay(ax, data, img=None, normtype="simple", **plt_props):
    if normtype == "simple":
        plt_props["alpha"] = double_norm(data)
    else:
        raise NotImplementedError(f"Normalization type {normtype} not implemented.")
    img = update_img(ax, img, data, **plt_props)
    return img


def plot_outlier(ax, data, img=None, **plt_props):
    plt_props["cmap"] = ""
    plt_props["alpha"] = data.astype(int)
    img = update_img(ax, img, data, **plt_props)
    return img


def update_clim(img, vmin, vmax):
    norm = get_color_norm(vmin, vmax)
    img.set(norm=norm)


def update_cbar(img, cax, vmin, vmax, original_cax_locator):
    extent = detect_extent(vmin=vmin, vmax=vmax, mn=img.get_array().min(), mx=img.get_array().max())

    label = cax.get_ylabel()
    cax.clear()
    cax.set_axes_locator(original_cax_locator)
    plt.colorbar(img, cax=cax, extend=extent, label=label)


def detect_extent(vmin, vmax, mn, mx):
    if vmin == mn and vmax == mx:
        return "neither"
    elif vmin > mn and vmax < mx:
        return "both"
    elif vmin > mn:
        return "min"
    else:
        return "max"


def update_img(ax, img, data, **plt_props):
    data_dimensions = plt_props.pop("data_dimensions", data.shape)
    plt_props["extent"] = [0, data_dimensions[1], 0, data_dimensions[0]]
    if img is None:
        img = ax.imshow(data, **plt_props)
    else:
        img.set_data(data)
    return img


def toggle_img(img=None):
    if img is None:
        return
    else:
        img.set_visibility(~img.visibility)


def check_fit_pixel(qdm_obj, idx):
    # noinspection PyTypeChecker
    f, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=True)
    polarities = ["+", "-"]
    model = [None, models.esrsingle, models.esr15n, models.esr14n][qdm_obj.diamond_type]
    print(f"IDX: {idx}, Model: {model.__name__}")
    lst = ["pol/side"] + qdm_obj.fit.fitting_parameter + ["chi2"]
    header = " ".join([f"{i:>8s}" for i in lst])
    print(f"{header}")
    print("-" * 100)

    for p, f in itertools.product(range(qdm_obj.odmr.n_pol), range(qdm_obj.odmr.n_frange)):
        f_new = np.linspace(min(qdm_obj.odmr.f_ghz[f]), max(qdm_obj.odmr.f_ghz[f]), 200)

        m_initial = model(parameter=qdm_obj.initial_parameter[p, f, [idx]], x=f_new)
        m_fit = model(parameter=qdm_obj.fit.fitting_parameter[p, f, [idx]], x=f_new)

        ax[f].plot(
            qdm_obj.odmr.f_ghz[f],
            qdm_obj.odmr.data[p, f, [idx]][0],
            "k",
            marker=["o", "^"][p],
            markersize=5,
            mfc="w",
            label=f"data: {polarities[p]}",
            ls="",
        )
        (l,) = ax[f].plot(f_new, m_initial[0], label="initial guess", alpha=0.5, ls=":")
        ax[f].plot(f_new, m_fit[0], color=l.get_color(), label="fit")
        ax[f].legend(
            ncol=2,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            borderaxespad=0.0,
        )

        line = " ".join([f"{v:>8.5f}" for v in qdm_obj.fit.fitting_parameter[p, f, idx]])
        line += f" {qdm_obj.fit._chi_squares[p, f, idx]:>8.2e}"
        print(f'{["+", "-"][p]},{["<", ">"][p]}:     {line}')

    for a in ax.flat:
        a.set(xlabel=FREQ_LABEL, ylabel="ODMR contrast [a.u.]")
    return f, ax


def plot_fit_params(qdm_obj, param, save=False):
    data = qdm_obj.get_param(param)

    if param == "contrast":
        data = data.mean(axis=2)
    if "contrast" in param:
        data *= 100
    if param == "width":
        data *= 1000

    labels = {
        "center": FREQ_LABEL,
        "resonance": FREQ_LABEL,
        "width": "f [MHz]",
        "contrast": "mean(c) [%]",
        "contrast_0": CONTRAST_LABEL,
        "contrast_1": CONTRAST_LABEL,
        "contrast_2": CONTRAST_LABEL,
        "chi2": "chi$^2$",
    }

    # noinspection PyTypeChecker
    f, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    f.suptitle(f"{param}")

    # determine min and max of the plot
    vminl = np.min(np.sort(data[:, 0].flat)[50:-50])
    vmaxl = np.max(np.sort(data[:, 0].flat)[50:-50])
    vminr = np.min(np.sort(data[:, 1].flat)[50:-50])
    vmaxr = np.max(np.sort(data[:, 1].flat)[50:-50])

    # positive field direction
    ax[0, 0].set_title(r"B$^+_\mathrm{lf}$")
    ax[0, 0].imshow(data[0, 0], origin="lower", vmin=vminl, vmax=vmaxl)
    ax[0, 1].set_title(r"B$^+_\mathrm{hf}$")
    ax[0, 1].imshow(data[0, 1], origin="lower", vmin=vminr, vmax=vmaxr)

    # negative field direction
    ax[1, 0].set_title(r"B$^-_\mathrm{lf}$")
    c = ax[1, 0].imshow(data[1, 0], origin="lower", vmin=vminl, vmax=vmaxl)
    cb = plt.colorbar(c, ax=ax[:, 0], shrink=0.9)
    cb.ax.set_ylabel(labels[param])

    ax[1, 1].set_title(r"B$^-_\mathrm{hf}$")
    c = ax[1, 1].imshow(data[1, 1], origin="lower", vmin=vminr, vmax=vmaxr)
    cb = plt.colorbar(c, ax=ax[:, 1], shrink=0.9)
    cb.ax.set_ylabel(labels[param])

    for a in ax.flat:
        a.set(xlabel="px", ylabel="px")

    if save:
        f.savefig(save)


# def plot_fluorescence(qdm_obj, f_idx):
#     # noinspection PyTypeChecker
#     f, ax = plt.subplots(2, 2, figsize=(9, 5), sharex=True, sharey=True)
#     f.suptitle(
#         f"Fluorescence of frequency " f"({qdm_obj.odmr.f_ghz[0, f_idx]:.5f};" f"{qdm_obj.odmr.f_ghz[1, f_idx]:.5f}) GHz"
#     )
#
#     vmin = np.min(qdm_obj.odmr.data)
#     vmax = 1
#
#     d = qdm_obj.odmr["r"]
#
#     # low frequency
#     ax[0, 0].imshow(d[0, 0, :, :, f_idx], origin="lower", vmin=vmin, vmax=vmax)
#     ax[1, 0].imshow(d[1, 0, :, :, f_idx], origin="lower", vmin=vmin, vmax=vmax)
#     # high frequency
#     ax[0, 1].imshow(d[0, 1, :, :, f_idx], origin="lower", vmin=vmin, vmax=vmax)
#     c = ax[1, 1].imshow(d[1, 1, :, :, f_idx], origin="lower", vmin=vmin, vmax=vmax)
#
#     cb = f.colorbar(c, ax=ax[:, 1], shrink=0.97)
#     cb.ax.set_ylabel("fluorescence intensity")
#
#     pol = ["+", "-"]
#     side = ["l", "h"]
#     for i, j in itertools.product(range(qdm_obj.odmr.n_pol), range(qdm_obj.odmr.n_frange)):
#         a = ax[i, j]
#         a.set_title(rf"B$^{pol[i]}_\mathrm{{{side[j]}f}}$")
#         a.text(
#             0.0,
#             1,
#             f"{qdm_obj.odmr.f_ghz[j, f_idx]:.5f} GHz",
#             va="bottom",
#             ha="left",
#             transform=a.transAxes,
#             color="k",
#             zorder=100,
#         )
#     plt.show()
