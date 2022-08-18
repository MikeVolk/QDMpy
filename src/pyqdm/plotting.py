import itertools
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from pyqdm.core import models

FREQ_LABEL = 'f [GHz]'
CONTRAST_LABEL = 'c [%]'


def check_fit_pixel(qdm_obj, idx):
    f, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=True)
    polarities = ['+', '-']
    model = [None, models.ESRSINGLE, models.ESR15N, models.ESR14N][qdm_obj._diamond_type]
    print(f'IDX: {idx}, Model: {model.__name__}')
    lst = ['pol/side'] + qdm_obj._fitting_params + ['chi2']
    header = ' '.join([f'{i:>8s}' for i in lst])
    print(f'{header}')
    print('-' * 100)

    for p, f in itertools.product(range(qdm_obj.ODMRobj.n_pol), range(qdm_obj.ODMRobj.n_frange)):
        f_new = np.linspace(min(qdm_obj.ODMRobj.f_GHz[f]), max(qdm_obj.ODMRobj.f_GHz[f]), 200)

        m_initial = model(parameter=qdm_obj.initial_guess[p, f, [idx]], x=f_new)
        m_fit = model(parameter=qdm_obj._fitted_parameter[p, f, [idx]], x=f_new)

        ax[f].plot(qdm_obj.ODMRobj.f_GHz[f], qdm_obj.ODMRobj.data[p, f, [idx]][0], 'k', marker=['o', '^'][p],
                   markersize=5,
                   mfc='w',
                   label=f'data: {polarities[p]}', ls='')
        l, = ax[f].plot(f_new, m_initial[0], label='initial guess', alpha=0.5, ls=':')
        ax[f].plot(f_new, m_fit[0], color=l.get_color(), label='fit')
        ax[f].legend(ncol=2, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode='expand', borderaxespad=0.)

        line = ' '.join([f'{v:>8.5f}' for v in qdm_obj._fitted_parameter[p, f, idx]])
        line += f' {qdm_obj._chi_squares[p, f, idx]:>8.2e}'
        print(f'{["+", "-"][p]},{["<", ">"][p]}:     {line}')

    for a in ax.flat:
        a.set(xlabel=FREQ_LABEL, ylabel='ODMR contrast [a.u.]')
    return f, ax


def plot_fit_params(qdm_obj, param, save=False):
    data = qdm_obj.get_param(param)

    if param == 'contrast':
        data = data.mean(axis=2)
    if 'contrast' in param:
        data *= 100
    if param == 'width':
        data *= 1000

    labels = {'center': FREQ_LABEL,
              'resonance': FREQ_LABEL,
              'width': 'f [MHz]',
              'contrast': 'mean(c) [%]',
              'contrast_0': CONTRAST_LABEL,
              'contrast_1': CONTRAST_LABEL,
              'contrast_2': CONTRAST_LABEL,
              'chi2': 'chi$^2$'}

    f, ax = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    f.suptitle(f'{param}')

    # determine min and max of the plot
    vminl = np.min(np.sort(data[:, 0].flat)[50:-50])
    vmaxl = np.max(np.sort(data[:, 0].flat)[50:-50])
    vminr = np.min(np.sort(data[:, 1].flat)[50:-50])
    vmaxr = np.max(np.sort(data[:, 1].flat)[50:-50])

    # positive field direction
    ax[0, 0].set_title('B$^+_\mathrm{lf}$')
    ax[0, 0].imshow(data[0, 0], origin='lower', vmin=vminl, vmax=vmaxl)
    ax[0, 1].set_title('B$^+_\mathrm{hf}$')
    ax[0, 1].imshow(data[0, 1], origin='lower', vmin=vminr, vmax=vmaxr)

    # negative field direction
    ax[1, 0].set_title('B$^-_\mathrm{lf}$')
    c = ax[1, 0].imshow(data[1, 0], origin='lower', vmin=vminl, vmax=vmaxl)
    cb = plt.colorbar(c, ax=ax[:, 0], shrink=0.9)
    cb.ax.set_ylabel(labels[param])

    ax[1, 1].set_title('B$^-_\mathrm{hf}$')
    c = ax[1, 1].imshow(data[1, 1], origin='lower', vmin=vminr, vmax=vmaxr)
    cb = plt.colorbar(c, ax=ax[:, 1], shrink=0.9)
    cb.ax.set_ylabel(labels[param])

    for a in ax.flat:
        a.set(xlabel='px', ylabel='px')

    if save:
        f.savefig(save)


def plot_fluorescence(qdm_obj, f_idx):
    f, ax = plt.subplots(2, 2, figsize=(9, 5), sharex=True, sharey=True)
    f.suptitle(f'Fluorescence of frequency '
               f'({qdm_obj.ODMRobj.f_GHz[0, f_idx]:.5f};'
               f'{qdm_obj.ODMRobj.f_GHz[1, f_idx]:.5f}) GHz')

    vmin = np.min(qdm_obj.ODMRobj.data)
    vmax = 1

    d = qdm_obj.ODMRobj['r']

    # low frequency
    ax[0, 0].imshow(d[0, 0, :, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)
    ax[1, 0].imshow(d[1, 0, :, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)
    # high frequency
    ax[0, 1].imshow(d[0, 1, :, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)
    c = ax[1, 1].imshow(d[1, 1, :, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)

    cb = f.colorbar(c, ax=ax[:, 1], shrink=0.97)
    cb.ax.set_ylabel('fluorescence intensity')

    pol = ['+', '-']
    side = ['l', 'h']
    for i, j in itertools.product(range(qdm_obj.ODMRobj.n_pol), range(qdm_obj.ODMRobj.n_frange)):
        a = ax[i, j]
        a.set_title('B$^%s_\mathrm{%sf}$' % (pol[i], side[j]))
        a.text(0.0, 1, f'{qdm_obj.ODMRobj.f_GHz[j, f_idx]:.5f} GHz',
               va='bottom', ha='left',
               transform=a.transAxes,
               color='k', zorder=100)
    plt.show()
