import itertools
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from pyqdm.core import models


def check_fit_pixel(QDMobj, idx):
    f, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=True)
    polarities = ['+', '-']
    model = [None, models.ESRSINGLE, models.ESR15N, models.ESR14N][QDMobj._diamond_type]
    print(f'IDX: {idx}, Model: {model.__name__}')
    lst = ['pol/side'] + QDMobj._fitting_params + ['chi2']
    header = ' '.join([f'{i:>8s}' for i in lst])
    print(f'{header}')
    print('-' * 100)

    for p, f in itertools.product(range(QDMobj.ODMRobj.n_pol), range(QDMobj.ODMRobj.n_frange)):
        f_new = np.linspace(min(QDMobj.ODMRobj.f_GHz[f]), max(QDMobj.ODMRobj.f_GHz[f]), 200)

        m_initial = model(parameter=QDMobj.initial_guess[p, f, [idx]], x=f_new)
        m_fit = model(parameter=QDMobj._fitted_parameter[p, f, [idx]], x=f_new)
        m_fit_ = model(parameter=QDMobj._fitted_parameter[p, f, [idx]], x=QDMobj.ODMRobj.f_GHz[f])

        ax[f].plot(QDMobj.ODMRobj.f_GHz[f], QDMobj.ODMRobj.data[p, f, [idx]][0], 'k', marker=['o', '^'][p], markersize=5,
                   mfc='w',
                   label=f'data: {polarities[p]}', ls='')
        l, = ax[f].plot(f_new, m_initial[0], label='initial guess', alpha=0.5, ls=':')
        ax[f].plot(f_new, m_fit[0], color=l.get_color(), label='fit')
        ax[f].legend(ncol=2, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', mode='expand', borderaxespad=0.)


        line = ' '.join([f'{v:>8.5f}' for v in QDMobj._fitted_parameter[p, f, idx]])
        line += f' {QDMobj._chi_squares[p, f, idx]:>8.2e}'
        print(f'{["+", "-"][p]},{["<", ">"][p]}:     {line}')

    for a in ax.flat:
        a.set(xlabel='f [GHz]', ylabel='ODMR contrast [a.u.]')
    return f, ax


def plot_fit_params(QDMobj, param, save=False):
    data = QDMobj.get_param(param)

    if param == 'contrast':
        data = data.mean(axis=2)
    if 'contrast' in param:
        data *= 100
    if param == 'width':
        data *= 1000

    labels = {'center': 'f [GHz]',
              'resonance': 'f [GHz]',
              'width': 'f [MHz]',
              'contrast': 'mean(c) [%]',
              'contrast_0': 'c [%]',
              'contrast_1': 'c [%]',
              'contrast_2': 'c [%]',
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


def plot(model):
    print(
        f'parameters: center: {p[0]: .3f}, width: {p[1]: .3f}, \n\
            contrast_0: {p[2]: .3f}, contrast_1: {p[3]: .3f}, contrast_2: {p[4]: .3f}, \n\
            offset: {p[5]: .3f}')
    plt.plot(x, model, 'r')
    plt.plot(x, 1 - dip1, 'g')
    plt.plot(x, 1 - dip2, 'b')
    plt.plot(x, 1 - dip3, 'y')
    plt.show()

    if debug:
        print(
            f'parameters: center: {p[0]: .3f}, width: {p[1]: .3f} \n\
                contrast_0: {p[2]: .3f}, contrast_1: {p[3]: .3f} \n\
                offset: {p[4]: .3f}')
        plt.plot(x, model, 'r')
        plt.plot(x, 1 - dip1, 'g')
        plt.plot(x, 1 - dip2, 'b')
        plt.show()

    if debug:
        print(
            f'parameters: center: {p[0]: .3f}, width: {p[1]: .3f} \n\
                contrast_0: {p[2]: .3f} \n\
                offset: {p[3]: .3f}')
        plt.plot(x, model, 'r')
        plt.plot(x, 1 - dip1, 'g')
        plt.show()


def plot_fluorescence(QDMobj, f_idx):
    f, ax = plt.subplots(2, 2, figsize=(9, 5), sharex=True, sharey=True)
    f.suptitle(f'Fluorescence of frequency '
               f'({QDMobj.ODMRobj.f_GHz[0, f_idx]:.5f};'
               f'{QDMobj.ODMRobj.f_GHz[1, f_idx]:.5f}) GHz')

    vmin = np.min(QDMobj.ODMRobj.data)
    vmax = 1

    d = QDMobj.ODMRobj['r']

    # low frequency
    ax[0, 0].imshow(d[0, 0, :, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)
    ax[1, 0].imshow(d[1, 0,:, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)
    # high frequency
    ax[0, 1].imshow(d[0, 1,:, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)
    c = ax[1, 1].imshow(d[1, 1,:, :, f_idx], origin='lower', vmin=vmin, vmax=vmax)

    cb = f.colorbar(c, ax=ax[:, 1], shrink=0.97)
    cb.ax.set_ylabel('fluorescence intensity')

    pol = ['+', '-']
    side = ['l', 'h']
    for i, j in itertools.product(range(QDMobj.ODMRobj.n_pol), range(QDMobj.ODMRobj.n_frange)):
        a = ax[i, j]
        a.set_title('B$^%s_\mathrm{%sf}$' % (pol[i], side[j]))
        a.text(0.0, 1, f'{QDMobj.ODMRobj.f_GHz[j, f_idx]:.5f} GHz',
               va='bottom', ha='left',
               transform=a.transAxes,
               # bbox=dict(facecolor='w', alpha=0.5, edgecolor='none', pad=0),
               color='k', zorder=100)
    plt.show()


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
