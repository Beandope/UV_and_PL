import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import math
import os
from scipy import optimize
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator
from Figure_Size import set_size, fig_fmt

filename_in = '/Users/yulongzheng/Dropbox (GaTech)/P3HT_IL_BLENDS/UV/Aggregation.xlsx'
# filepath_in = os.getcwd()
# filename = UV_a
pd.read_excel(filename_in, sheet_name='6 wt%', header=None)
data = pd.read_excel(filename_in, sheet_name='6 wt%', header=None)
nrow, ncol = np.shape(data)[0], np.shape(data)[1]

## Functions defined for FC

h_bar = 1.0
n = 4
HR = 1.1


## functions defined for the fit--------------------------------

def Gau(x, sig):  # sig is squared already
    val_G = np.exp(-(x) ** 2 / (2 * sig ** 2))
    return val_G


def Poss(m):
    sum0 = 0
    for i in range(n):
        if m != i:
            sum0 += HR ** i / (math.factorial(i) * (i - m))
            # print(i)
    return sum0


def I(w, sig, E_0, W, E_p, cons):
    A0 = 0
    for i in range(n):
        A0 += cons * (np.exp(-HR) * HR ** i) / (math.factorial(i)) * \
              (1 - (W * np.exp(-HR) / (2 * E_p)) * Poss(i)) ** 2 * \
              Gau(w - E_0 - i * E_p, sig)
    return A0


def Gau_jac(w, sig, E_0, E_p, m):
    return -1 / sig ** 2 * (x - E_0 - m * E_p) * np.exp(-(x - E_0 - m * E_p) ** 2 / 2 / sig ** 2)


def I_mvf(w, lam_sqr, sig, E_0, W, E_p, cons):
    A0 = 0
    for i in range(n):
        A0 += cons * (np.exp(-lam_sqr) * lam_sqr ** i) / (math.factorial(i)) * \
              (1 - (W * np.exp(-lam_sqr) / (2 * E_p)) * Poss(lam_sqr, i)) ** 2 * \
              Gau(w - E_0 - i * E_p - W * (np.exp(-lam_sqr) * lam_sqr ** i) / \
                  (math.factorial(i)) * (1 - (W * np.exp(-lam_sqr) / (2 * E_p)) * Poss(lam_sqr, i)), sig)
    return A0


def I_lim(w, sig, E_0, W, E_p, cons):
    A0 = 0
    for i in range(n):
        A0 += cons * (np.exp(-1) * 1 ** i) / (math.factorial(i)) * \
              (1 - (W * np.exp(-1) / (2 * E_p)) * Poss(1, i)) ** 2 * \
              Gau(w - E_0 - i * E_p, sig)
    return A0


def II(w, p):
    return I(w, *p)


# errfunc = lambda p, x, y: function(x,p) - y

def fit_leastsq(p0, datax, datay, function):
    errfunc = lambda p, x, y: function(x, p) - y
    pfit, pcov, infodict, errmsg, success = optimize.leastsq(errfunc, p0, args=(datax, datay),
                                                             full_output=1, epsfcn=0.0001)
    #                            #      lam_sqr, sig, E_0,  W,   E_p, cons
    #                            bounds = ([1.000, 0.07, 1.45, -0.2, 0.1, 2],
    #                                      [1.0001, 0.09, 1.56, 0,  0.2, 3.5]))

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay) ** 2).sum() / (len(datay) - len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = []
    for i in range(len(pfit)):
        try:
            error.append(np.absolute(pcov[i][i]) ** 0.5)
        except:
            error.append(0.00)
    pfit_leastsq = pfit
    perr_leastsq = np.array(error)
    return pfit_leastsq, perr_leastsq


## Plotting

## index of the starting complete amorphous temp
ind_a = 5

colors = pl.cm.RdBu(np.linspace(0, 1, ncol - 2))

fig_fmt()
aa = plt.figure(dpi=200)
ax = aa.add_subplot(211)

for i in range(ncol - 2):
    ax.plot(data[0][2:], data[i + 2][2:], color=colors[i],
            label=str(int(data[i + 2][0])) + '$\degree$C')

ax.set_xlim(1.75, 3.25)
ax.set_xlabel('Photon Energy (eV)')
ax.set_ylim(-0.05, 1)
ax.set_ylabel('Absorbance')

wvl = np.array([750, 650, 550, 450, 350])
ev = 1240 / wvl
bx = ax.twiny()
bx.set_xlim(ax.get_xlim())
bx.set_xticks(ev)
bx.set_xticklabels(wvl)
bx.set_xlabel('Wavelength (nm)')
ax.legend(frameon=False, loc='upper left', fontsize=8)

ax1 = aa.add_subplot(212)
nmb_temp = ncol - 2
colors_r = pl.cm.RdBu_r(np.linspace(0, 1, ncol - 2))
aggre_list = []
iso_list = []
for i in range(nmb_temp - 1):
    aggre = data[ncol - 2 - i + 1][2:] - data[ind_a][2:]
    aggre_list.append(aggre)
    iso = np.argmin(np.array(np.abs(aggre[153: 199]))) + 153  ## index for the isosbestic point

    iso_list.append(iso)
    ax1.plot(data[0][2:], aggre, color=colors_r[i],
             label=str(int(data[ncol - 2 - i + 1][0])) + '$\degree$C')

ax1.set_xlim(1.5, 3.25)
ax1.set_xlabel('Photon Energy (eV)')
ax1.set_ylim(-0.5, 1)
ax1.set_ylabel('Diff. Absorbance')

bx1 = ax1.twiny()
bx1.set_xlim(ax.get_xlim())
bx1.set_xticks(ev)
bx1.set_xticklabels(wvl)
bx1.set_xlabel('Wavelength (nm)')
ax1.legend(frameon=False, loc='upper left', fontsize=8)

set_size(4, 8)
aa.tight_layout(pad=0.2)

## Take the integral under the curve for neg and pos parts, repectively.
ratio_list = []
ind_325 = np.argmin(np.array(np.abs(data[0][2:]-3.25))) + 2 ## index for 3.25 eV

cc_list = [] ## absorption cross-section list
for i in range(len(aggre_list)):
    agg = np.trapz(aggre_list[i][int(iso_list[i]):])
    amo = np.trapz(aggre_list[i][ind_325:int(iso_list[i])])
    cc = -agg/amo
    cc_list.append(cc)

cc_ratio = np.average(cc_list[:12])
### Fitting

lb = 250  # index of low energy  ## 470  ## new set data 360
ub = 322  # high  ## 710   ## new set data 570

A00_list = []
A01_list = []
eb_list = []
cov_list = []
eb_cov_list = []
pars_list = []

total_ls = []
osc_agg_ls = []
osc_amo_ls = []
per_agg_ls = []
colors = pl.cm.Reds(np.linspace(0.2, 1, 10))

# colors = colors[::-1]

fig_fmt()
for i in range(nmb_temp):
    ls = []
    x = data[0][2:].values.astype(float)  ## 'data' type is object
    y = data[ncol - 2 - i + 1][2:].values.astype(float)
    y = y / max(y[ind_325:])
    fig1 = plt.figure(dpi=200)
    ax2 = fig1.add_subplot(111)

    ax2.plot(x[ind_325 - 2:], y[ind_325 - 2:], marker='o', markevery=15, linewidth=1.5, label='measured')
    pi = [0.0848, 2.064, 0.118, 0.18, 2.362]  # inital parameters
    #     meas = I(x[ind_325-2:], *pi) #inital guesses
    #     ax2.plot(x[ind_325-2:], meas, 'm-', label='Intial Guess')

    #     popt1, pcov1 = fit_leastsq(pi, x[lb:ub], y[lb:ub], II)
    # # #                           full_output=1, epsfcn=0.0001,
    # #                            #          sig,   E_0,   W,   E_p, cons
    # # #                            bounds = ([ 0.07, 1.45, -0.2, 0.1, 2],
    # # #                                      [ 0.09, 1.56, 0,  0.2, 3.5]))

    popt1, pcov1 = curve_fit(I, x[lb: ub], y[lb: ub],
                             #          sig,   E_0,  W,  E_p, cons
                             bounds=([0.05, 2.0, 0.02, 0.15, 1],
                                     [0.2, 2.2, 0.2, 0.2, 3.5]))

    A00 = I(popt1[1], *popt1)
    A01 = I(popt1[1] + popt1[3], *popt1)
    A00_list.append(A00)
    A01_list.append(A01)
    eb_list.append(popt1[2] * 1000)

    #     parcovs_list.append(['%.1f' % (popt1[0]*1000) u"\u00B1" '%.2f' % (pcov1[0]*1000), '%.1f' % (popt1[1]*1000), '%.1f' % (popt1[2]*1000),
    #                               '%.1f' % (popt1[3]*1000), '%.1f' % (popt1[4])])
    nn = 0
    #     for nn in range(5):
    #         ls.append('%.1f' % (popt1[nn]*1000) + '\u00B1' + '%.2f' % (pcov1[nn]*1000))
    #         nn += 1

    for nn in range(5):
        ls.append('%.1f' % (popt1[nn] * 1000) + '\u00B1' + '%.2f' % (np.sqrt(np.diag(pcov1))[nn] * 1000))
        nn += 1
    total_ls.append(ls)
    #     pars_list.append('%.1f, %.1f, %.1f, %.1f, %.0f,' % tuple(popt1 * 1000))
    #     cov_list.append('%.2f, %.2f, %.3f, %.2f, %.1f,' % tuple(pcov1 * 1000))
    #     eb_cov_list.append(round(pcov1[2] * 1000, 3))
    eb_cov_list.append(round(np.sqrt(np.diag(pcov1))[2] * 1000, 3))
    ax2.plot(x[ind_325 - 2:], I(x[ind_325 - 2:], *popt1), 'r', linewidth=1.5,
             label='ufit: sig=%5.3f,\n'
                   ' E_0=%5.3f,W=%5.5f,\n'
                   ' E_p=%5.3f, cons=%5.3f' % tuple(popt1))
    ax2.plot(x[lb:ub], I(x[lb:ub], *popt1), 'b', linewidth=1.5, )
    ax2.set_xlabel('Photon Energy (eV)')
    ax2.set_ylabel('Norm. Absorbance')
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    text_inside = f'{int(data[ncol - 2 - i + 1][0])}' + ' $\degree$C'
    ax2.annotate(text_inside, xy=(0.04, 0.93), xycoords='axes fraction', size=8)

    ## amorphous
    I_amor = y[ind_325 - 2:] - I(x[ind_325 - 2:], *popt1)
    ax2.plot(x[ind_325 - 2:], I_amor, linestyle='--', color='k', linewidth=1.5)

    osc_agg = np.sum(I(x[ind_325 - 2:], *popt1))  ## oscillator strength for the aggregated region
    osc_agg_ls.append(osc_agg)
    osc_amo = np.sum(I_amor)  ## oscillator strength for the amorphous region
    osc_amo_ls.append(osc_amo)
    per_agg = osc_agg / cc_ratio / (osc_agg / cc_ratio + osc_amo / 1) * 100
    per_agg_ls.append(per_agg)
    ### together
    set_size(3, 2)
    fig1.show()

    print('Temperature', str(int(data[ncol - 2 - i + 1][0])) + 'C')
    print('Paras    sig,  E0,   W,    Ep,    cons')
    print('popt1', '%.1f, %.1f, %.1f, %.1f, %.0f,' % tuple(popt1 * 1000))  # unit meV
    print('pcov1', '%.2f, %.2f, %.3f, %.2f, %.1f,' % tuple(np.sqrt(np.diag(pcov1)) * 1000))
    print('agg amount %', per_agg)

# Fit_Data = I(data[0:, 0], *popt)
# Data_set = np.transpose((data[0:, 0], Fit_Data))
# print(cov_list)


## Aggregation as a function of temperature

tr = 6  ## number of temperature points included: nmb_temp - tr
fig_fmt()
bb = plt.figure(dpi = 600)
bx1 = bb.add_subplot(111)
temp_ls = []
for i in range(nmb_temp):
    temp_ls.append(data[i+2][0])

print(temp_ls)
print(per_agg_ls)

bx1.plot(temp_ls[tr:], per_agg_ls[::-1][tr:], ls = '', marker = 'd', color = 'b', markersize = 10)
bx1.set_xlim(5, 75)
bx1.xaxis.set_major_locator(MultipleLocator(10))
bx1.xaxis.set_minor_locator(MultipleLocator(5))
# bx1.set_ylim(15, 31)
bx1.yaxis.set_major_locator(MultipleLocator(5))
bx1.yaxis.set_minor_locator(MultipleLocator(1))
bx1.set_xlabel('Temperature $\degree$C')
bx1.set_ylabel('Aggregates %')

bx2 = bx1.twinx()
bx2.plot(temp_ls[tr:], eb_list[::-1][tr:], ls = '', marker = 'o', color = 'r', markerfacecolor = 'white', markersize = 10)
bx2.set_ylim(90, 110)
bx2.set_ylabel('W (meV)')
bx2.yaxis.set_major_locator(MultipleLocator(5))
# bx2.yaxis.set_minor_locator(MultipleLocator(5))

set_size(3,2)