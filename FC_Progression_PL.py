import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import matplotlib.pylab as pl
from Figure_Size import fig_fmt, set_size

lb = 250 #high energy  220
ub = 670 #low  650

h_bar = 1
ome_0 = 0.175 # eV/h_bar
m = 3
lam_sqr = 1

## functions defined for the fit--------------------------------

def Gau(x, sig): # sig is squared already
    return np.exp(-(x)**2 / (2 * sig**2))

def Oscilator(w, sig, E_0, E_p):
    sum0 = 0.0
    for i in range(m):
        sum0 = sum0 + (lam_sqr**(i+1) / math.factorial(i+1))\
               * Gau(h_bar*w - E_0 + (i+1)*h_bar* E_p, sig)
    return sum0

def I(w, a, sig, E_0, E_p, const):
    z = const* np.exp(-lam_sqr)* (h_bar * w)**3 * np.exp(-lam_sqr)\
        *(a * Gau(h_bar*w - E_0, sig) + Oscilator(w, sig, E_0, E_p))
    return z

## Data for nP3HT PL Spectra; nP3HT_Xy Mar302021_Cycle1
filepath_in = "/Users/yulongzheng/Dropbox (GaTech)/Experimental Data/PL/Seongmin_TDPL_Cycling_Mar292021/Mar302021_Cycle1/nP3HT_2__Xy_Avg5_ET2s_730nm_Cycle1-"
filename_in = [11]#, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
               #150, 160, 175, 182, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290]
ext = 'K.dat'


# background noise
filepath_bkg = "/Users/yulongzheng/Dropbox (GaTech)/Experimental Data/PL/Apr102021/Apr102021/bkg-"
filename_bkg = '294'
ext_bkg = 'K.dat'

bkg = np.loadtxt(filepath_bkg + filename_bkg + ext_bkg)

# White light spectra
Instr_Resp = np.loadtxt('/Users/yulongzheng/Desktop/Seongmin_TDPL/Instrument_Response.txt')

nmb_samples = len(filename_in)
file_list = []
title = 'Cool Down'
# colors_cool = pl.cm.Blues(np.linspace(0.3, 0.8, int(len(filename_in)//2)))[::-1]
# colors_warm = pl.cm.Reds(np.linspace(0.3, 0.8, int(len(filename_in) - (len(filename_in)//2))))
# colors = np.vstack((colors_cool, colors_warm))
colors = pl.cm.RdBu_r(np.linspace(0.1, 0.8, len(filename_in)))


for i in range(len(filename_in)):
    file = filepath_in + str(filename_in[i]) + ext
    fo = open(file, "r") #, encoding="utf-16")  # encoding: special signs
    file_list.append(fo)

## Physical Constants
h = 6.63 * 10**-34
c = 3 * 10**8
eV = 1.602 * 10**-19
#          a,   sig, E_0, E_p, cons
bounds = [[0.4, 0.05, 1.75, 0.14, 1.0],
         [0.6, 0.08, 1.9, 0.2, 1.5]]


## Background subtraction, Jacobian transformation, unit conversion
max_val = []
I00_list = []
I01_list = []
ar = []
ar_list = []
y, x = [], []
y_nonnor = []
for j in range(len(file_list)):
    for line in file_list[j].readlines():
        entry = line.split()
        entry = [float(i) for i in entry]
        ar.append(entry)
    ar_list.append(np.asarray(ar))
    ar = []
    #     print(np.shape(ar_list))
    for n in range(len(ar_list[j][:, 0])):
        ar_list[j][n, 0] = 1240 / ar_list[j][n, 0]
        ar_list[j][n, 1] = ar_list[j][n, 1] - bkg[n, 1]  ## substract the bkg
        ar_list[j][n, 1] = ar_list[j][n, 1] * h * c / np.square((ar_list[j][n, 0] * eV))

    x.append(ar_list[j][:, 0])
    y.append(ar_list[j][:, 1])

x_new = x
y_new = y / np.max(y)
for j in range(np.shape(y)[0]):
    y[j] = [y[j][i] / Instr_Resp[i, 1] for i in range(np.shape(y)[1])]

for j in range(np.shape(y)[0]):
    y_nonnor.append(y[j])

for j in range(np.shape(y)[0]):
    y[j] = y[j] / np.max(y[j][419])

## Plotting
I00_sum_list = []
I01_sum_list = []

pts_int = 21

fig_fmt()
aa = [plt.figure(i) for i in range(nmb_samples)]
# ax1 = [aa.add_subplot(nmb_samples, 1, j+1) for j in range(nmb_samples)]

bb = plt.figure()
cc = plt.figure(dpi=300)
dd = plt.figure(dpi=300)
ax2 = cc.add_subplot(111)
for i in range(nmb_samples):
    ax1 = aa[i].add_subplot(111)
    ax1.plot(x_new[i][lb:ub], y_new[i][lb:ub], 'm', label='Raw data')
    #     ax1.plot(x_new[i], y_new[i], 'm', label = 'Raw data')
    #     ax1.plot(Instr_Resp[:, 0], Instr_Resp[:, 1]/np.max(Instr_Resp[:, 1]), label = 'Instrument response')
    ax1.plot(x[i], y[i], marker='.', markevery=10, linewidth=1.5, color=colors[i], label='Corrected')
    #     meas = I(x[i], 0.52, 0.072, 1.85, 0.165, 1.4) #inital guesses
    #     ax1.plot(x[i], meas, 'b-', label='Intial Guess')

    popt, pcov = curve_fit(I, x[i][lb: ub], y[i][lb: ub],
                           #          a,   sig, E_0, E_p, cons
                           bounds=([0.2, 0.04, 1.80, 0.1, 1.0],
                                   [0.6, 0.1, 1.95, 0.25, 5]))

    #     popt, pcov = curve_fit(I, x[i][lb: ub], y[i][lb: ub],
    #                        #          a,   sig, E_0, E_p, cons
    #                        bounds = ([0.2, 0.05, 1.8, 0.1, 1.0],
    #                                  [0.6, 0.1, 1.9, 0.25, 3]))
    I_fit = I(x[i], *popt)
    #     I00 = I(popt[2], *popt)
    E0_ind = np.argmin(np.abs(popt[2] - x[i]))  # index of the E_0
    E1_ind = np.argmin(np.abs(popt[2] - popt[3] - x[i]))  # index of E_1
    E0 = x[i][E0_ind]
    E1 = x[i][E1_ind]
    I00 = y[i][E0_ind]
    I01 = y[i][E1_ind]
    #     I00_range = np.where((abs(I00-I_fit) <= 4e-3) & (abs(x[i] - popt[2]) <= 1e-2))
    # #     I01 = I(popt[2]-popt[3], *popt)
    #     I01_range = np.where((abs(I01-I_fit) <= 4e-3) & (abs(x[i] - popt[2] + popt[3]) <= 1e-2))
    #     print('I00', I00)
    #     print('I01', I01)
    #     print('I00', I00_range)
    #     print('I01', I01_range)
    I00_sum, I01_sum = 0, 0

    for j in range(pts_int):  # integrate over 21 points centered at the I00/I01 peak fit from the FC analysis
        I00_sum = I00_sum + y[i][E0_ind - int((pts_int - 1) / 2) + j]
        I01_sum = I01_sum + y[i][E1_ind - int((pts_int - 1) / 2) + j]

    #     for j in range(len(I00_range)):
    #         I00_sum = I00_sum + I(x[i][I00_range[0][j]], *popt)
    #         I01_sum = I01_sum + I(x[i][I01_range[0][j]], *popt)

    I00_sum_list.append(I00_sum)
    I01_sum_list.append(I01_sum)
    I00_list.append(I00)
    I01_list.append(I01)

    #     print(x[i][E1_ind])
    ax1.plot(x[i], I(x[i], *popt), 'k', linewidth=1.5,
             label='fit: a=%1.2f, sig=%5.3f,\n'
                   'E_0=%5.3f, E_p=%5.3f, cons=%5.3f' % tuple(popt))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    # I_aggregated = datay - I(datax, *popt)
    # plt.plot(datax, I_aggregated, 'm-', markersize = 4, label = 'amorphous')
    ax1.set_xlabel('Photon Energy (eV)')
    ax1.set_ylabel('Norm. Absorbance')
    #     ax1.vlines([I00, I01], 0, 1, transform=ax1.get_xaxis_transform(), colors='r')
    ax1.axvline(x=E0, color='black', linestyle='-')
    ax1.axvline(x=E1, color='black', linestyle='-')
    ax1.legend(frameon=False, loc='best', fontsize=10)
    text_inside = f'{filename_in[i]}'
    ax1.annotate(text_inside, xy=(0.02, 0.93), xycoords='axes fraction', size=12)

    #     ax1.set_xlim(1.50, 2.00)
    aa[i].tight_layout()

    ## PL Spectrum
    ax2.plot(x[i], y_nonnor[i], marker='', markevery=10, linestyle='-', color=colors[i],
             label=str(filename_in[i]) + ' K')  # /y_nonnor[i][419]
    ax2.set_xlabel('Photon Energy (eV)')
    ax2.set_ylabel('Intensity')
    ax2.legend(frameon=False, loc='best', fontsize=5)
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
#     ax2.set_xlim(1.5, 2.2)
#     ax2.set_ylim(0, 1.2)

plt.show()
