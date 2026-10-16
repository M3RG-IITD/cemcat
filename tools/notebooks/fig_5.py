from new_plot import *
# from plots_crysttaline import infodictcem
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pickle

set_font(weight='normal', size=20)

import re

def make_hill(chemicals):
    '''
    Function for converting chemical formula to hill notation.
    Takes input as a list of chemical formulas, e.g.,
    
    >> chemicals = ['Al2O3', 'C12A7']
    >> make_hill(chemicals)
    Output: ['Al$_2$O$_3$', 'C$_{12}$A$_7$']
    '''
    
    hill_all = []
    
    for col in chemicals:
        # Use regular expression to find elements and their subscripts
        hill = re.sub(r'([A-Za-z])(\d+)', r'\1$_{\2}$', col)  # Convert subscripts
        hill_all.append(hill)
    
    return hill_all

label_dict = {'[0]': 'Fly ash',
 '[1]': '\nClays',
 '[2]': '\nSlag',
 '[3]': 'Fume',
 '[4]': 'Pumice'}

# --- Data ---
dfscms_bar = pd.read_csv('../data/scms_plotting.csv')
infobar = dfscms_bar.label.value_counts()

infodictcem = {'\nPortland': 5294,
 'Limestone': 797,
 '\nGypsum': 548,
    'Aggregate': 398,
    '\nGlass': 204,
    'Clay': 170}

label_dict_cd = {
 '[0]': 'Fly ash',
 '[1]': 'Clays',
 '[2]': 'Slag',
 '[3]': 'Fume',
 '[4]': 'Pumice'}

colors = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#D55E00",  # Red
    "#CC79A7",  # Purple
    "#999999"   # Gray
]

dfa = pd.read_csv('../data/fig_sharexx_top.csv')
dfscms_cd = pd.read_csv('../data/fig_sharexx_scms.csv')

info = dfa.label.value_counts()
labels_cd = info.index[1:6]

with open('co2.pkl','rb') as f:
    co2info = pickle.load(f)

# --- Combined figure: row1=(a)+(b), row2=(c)+(d) with sharex, no gap ---
fig = plt.figure(figsize=(15, 16.5))
outer_gs = GridSpec(2, 1, height_ratios=[5.5, 11], hspace=0.35)

top_gs = outer_gs[0].subgridspec(1, 2, wspace=0.3)
bottom_gs = outer_gs[1].subgridspec(2, 1, hspace=0.0)

# (a) SCMs bar — top left
ax_a = fig.add_subplot(top_gs[0])
labs_a = infobar.index
vals_a = infobar.values
labsx_a = [label_dict[x] for x in labs_a[:5]]
ax_a.bar(labsx_a[:5], vals_a[:5], fc='hotpink', ec='k')
ax_a.tick_params(axis='x', which='minor', bottom=False, top=False)
ax_a.set_ylim(0, 1750)
ax_a.set_ylabel('Frequency', labelpad=15)
ax_a.set_xlabel('SCMs', labelpad=15)
ax_a.text(0.85, 0.85, '(a)', transform=ax_a.transAxes)

# (b) Materials bar — top right
ax_b = fig.add_subplot(top_gs[1])
labs_b = list(infodictcem.keys())
vals_b = list(infodictcem.values())
ax_b.bar(labs_b, vals_b, fc='hotpink', ec='k')
ax_b.tick_params(axis='x', which='minor', bottom=False, top=False)
ax_b.set_ylim(0, 5500)
ax_b.set_xlabel('Materials', labelpad=15)
ax_b.text(0.85, 0.85, '(b)', transform=ax_b.transAxes)

# (c) SCMs year trend — bottom top, full width
ax_c = fig.add_subplot(bottom_gs[0])
year_freq_all = dfscms_cd['rcvd_year'].value_counts().sort_index()
ax_c.plot(year_freq_all.index, year_freq_all.values, '-ok', label='SCMs')
for lid, label in enumerate(labels_cd):
    dfsc = dfscms_cd[dfscms_cd.label == label]
    yf = dfsc['rcvd_year'].value_counts().sort_index()
    ax_c.plot(yf.index, yf.values, '-o', label=label_dict_cd[label], c=colors[lid])
ax_c.set_ylabel('No. of data-points', labelpad=15)
ax_c.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.setp(ax_c.get_xticklabels(), visible=False)
ax_c.text(0.025, 0.9, '(c)', transform=ax_c.transAxes)

# (d) CO2 emissions — bottom bottom, full width, sharex with (c)
ax_d = fig.add_subplot(bottom_gs[1], sharex=ax_c)
ax_d.plot(co2info['x'], co2info['p5'], '--s', color='#9B59B6', label='5 %ile')
ax_d.plot(co2info['x'], co2info['med'], '-D', color='#8E44AD', label='Median')
ax_d.plot(co2info['x'], co2info['p95'], '--^', color='#6C3483', label='95 %ile')
ax_d.set_xticks(co2info['x'])
ax_d.set_ylabel('CO$_2$ Emissions \n(kg/tonne)', labelpad=15)
ax_d.set_xlabel('Year', labelpad=15)
ax_d.tick_params(axis='x', which='minor', top=False, bottom=False)
ax_d.tick_params(axis='x', rotation=90)
ax_d.text(0.025, 0.9, '(d)', transform=ax_d.transAxes)

handles_c, labels_c = ax_c.get_legend_handles_labels()
handles_d, labels_d = ax_d.get_legend_handles_labels()
ax_c.legend(handles_c + handles_d, labels_c + labels_d, frameon=False, ncol=3, loc='center left')

plt.savefig('../figures/fig_5_abcd.svg', format='svg', bbox_inches='tight')
plt.savefig('../figures/fig_5_abcd.pdf', format='pdf', bbox_inches='tight')
plt.show()

