from new_plot import *
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

set_font(weight='normal', size=25)

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

# Example usage
chemicals = ['Al2O3', 'C12A7']
print(make_hill(chemicals))


df = pd.read_csv('phase_plot_data.csv')
df.columns = ['phase','freq']
df['labels'] = ['C2S','C3A','C3S','C4AF','Gypsum + \n Anhydrite', 'CaCO3','CaO','C4A3$\overline{S}$', 'Quartz','Portlandite','C12A7','Dolomite','SO3','Mullite','Wollastonite']
df = df[['phase','labels','freq']]

# Data for (a)
labels_a = make_hill(list(df.labels)[:-2][::-1])
freq = list(df.freq)[:-2][::-1]

# Data for (b)
compounds = {
 'SiO2': 23754,
 'CaO': 23483,
 'Al2O3': 22999,
 'Fe2O3': 21576,
 'MgO': 20852,
 'SO3': 16491,
 'K2O': 15911,
 'Na2O': 14871,
 'TiO2': 7978,
 'P2O5': 5159,
 'MnO': 3410,
 'SrO': 1091,
 'Cr2O3': 751,
 'Mn2O3': 503,
 'ZnO': 499,
 'BaO': 395,
 'FeO': 345,
 'ZrO2': 316,
 'CuO': 216,
 'CaCO3': 175,
 'NiO': 159}

def make_hill_b(chemicals):
    hill_all = []
    for col in chemicals:
        hill = ''
        for ele in col:
            if ele in ['2','3','4','5','6','7','8','9','10','11','12','13']:
                hill = hill + '$_' + ele + '$'
            else:
                hill = hill + ele
        hill_all.append(hill)
    return hill_all

labs = list(compounds.keys())
vals = list(compounds.values())
labsx = make_hill_b(labs)

# --- Combined figure: (a) left, (b) right ---
fig = plt.figure(figsize=(22, 10))
gs = GridSpec(1, 2, width_ratios=[9, 16], wspace=0.15)

# (a) Phases horizontal bar chart
ax1 = fig.add_subplot(gs[0])
ax1.barh(labels_a, freq, fc='hotpink', ec='k')
ax1.set_xlabel('Frequency', labelpad=15)
ax1.set_ylabel('Phases', labelpad=15)
ax1.tick_params(axis='y', which='minor', right=False, left=False)
ax1.text(0.9, 0.1, '(a)', transform=ax1.transAxes)

# (b) Oxides bar chart
ax2 = fig.add_subplot(gs[1])
ax2.bar(labsx, vals, fc = 'hotpink', ec='k')
ax2.set_xticklabels(labsx, rotation=90)
ax2.tick_params(axis='x', which='minor', bottom=False, top=False)
ax2.set_ylabel('Frequency', labelpad=15)
ax2.set_xlabel('Oxides', labelpad=15)
ax2.set_yscale('log')
ax2.set_ylim(0, 30000)
ax2.text(0.9, 0.9, '(b)', transform=ax2.transAxes)

# plt.savefig('../figures/fig_4_ab.png', dpi=1200, bbox_inches='tight')
plt.savefig('../figures/fig_4_ab.svg', format='svg', bbox_inches='tight')
plt.savefig('../figures/fig_4_ab.pdf', format='pdf', bbox_inches='tight')
plt.show()

