from new_plot import *
# from plots_crysttaline import infodictcem
from utils import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

set_font(weight='normal', size=20)

label_dict = {
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
dfscms = pd.read_csv('../data/fig_sharexx_scms.csv')


info = dfa.label.value_counts()
labels = info.index[1:6]
values = info.values[1:6]


year_freq = dfa['rcvd_year'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=[15,5.5])

year_freq = dfscms['rcvd_year'].value_counts().sort_index()
plt.plot(year_freq.index, year_freq.values,'-ok', label='SCMs')

plt.xlabel('Year', labelpad=15)
# plt.xticks(year_freq.index, rotation=90)
plt.xticks(year_freq.index,[])
plt.tick_params(axis='x', which='minor', bottom=False, top=False)
plt.ylabel('No. of data-points', labelpad=15)



for lid, label in enumerate(labels):
    dfsc = dfscms[dfscms.label == label]
    year_freq = dfsc['rcvd_year'].value_counts().sort_index()
    plt.plot(year_freq.index, year_freq.values, '-o', label = label_dict[label], c = colors[lid])
plt.legend()
plt.show()
