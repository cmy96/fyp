import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib.pyplot as plt
from matplotlib import cm
from pywaffle import Waffle
from plotly.tools import mpl_to_plotly
import requests
import io


# Draw Plot and Decorate
data = {'Survivors with surgery alone': 89, 'Breast cancer related deaths': 10, 'Death due to other causes': 1}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    cmap_name="tab10",
    legend={
         'labels': ["{0} ({1})".format(k, v) for k, v in data.items()],
        'loc': 'upper left', 'bbox_to_anchor': (1, 1)
        },
    icons='child', icon_size=18, 
    icon_legend=True,
    figsize=(10, 9),
    title={
    'label': 'Survivability Rate for Breast Cancer Patients ',
    'loc': 'center',
    'fontdict': {
    'fontsize': 12
    }
    })

plt.savefig('assets/waffle-1-chart.png',bbox_inches='tight', pad_inches=0)

fig2 = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=[34,66],
    colors=["#e47446", "#5cb0c2"],
    icons='child',
    font_size=18,
    icon_style='solid',
    icon_legend=True,
    legend={
        'labels': ['Chemotherapy (34)', 'No Chemotherapy (66)'], 
        'loc': 'upper left', 
        'bbox_to_anchor': (1, 1)
    },
    figsize=(10, 9),
    title={
    'label': 'Treatment (Chemotherapy)',
    'loc': 'center',
    'fontdict': {
    'fontsize': 12
    }
    }
)
plt.savefig('assets/waffle-2-chart.png',bbox_inches='tight', pad_inches=0)



data = {'0-6 months': 25, '> 6 months': 15, 'After 1 Year': 8, 'After 2 Years': 7,'After 5 Years': 45}
fig3 = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    cmap_name="tab20",
    legend={
        'labels': ["{0} ({1})".format(k, v) for k, v in data.items()],
        'loc': 'upper left', 'bbox_to_anchor': (1, 1)},
    icons='child', icon_size=18, 
    icon_legend=True,
    figsize=(10, 9),
    title={
    'label': 'Survivability for Breast Cancer Patients ',
    'loc': 'center',
    'fontdict': {
    'fontsize': 12
    }
    })
plt.savefig('assets/waffle-3-chart.png',bbox_inches='tight', pad_inches=0)

fig4 = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=[48,52],
    colors=["#e47446", "#5cb0c2"],
    icons='child',
    font_size=18,
    icon_style='solid',
    icon_legend=True,
    legend={
        'labels': ['Radiotherapy', 'No Radiotherapy'], 
        'loc': 'upper left', 
        'bbox_to_anchor': (1, 1)
    },
    figsize=(10, 9),
    title={
    'label': 'Treatment (Radiotherapy)',
    'loc': 'center',
    'fontdict': {
    'fontsize': 12
    }
    }
)

plt.savefig('assets/waffle-4-chart.png',bbox_inches='tight', pad_inches=0)

fig5 = plt.figure(
    FigureClass=Waffle,
    rows=5,
    values=[49,51],
    colors=["#e47446", "#5cb0c2"],
    icons='child',
    font_size=18,
    icon_style='solid',
    icon_legend=True,
    legend={
        'labels': ['Surgery', 'No Surgery'], 
        'loc': 'upper left', 
        'bbox_to_anchor': (1, 1)
    },
    figsize=(10, 9),
    title={
    'label': 'Treatment (Surgery)',
    'loc': 'center',
    'fontdict': {
    'fontsize': 12
    }
    }
)


plt.savefig('assets/waffle-5-chart.png',bbox_inches='tight', pad_inches=0)
plt.show()




