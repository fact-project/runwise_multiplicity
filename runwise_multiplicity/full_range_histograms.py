import matplotlib.pyplot as plt
import matplotlib.patches as pat
import runwise_multiplicity as rwm
import pandas as pd
import numpy as np
import os
import subprocess as sp
import math
import matplotlib.colors as colors

run_info = pd.read_msgpack(
    'fact_run_info_table_20180319.msg')

rates = rwm.read_run_wise_air_shower_and_nsb_rate(
    'run_database_of_photon_multiplicity_20180207.jsonl.gz')

overlap = rwm.run_wise_overlap(
    run_info=run_info,
    multiplicity_rates_of_air_shower_and_nsb_photons=rates)

auxilluary = overlap['run_info']
night_sky_background_photon_multiplicity = overlap['night_sky_background_photon_multiplicity']
air_shower_photon_multiplicity = overlap['air_shower_photon_multiplicity']

above20 = night_sky_background_photon_multiplicity[:, 20:-1].sum(axis=1) > 0  # for cutting all the nsb values above 20, because we assume them to be fluctuations and junk

auxilluary = auxilluary[np.invert(above20)]
night_sky_background_photon_multiplicity = night_sky_background_photon_multiplicity[np.invert(above20)]
air_shower_photon_multiplicity = air_shower_photon_multiplicity[np.invert(above20)]

mag_dark_night = 21
delta_mag = mag_dark_night - auxilluary.fSqmMagMean
auxilluary['fSqmFluxRatio'] = np.power(100, (delta_mag/5))

plot_settings = [
    {   'value_bin_edges': np.linspace(780, 795, 4),
        'value': 'fAirPressureMean',
        'unit': 'mbar',
        'name': 'Air Pressure',
        'selection': [786, 791]
    },
    {   'value_bin_edges': np.linspace(0, 25, 4),
        'value': 'fOutsideTempMean',
        'unit': 'C',
        'name': 'Temperature',
        'selection': [6, 14]
    },
    {   'value_bin_edges': np.linspace(0.5, 6, 4),
        'value': 'fSqmFluxRatio',
        'unit': '$F_{dark night}$',
        'name': 'SQM',
        'selection': [0.9, 1.1]
    },
    {   'value_bin_edges': np.linspace(5, 99, 4),
        'value': 'fHumidityMean',
        'unit': '%',
        'name': 'Humidity',
        'selection': [5, 30]
    },
    {   'value_bin_edges': np.linspace(-45, 10, 4),
        'value': 'fDewPointMean',
        'unit': 'C',
        'name': 'Dew Point',
        'selection': [-13, 2]
    },
    {   'value_bin_edges': np.linspace(0, 15, 4),
        'value': 'fTNGDust',
        'unit': '(ugr/m$^3$)',
        'name': 'Dust Concen.',
        'selection': [0, 5]
    },
    {   'value_bin_edges': np.linspace(10, 70, 4),
        'value': 'fZenithDistanceMean',
        'unit': 'deg',
        'name': 'Zenith Distance',
        'selection': [20, 30]
    },
]

for ps in plot_settings:

    h_asp, v_asp = rwm.histogram(
        auxilluary=auxilluary,
        multiplicity=air_shower_photon_multiplicity,
        key=ps['value'],
        value_bin_edges=ps['value_bin_edges'])

    h_nsb, v_nsb = rwm.histogram(
        auxilluary=auxilluary,
        multiplicity=night_sky_background_photon_multiplicity,
        key=ps['value'],
        value_bin_edges=ps['value_bin_edges'])

    not_nan_asp = np.invert(np.isnan(h_asp))
    larger_zero_asp = h_asp > 0.0
    min_asp_rate = np.min(h_asp[not_nan_asp*larger_zero_asp])
    max_asp_rate = np.max(h_asp[not_nan_asp*larger_zero_asp])

    not_nan_nsb = np.invert(np.isnan(h_nsb))
    larger_zero_nsb = h_nsb > 0.0
    min_nsb_rate = np.min(h_nsb[not_nan_nsb*larger_zero_nsb])
    max_nsb_rate = np.max(h_nsb[not_nan_nsb*larger_zero_nsb])

    min_rate = np.min([min_asp_rate , min_nsb_rate])
    max_rate = np.max([max_asp_rate , max_nsb_rate])

    valid_runs = v_asp & v_nsb
    fig, ax = plt.subplots()
    for i in range(len(ps['value_bin_edges']) - 1):
        start = ps['value_bin_edges'][i]
        stop = ps['value_bin_edges'][i+1]
        bins=np.linspace(start, stop, 20)
        hhh = ax.hist(
            auxilluary[ps['value']][valid_runs],
            bins=bins,
            range=(start, stop),
            color='xkcd:sea blue'
        )
        ax.vlines(ps['selection'], 0, np.max(hhh[0]), colors='k', linestyle='dashed')
    plt.xlabel(ps['name']+ '/'+ ps['unit'])
    plt.ylabel('Number of runs')
    plt.title('# observations corresponding to different ' + ps['name'] + ' values')
    plt.savefig(
                (ps['value'] + '-fullrange.png'),
                dpi= 'figure',
                bbox_inches= 'tight'
                )
    plt.clf()
