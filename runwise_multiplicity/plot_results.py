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

auxilluary_before_cut = overlap['run_info']
night_sky_background_photon_multiplicity_before_cut = overlap['night_sky_background_photon_multiplicity']
air_shower_photon_multiplicity_before_cut = overlap['air_shower_photon_multiplicity']

above20 = night_sky_background_photon_multiplicity_before_cut[:, 20:-1].sum(axis=1) > 0  # for cutting all the nsb values above 20, because we assume them to be fluctuations and junk

auxilluary = auxilluary_before_cut[np.invert(above20)]
night_sky_background_photon_multiplicity = night_sky_background_photon_multiplicity_before_cut[np.invert(above20)]
air_shower_photon_multiplicity = air_shower_photon_multiplicity_before_cut[np.invert(above20)]

mag_dark_night = 21
delta_mag = mag_dark_night - auxilluary.fSqmMagMean
auxilluary['fSqmFluxRatio'] = np.power(100, (delta_mag/5))

color_sequence = ['xkcd:peach','xkcd:navy','xkcd:grey green']

plot_settings = [
    {   'value_bin_edges': np.linspace(783, 793, 4),
        'value': 'fAirPressureMean',
        'unit': 'mbar',
        'color': color_sequence,
        'name': 'Air Pressure',
        'yrange': [2.7*(10**6), 5*(10**6)],
        'selection': [786, 791]
    },
    {   'value_bin_edges': np.linspace(6, 17, 4),
        'value': 'fOutsideTempMean',
        'unit': 'C',
        'color': color_sequence,
        'name': 'Temperature',
        'yrange': [3*(10**6), 5.5*(10**6)],
        'selection': [6, 14]
    },
    {   'value_bin_edges': np.linspace(0.5, 6, 4),
        'value': 'fSqmFluxRatio',
        'unit': '$F_{dark night}$',
        'color': color_sequence,
        'name': 'SQM',
        'yrange': [3.5*(10**6), 2.5*(10**7)],
        'selection': [0.9, 1.1]
    },
    {   'value_bin_edges': np.linspace(15, 70, 4),
        'value': 'fHumidityMean',
        'unit': '%',
        'color': color_sequence,
        'name': 'Humidity',
        'yrange': [2.7*(10**6), 5*(10**6)],
        'selection': [5, 30]
    },
    {   'value_bin_edges': np.linspace(-30, 0, 4),
        'value': 'fDewPointMean',
        'unit': 'C',
        'color': color_sequence,
        'name': 'Dew Point',
        'yrange': [2.7*(10**6), 5.5*(10**6)],
        'selection': [-13, 2]
    },
    {   'value_bin_edges': np.linspace(0, 10, 4),
        'value': 'fTNGDust',
        'unit': '(ugr/m$^3$)',
        'color': color_sequence,
        'name': 'Dust Concen.',
        'yrange': [2*(10**6), 5*(10**6)],
        'selection': [0, 5]
    },
    {   'value_bin_edges': np.linspace(5, 50, 4),
        'value': 'fZenithDistanceMean',
        'unit': 'deg',
        'color': color_sequence,
        'name': 'Zenith Distance',
        'yrange': [2.5*(10**6), 5*(10**6)],
        'selection': [20, 30]
    },
]

def make_selection_mask(plot_settings, value, auxilluary):
    mask = np.ones(auxilluary.shape[0], dtype=np.bool)
    for ps in plot_settings:
        if ps['value'] != value:
            above_lower_selection_mask = auxilluary[ps['value']] >= ps['selection'][0]
            below_upper_selection_mask = auxilluary[ps['value']] < ps['selection'][1]
            sub_mask = above_lower_selection_mask & below_upper_selection_mask
            mask = mask & sub_mask
            print(ps['value'], np.sum(sub_mask))
    return mask

for plot_setting in plot_settings:
    ps = plot_setting

    selection_mask = make_selection_mask(
        plot_settings=plot_settings,
        value=ps['value'],
        auxilluary=auxilluary)

    h_asp, v_asp = rwm.histogram(
        auxilluary=auxilluary[selection_mask],
        multiplicity=air_shower_photon_multiplicity[selection_mask],
        key=ps['value'],
        value_bin_edges=ps['value_bin_edges'])

    h_nsb, v_nsb = rwm.histogram(
        auxilluary=auxilluary[selection_mask],
        multiplicity=night_sky_background_photon_multiplicity[selection_mask],
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

    os.makedirs(ps['value'], exist_ok=True)

    #for different log plots
    fig, ax = plt.subplots()
    for number in range(h_asp.shape[1]):
        ax.loglog(
            np.linspace(11, 100, 90),
            h_asp[10:100, number]*np.linspace(11, 100, 90)**2.7,
            label=(
                str((ps['value_bin_edges'][number]).round(1)) +
                ' to ' +
                str((ps['value_bin_edges'][number + 1]).round(1)) +
                ' ' +
                ps['unit']
            ),
            color=ps['color'][number],
            )
    #ax.set_ylim(ps['yrange'])
    #ax.set_xlim(8, 100)
    plt.legend()
    plt.xlabel('Multiplicity/1')
    plt.ylabel('Multiplicity$^{2.7}$ Rates(Multiplicity)/s$^{-1}$')
    plt.title(ps['name']+ '/'+ ps['unit'])
    plt.savefig(
                os.path.join(ps['value'],'2DPlot_loglog.png'),
                dpi= 'figure',
                bbox_inches= 'tight'
                )
    plt.clf()

    # for different histograms corresponding to constant selected range
    valid_runs = v_asp & v_nsb
    for i in range(len(ps['value_bin_edges']) - 1):
        start = ps['value_bin_edges'][i]
        stop = ps['value_bin_edges'][i+1]
        bins=np.linspace(start, stop, 20)
        plt.hist(
            auxilluary[ps['value']][valid_runs & selection_mask],
            bins=bins,
            range=(start, stop),
            color=ps['color'][i]
        )
    plt.xlabel(ps['name']+ '/'+ ps['unit'])
    plt.ylabel('Number of runs')
    plt.title('# observations corresponding to different ' + ps['name'] + ' values')
    plt.savefig(
                (ps['value'] + '.png'),
                dpi= 'figure',
                bbox_inches= 'tight'
                )
    plt.clf()
