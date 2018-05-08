import matplotlib.pyplot as plt
import matplotlib.patches as pat
import runwise_multiplicity as rwm
import pandas as pd
import numpy as np
import os
import subprocess as sp

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

color_sequence = ['xkcd:prussian blue','xkcd:pastel blue','xkcd:mahogany','xkcd:greyish','xkcd:bottle green']

plot_settings = [
    {   'value_bin_edges': np.linspace(780, 797.5, 6),
        'value': 'fAirPressureMean',
        'unit': 'mbar',
        'color': color_sequence,
        'name': 'Air Pressure',
    },
    {   'value_bin_edges': np.linspace(0, 22, 6),
        'value': 'fOutsideTempMean',
        'unit': 'C',
        'color': color_sequence,
        'name': 'Temperature'
    },
    {   'value_bin_edges': np.linspace(18.75, 21.2, 6),
        'value': 'fSqmMagMean',
        'unit': '[]',
        'color': color_sequence,
        'name': 'SQM'
    },
    {   'value_bin_edges': np.linspace(0, 100, 6),
        'value': 'fHumidityMean',
        'unit': '%',
        'color': color_sequence,
        'name': 'Humidity'
    },
    {   'value_bin_edges': np.linspace(-48, 15, 6),
        'value': 'fDewPointMean',
        'unit': 'C',
        'color': color_sequence,
        'name': 'Dew Point'
    },
    {   'value_bin_edges': np.linspace(15.91, 35, 6),
        'value': 'fContainerTempMean',
        'unit': 'C',
        'color': color_sequence,
        'name': 'Container Temp.'
    },
    {   'value_bin_edges': np.linspace(0, 75, 6),
        'value': 'fCurrentsMedMean',
        'unit': 'uA',
        'color': color_sequence,
        'name': 'Current'
    },
    {   'value_bin_edges': np.linspace(0, 350, 6),
        'value': 'fTNGDust',
        'unit': '(ugr/m3)',
        'color': color_sequence,
        'name': 'Dust Concen.'
    },
    {   'value_bin_edges': np.linspace(4.83, 75.58, 6),
        'value': 'fZenithDistanceMean',
        'unit': 'deg',
        'color': color_sequence,
        'name': 'Zenith Distance'
    },
]

for plot_setting in plot_settings:
    ps = plot_setting

    h_asp, v = rwm.histogram(
        auxilluary=auxilluary,
        multiplicity=air_shower_photon_multiplicity,
        key=ps['value'],
        value_bin_edges=ps['value_bin_edges'])

    h_nsb, v = rwm.histogram(
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

    os.makedirs(ps['value'], exist_ok=True)

    #for different log plots
    for number in range(h_asp.shape[1]):
        plt.semilogx(
            np.linspace(1, 101, 100),
            h_asp[:, number],
            label=(
                str(ps['value_bin_edges'][number]) +
                ' to ' +
                str(ps['value_bin_edges'][number + 1]) +
                ' ' +
                ps['unit']
            ),
            color=ps['color'][number],
            )
    plt.legend()
    plt.xlabel('Multiplicity/1')
    plt.ylabel('Rate/s')
    plt.title(ps['name']+ '/'+ ps['unit'])
    plt.savefig(
                os.path.join(ps['value'],'2DPlot_semilogx.png'),
                dpi= 'figure',
                bbox_inches= 'tight'
                )
    plt.clf()

    for i in range(len(ps['value_bin_edges']) - 1):
        start = ps['value_bin_edges'][i]
        stop = ps['value_bin_edges'][i+1]
        bins=np.linspace(start, stop, 20)
        plt.hist(
            auxilluary[ps['value']],
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
