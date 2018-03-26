import pandas as pd
import numpy as np
import gzip
import json

def read_gzipped_jsonl(path):
    list_of_dicts = []
    with gzip.open(path, 'rt') as fout:
        for line in fout:
            m = json.loads(line)
            list_of_dicts.append(m)
    return list_of_dicts


def multiplicity_intenity_to_rates(run_multiplicity, min_exposure_time=0.001):
    run_multiplicity_rate = []
    for m in run_multiplicity:
        if (
            m['ExposureTimePhysicsTrigger'] >= min_exposure_time and
            m['ExposureTimePedestalTrigger'] >= min_exposure_time
        ):
            o = {
                'fNight': m['fNight'],
                'fRunID': m['fRunID'],
                'MultiplicityPhysicsTriggerRate': (
                    np.array(m['MultiplicityPhysicsTrigger'])/
                    m['ExposureTimePhysicsTrigger']
                ),
                'MultiplicityPedestalTriggerRate': (
                    np.array(m['MultiplicityPedestalTrigger'])/
                    m['ExposureTimePedestalTrigger']
                ),
            }
            run_multiplicity_rate.append(o)
    return run_multiplicity_rate


def extract_air_shower_and_nsb_rates(run_multiplicity_rates):
    air_shower_and_nsb_rates = []
    for m in run_multiplicity_rates:
        o = {
            'fNight': m['fNight'],
            'fRunID': m['fRunID'],
            'MultiplicityRateNighSkyBackground': (
                m['MultiplicityPedestalTriggerRate']
            ),
            'MultiplicityRateAirShower': (
                m['MultiplicityPhysicsTriggerRate'] -
                m['MultiplicityPedestalTriggerRate']
            ),
        }
        air_shower_and_nsb_rates.append(o)
    return air_shower_and_nsb_rates


def extract_2D_matrix(dataframe, key):
    out = []
    for m in dataframe[key]:
        out.append(np.array(m))
    out = np.array(out)
    return out


def read_run_wise_air_shower_and_nsb_rate(path):
    run_wise_multiplicity = read_gzipped_jsonl(path)
    run_wise_multiplicity_rates = multiplicity_intenity_to_rates(
        run_wise_multiplicity)
    run_wise_air_shower_and_nsb_rates = extract_air_shower_and_nsb_rates(
        run_wise_multiplicity_rates)
    return run_wise_air_shower_and_nsb_rates


def run_wise_overlap(
    run_info,
    multiplicity_rates_of_air_shower_and_nsb_photons
):
    m = multiplicity_rates_of_air_shower_and_nsb_photons
    m_df = pd.DataFrame(m)
    overlap = pd.merge(
        run_info,
        m_df,
        on=['fNight', 'fRunID'])

    run_info_keys = overlap.keys().tolist()
    run_info_keys.remove('MultiplicityRateAirShower')
    run_info_keys.remove('MultiplicityRateNighSkyBackground')

    air_shower_photon_multiplicity = extract_2D_matrix(
         overlap,
         'MultiplicityRateAirShower')

    night_sky_background_photon_multiplicity = extract_2D_matrix(
         overlap,
         'MultiplicityRateNighSkyBackground')

    return {
        'run_info': overlap[run_info_keys],
        'air_shower_photon_multiplicity': air_shower_photon_multiplicity,
        'night_sky_background_photon_multiplicity': night_sky_background_photon_multiplicity
    }



# Example
def histogram(
    auxilluary,
    multiplicity,
    key='fCurrentsMedMean',
    value_bining=np.linspace(0, 100, 100),
):
    NUM_MULTIPLICITY_BINS = 100

    num_value_bins = value_bining.shape[0]
    valid_runs = np.invert(np.isnan(auxilluary[key]))

    values = auxilluary[valid_runs][key]
    valid_multiplicities = multiplicity[valid_runs, :]

    bin_idx = np.digitize(values, bins=value_bining)

    hist = np.zeros(shape=(NUM_MULTIPLICITY_BINS, num_value_bins))
    normalization = np.zeros(num_value_bins)

    for i, b in enumerate(bin_idx):
        if b < num_value_bins:
            normalization[b] += 1
            hist[:, b] += valid_multiplicities[i, :]

    for i in range(num_value_bins):
        hist[:, i] /= normalization[i]
    return hist, valid_runs
