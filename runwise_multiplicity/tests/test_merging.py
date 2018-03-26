import pandas as pd
import runwise_multiplicity as rwm
import pkg_resources
import os

def test_merging():
    run_info = pd.read_msgpack(
        pkg_resources.resource_filename(
            'runwise_multiplicity',
            os.path.join('tests',  'resources',
            'run_info_201712.msg')))

    assert len(run_info) == 4954

    rates = rwm.read_run_wise_air_shower_and_nsb_rate(
        pkg_resources.resource_filename(
            'runwise_multiplicity',
            os.path.join('tests', 'resources',
            'multiplicity_20171115_20171215.jsonl.gz')))

    assert len(rates) == 2593

    overlap = rwm.run_wise_overlap(
        run_info=run_info,
        multiplicity_rates_of_air_shower_and_nsb_photons=rates)

    assert (
        len(overlap['run_info']) ==
        overlap['air_shower_photon_multiplicity'].shape[0])

    assert (
        len(overlap['run_info']) ==
        overlap['night_sky_background_photon_multiplicity'].shape[0])

    assert overlap['night_sky_background_photon_multiplicity'].shape[1] == 100
    assert overlap['air_shower_photon_multiplicity'].shape[1] == 100

    assert len(overlap['run_info']) < len(run_info)
