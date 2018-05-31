import matplotlib.pyplot as plt
import runwise_multiplicity as rwm
import numpy as np
import pandas as pd

run_info = pd.read_msgpack(
    'fact_run_info_table_20180319.msg')

read_file = rwm.read_gzipped_jsonl(
    'run_database_of_photon_multiplicity_20180207.jsonl.gz')
r = rwm.multiplicity_intenity_to_rates(read_file)

for run_index in range(len(r)):
    only_asp_rates = (
        r[run_index]['MultiplicityPhysicsTriggerRate'] -
        r[run_index]['MultiplicityPedestalTriggerRate'])

    fNight = r[run_index]['fNight']
    fRunID = r[run_index]['fRunID']

    run_mask = (
        (run_info.fNight == fNight) &
        (run_info.fRunID == fRunID))

    fCurrentsMedMean = run_info.fCurrentsMedMean[run_mask].as_matrix()[0]

    if fCurrentsMedMean < 10:
        break

print('Night:', fNight, 'Run:', fRunID, 'Current:', fCurrentsMedMean)

#MultiplicityPhysicsTriggerRate ~ NSB + ASP --> r[0]['MultiplicityPhysicsTriggerRate']
#MultiplicityPedestalTriggerRate ~ NSB --> r[0]['MultiplicityPedestalTriggerRate']
#only_asp_rates ~ subtracting NSB from Total rates, we get ASP rates
fig, ax = plt.subplots()
bining = np.linspace(1, 101, 100)
ax.loglog( bining,
            r[run_index]['MultiplicityPedestalTriggerRate']*np.linspace(1, 101, 100)**2.7,
            color = 'r')
ax.loglog( bining,
            only_asp_rates*np.linspace(1, 101, 100)**2.7,
            color = 'b')
ax.loglog( bining,
            r[run_index]['MultiplicityPhysicsTriggerRate']*np.linspace(1, 101, 100)**2.7,
            color = 'xkcd:lime',
            linestyle = '--',
            linewidth = 2 )
plt.xlabel('Multiplicity/1')
plt.ylabel('Multiplicity$^{2.7}$ Rates(Multiplicity)/s$^{-1}$')
plt.savefig(
            'asp_extraction_method.png',
            dpi= 'figure',
            bbox_inches= 'tight'
            )
plt.clf()
