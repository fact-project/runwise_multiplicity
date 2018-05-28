import matplotlib.pyplot as plt
import runwise_multiplicity as rwm
import numpy as np

read_file = rwm.read_gzipped_jsonl('run_database_of_photon_multiplicity_20180207.jsonl.gz')

r = rwm.multiplicity_intenity_to_rates(read_file)
only_asp_rates = r[0]['MultiplicityPhysicsTriggerRate'] - r[0]['MultiplicityPedestalTriggerRate']

# green -> MultiplicityPhysicsTriggerRate ~ NSB + ASP --> r[0]['MultiplicityPhysicsTriggerRate']
# red -> MultiplicityPedestalTriggerRate ~ NSB --> r[0]['MultiplicityPedestalTriggerRate']
# blue -> only_asp_rates ~ subtracting NSB from Total rates, we get ASP rates

bining = np.linspace(1, 101, 100)
plt.loglog(bining, r[0]['MultiplicityPedestalTriggerRate'], color = 'r')
plt.loglog(bining, only_asp_rates, color = 'b')
plt.loglog(bining, r[0]['MultiplicityPhysicsTriggerRate'], color = 'xkcd:lime', linestyle = '--', linewidth = 2 )
plt.xlabel('Multiplicity/1')
plt.ylabel('Multiplicity$^2$ Rates(Multiplicity)/s$^{-1}$')
plt.title('Extracting air shower rates')
plt.savefig(
            'asp_extraction_method.png',
            dpi= 'figure',
            bbox_inches= 'tight'
            )
plt.clf()
