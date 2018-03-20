from distutils.core import setup

setup(
    name='runwise_multiplicity',
    version='0.0.0',
    description='Read & plot the run-wise photon multiplicities',
    url='https://github.com/fact-project/',
    author='Amandeep Singh, Sebastian Achim Mueller',
    author_email='ads4169@gmail.com, sebmuell@phys.ethz.ch',
    license='MIT',
    packages=[
        'runwise_multiplicity',
    ],
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'pyfact',
        'pandas',
    ],
    zip_safe=False,
)
