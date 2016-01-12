import numpy as np

### Datoteka s konfiguracijskimi parametri ###

PROTEIN_NUM_MAX = 5 # najvecje stevilo genereiranih proteinov
KD_MAX = 100 # omejitev mejne koncentracije aktivatorja potrebnega za aktivacijo Kd
KM_MAX = 30 # omejitev za koncentracije proteina, pri kateri je razgradnja polovicna Km

# zgornje meje alfe, bete in delte
ALPHA_MAX = 1.0 # maksimalne hitrosti izrazanja gena
BETA_MAX = 1.0 # hitrosti modifikacije
DELTA_MAX = 1.0 # hitrost degradacije

# stikalo za generiranje topologije represilatorja
M_SETUP_OSCILATE = False


PERTURBATION_PERMUTATION_WEIGHTS = [0.5, 0.05, 0.2, 0.1, 0.1, 0.05] # utezi za tipe perturbacij
PERTURBATION_POPULATION_PERCENTAGE = 0.1 # delez populacije  ki se ohrani
PERTURBATION_WEIGHT_DEGRADATION = np.array([0.6, 0.2, 0.2]) # utezi za tip degradacije
PERTURBATION_WEIGHT_TYPE = np.array([0.6, 0.2, 0.2]) # utezi za preminjanje tipa generiranja proteina

# velikost populacije
POPULATION_SIZE = 100

# dolzina casovnega intervala
T_MAX = 2

# casovni korak
dt = 0.01

### amplituda in frekvenca referencnega signala ###
IN_FREQ = 1
IN_AMPL = 2

# stikalo za prikaz in shranitev rezultatov algoritma
OUTPUT = True