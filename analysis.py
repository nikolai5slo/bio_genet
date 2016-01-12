import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import peakutils
import findpeaks as fp

"""
    Fitnes funkcija za evalvacijo z metodo MRS (Root Mean Square)
        :param ref
            Koncentracije referencnega (vhodnega) proteina
        :param pred
            Koncentracije, ki so rezultat napovednega modela

        :return
            RMS
"""
def fitness(ref, pred):
    return np.sum(np.power((ref-pred), 2)) / len(ref)