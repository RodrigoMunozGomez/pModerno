import numpy as np
import scipy.io
def CSMatrixGen(nKernel=64,CR=0.1,number=1):
    N = nKernel * nKernel  # Número total de píxeles

    M = int(N * CR)  # Número de mediciones
    p = CR  # Probabilidad de que un elemento sea diferente de cero

    # Generar la matriz de sensado con distribución de Bernoulli
    Phi = np.random.choice([0, 1], size=(M, N), p=[1 - p, p])
    CSmatrix = {'CSmatrix': Phi}
    scipy.io.savemat(f'/data2/rmunoz/P_Moderno/CS_Decoder/CSMatrices/{nKernel}Matrix/CR_{CR}_{number}.mat', CSmatrix)
    return Phi