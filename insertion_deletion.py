import numpy as np
from numba import njit
@njit
def insert_regular_markers(m, Nc, marker_sequence):

    # Get parameters of the code from the specified marker and Nc.
    Nm = marker_sequence.shape[-1]  # Length of the marker sequence.
    N = Nm + Nc                     # Total codeword block with markers.
    rm = Nc/N                       # Rate of the marker code.
    m = m.reshape(1,m.shape[0])
    # If message length does not divide Nc, then pad minimum number of zeros to message bits
    # until message length divides Nc.
    if m.shape[-1] % Nc != 0:
        m = np.concatenate((m, np.zeros((1, Nc-(m.shape[-1] % Nc)))), axis = 1) 

    # Then get the total number of meesage bits!
    mtotal = m.shape[-1]

    # Init the codeword.
    c = np.zeros((1, int(mtotal/rm)))
    
    # marker bit
    mask = np.zeros((1, int(mtotal/rm)))
    for i in range(int(mtotal/Nc)):

        # Get neccessarty indexes!
        low_ind = N*i # low ind
        high_ind = N*(i+1)# high ind
        low_ind_m = (N-Nm)*i
        high_ind_m = (N-Nm)*(i+1)

        # Insert markers!
        c[0, low_ind : high_ind - Nm] = m[0, low_ind_m: high_ind_m]
        c[0, high_ind - Nm: high_ind] = marker_sequence

        # Specfy the locations where markers are inserted !
        mask[0, high_ind - Nm: high_ind] = np.ones((1, Nm))

    return c, mask

