import numpy as np
from numba import njit
@njit
def ins_del_channel(x,pd,pi,ps,safety_bits):
    new_array = np.zeros((x.shape[0],safety_bits))
    ps_effective = (1-pi-pd)*ps
    for j in range(x.shape[0]):
        curr=0
        for k in range(x.shape[1]):
            p = np.random.rand(1)
            while p<pi:
                new_array[j,curr] = np.random.randint(low=0,high=2,size=(1,))[0]
                curr += 1
                p = np.random.rand(1)
                #print(p)
            else:
                if pi+pd<p and p<pi+pd+ps_effective:
                    new_array[j,curr] = -x[j,k]+ 1
                    curr += 1
                elif pi+pd+ps_effective<p:
                    new_array[j,curr] = x[j,k] 
                    curr += 1
    return new_array,new_array

def create_batch(m_total, num_code,safety_bits, Pd, Pi, Ps, Nc, marker_sequence):

  Nr = marker_sequence.shape[-1]
  r = Nc/(Nc+Nr)

  # If m shape does not divide N_c
  if m_total % Nc != 0:
      m_total = m_total + Nc - (m_total % Nc);

  trainX = np.zeros((num_code, safety_bits, safety_bits))
  trainY = np.zeros((num_code, int(m_total/r), 1))

  for i in range(num_code):

    # Get random samples from the specified training Ps and Pd.
    Pd_sample = np.random.uniform(0,Pd)
    Ps_sample = np.random.uniform(0,Ps)
    Pi_sample = np.random.uniform(0,Pi)

    # create message
    m = np.random.randint(0,2, size = (1, m_total))

    # create marked code bits
    c, mask = insert_regular_markers(m, Nc, marker_sequence)

    # channel
    y,trans = ins_del_channel(c, Pd_sample, Pi_sample, Ps_sample,safety_bits)
  
    #y = np.array(y).T
    #print(c)
    numR = y.shape[-1]
    #print(numR)
    T = c.shape[-1]
    
    # train Y
    trainY[i,:,:] = c.T;

    # train X
    for j in range(numR):

      trainX[i, j, 0:j] = -2*y[0,0:j] + 1;
    #for j in range(T):
      #if (j + 2) % 12 == 0 or (j + 1) % 12 == 0:
        #if j <= numR - 1:
          #trainX[i, j, 0:j+1] = -2*y[0,0:j+1] + 1;
        #else:
          #ind = j - numR + 1
          #trainX[i, j, ind:numR] = -2*y[0,ind:] + 1;
    
  mask = np.array(mask).T
  return trainX, trainY, mask
  
def insert_regular_markers(m, Nc, marker_sequence):

    # Get parameters of the code from the specified marker and Nc.
    Nm = marker_sequence.shape[-1]  # Length of the marker sequence.
    N = Nm + Nc                     # Total codeword block with markers.
    rm = Nc/N                       # Rate of the marker code.
    # If message length does not divide Nc, then pad minimum number of zeros to message bits
    # until message length divides Nc.
    #print(m.shape)
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
