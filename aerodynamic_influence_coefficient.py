import numpy as np
from vortex_ring import VRTXRING

def aic(num_chord, num_span, num_wake, panel_vortex, panel_normal, panel_collocation, wake_vortex):
    A_ww = np.eye(num_wake*num_span)
    A_wb = np.zeros((num_wake*num_span, num_chord*num_span))
    B_b = np.zeros((num_wake*num_span, num_chord*num_span))
    B_w = np.zeros((num_wake*num_span, num_wake*num_span))

    # Setup Wake Update Matrices
    for j in range(num_span):
        for k in range(num_wake):
            for i in range(num_chord):
                #for j in range(span_panels):
                if k == 0 and i == num_chord-1:
                    B_b[k*num_span+j, i*num_span+j] = 1

    for j in range(num_span):
        for i in range(num_wake):
            for k in range(num_wake):
                if i == 0:
                    pass
                elif ((i*num_span+j) - (k*num_span+j)) == num_span:
                    B_w[i*num_span+j, k*num_span+j] = 1

    

    # Compute ICM from bound vortex (A_b) (CONSTANT)
    A_b = np.zeros((num_chord*num_span, num_chord*num_span))
    for i in range(num_chord):
        for j in range(num_span):
            for a in range(num_chord):
                for b in range(num_span):
                    pt1 = panel_vortex[a,b,0]
                    pt2 = panel_vortex[a,b,1]
                    pt3 = panel_vortex[a,b,2]
                    pt4 = panel_vortex[a,b,3]

                    u_i, v_i, w_i = VRTXRING(panel_collocation[i,j], pt1, pt2, pt3, pt4, 1)
                    influence = np.dot([u_i, v_i, w_i], panel_normal[i,j])
                    A_b[num_span*i+j, num_span*a+b] = influence

    

    # Compute ICM from wake vortex (A_w) (CONSTANT)
    A_w = np.zeros((num_chord*num_span, num_wake*num_span))
    for i in range(num_chord):
        for j in range(num_span):
            for a in range(num_wake):
                for b in range(num_span):
                    pt1 = wake_vortex[a,b,0]
                    pt2 = wake_vortex[a,b,1]
                    pt3 = wake_vortex[a,b,2]
                    pt4 = wake_vortex[a,b,3]

                    u_i, v_i, w_i = VRTXRING(panel_collocation[i,j], pt1, pt2, pt3, pt4, 1)
                    influence = np.dot([u_i, v_i, w_i], panel_normal[i,j])
                    A_w[num_span*i+j, num_span*a+b] = influence

    # Assemble full matrices and Vectors (CONSTANT)
    A_top = np.concatenate((A_b, A_w), axis=1)
    A_bot = np.concatenate((A_wb, A_ww), axis=1)
    A = np.concatenate((A_top, A_bot), axis=0)

    B_bot = np.concatenate((B_b, B_w), axis=1)
    B = np.concatenate((np.zeros((num_chord*num_span,((num_wake+num_chord)*num_span))), B_bot), axis=0)
    
    return A, B