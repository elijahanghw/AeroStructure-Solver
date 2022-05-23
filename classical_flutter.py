import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eig

from aerodynamics_mesh import Wing
from aerodynamics_mesh import Wake
from vortex_ring import VRTXRING

def classical_flutter(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_start, V_end, AoA, rho, mu, EI, GJ, I_0, e_m, e_a, dt, alpha):
    ## Structural Dynamics ##
    print("Setting up structural matrices...")
    nodes = np.linspace(0, semi_span, num_span+1)
    L_e = semi_span/np.cos(sweep/180*np.pi)/num_span
    
    # Define Element Matrices
    K_e = np.array([[12*EI/(L_e**3), 6*EI/(L_e**2), 0, -12*EI/(L_e**3), 6*EI/(L_e**2), 0],
                    [6*EI/(L_e**2), 4*EI/L_e, 0, -6*EI/(L_e**2), 2*EI/L_e, 0],
                    [0, 0, GJ/L_e, 0, 0, -GJ/L_e],
                    [-12*EI/(L_e**3), -6*EI/(L_e**2), 0, 12*EI/(L_e**3), -6*EI/(L_e**2), 0],
                    [6*EI/(L_e**2), 2*EI/L_e, 0, -6*EI/(L_e**2), 4*EI/L_e, 0],
                    [0, 0, -GJ/L_e, 0, 0, GJ/L_e]])

    M_e = mu*L_e/420*np.array([[156, 22*L_e, 0, 54, -13*L_e, 0],
                            [22*L_e, 4*L_e**2, 0, 13*L_e, -3*L_e**2, 0],
                            [0, 0, 140*I_0/mu, 0, 0, 70*I_0/mu],
                            [54, 13*L_e, 0, 156, -22*L_e, 0],
                            [-13*L_e, -3*L_e**2, 0, -22*L_e, 4*L_e**2, 0],
                            [0, 0, 70*I_0/mu, 0, 0, 140*I_0/mu]])

    S_e = mu*L_e*e_m/60*np.array([[0, 0, 21, 0, 0, 9],
                                [0, 0, 3*L_e, 0, 0, 2*L_e],
                                [21, 3*L_e, 0, 9, -2*L_e, 0],
                                [0, 0, 9, 0, 0, 21],
                                [0, 0, -2*L_e, 0, 0, -3*L_e],
                                [9, 2*L_e, 0, 21, -3*L_e, 0]])

    M_e = M_e + S_e
    
    # Assemble Global Matrices
    K_global = np.zeros((3*len(nodes), 3*len(nodes)))
    M_global = np.zeros((3*len(nodes), 3*len(nodes)))

    for element in range(num_span):
        r, c = element*3, element*3
        K_global[r:r+K_e.shape[0], c:c+K_e.shape[1]] += K_e
        M_global[r:r+M_e.shape[0], c:c+M_e.shape[1]] += M_e


    # Additional matrix for Newmark-beta Method
    zero_matrix = np.zeros((3*len(nodes), 3*len(nodes)))
    identity = np.eye(3*len(nodes))

    # Clamped-Free BC
    K_global = K_global[3:, 3:]
    M_global = M_global[3:, 3:]
    zero_matrix = zero_matrix[3:, 3:]
    identity = identity[3:, 3:]

    # Newmark Beta Method (Avg Acceleration)
    phi = 0.5 + alpha
    beta = 0.25*(phi+0.5)**2

    S_1 = M_global + dt**2*beta*K_global
    S_2 = dt**2*(0.5-beta)*K_global

    N_11 = np.concatenate((zero_matrix, zero_matrix, S_1), axis=1)
    N_12 = np.concatenate((zero_matrix, -identity, dt*phi*identity), axis=1)
    N_13 = np.concatenate((-identity, zero_matrix, dt**2*beta*identity), axis=1)

    N_1 = np.concatenate((N_11, N_12, N_13), axis = 0)

    N_21 = np.concatenate((K_global, K_global*dt, S_2), axis=1)
    N_22 = np.concatenate((zero_matrix, identity, (1-phi)*dt*identity), axis=1)
    N_23 = np.concatenate((identity, dt*identity, (0.5-beta)*dt**2*identity), axis=1)

    N_2 = np.concatenate((N_21, N_22, N_23), axis = 0)

    print("Done.")
    
    ########## AERODYNAMICS ##########
    print('Setting up AIC matrices...')
    # Freestream Conditions
    AoA = AoA*np.pi/180
    
    # Aerodynamics panels
    delta_c = chord_length/num_chord
    delta_b = semi_span/num_span
    
    wing = Wing(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_start, dt)
    panel_vortex, panel_collocation, panel_normal, panel_quarterchord = wing.bound_vortex_ring()
    
    wake = Wake(chord_length, semi_span, num_span, num_wake, sweep, V_start, dt)
    wake_vortex = wake.wake_vortex_ring()
    
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
                    sym_collocation = panel_collocation[i,j].copy()
                    sym_collocation[1] = -sym_collocation[1]
                    u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
                    
                    influence = np.dot([u_i+u_ii, v_i-v_ii, w_i+w_ii], panel_normal[i,j])
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
                    sym_collocation = panel_collocation[i,j].copy()
                    sym_collocation[1] = -sym_collocation[1]
                    u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)   
                    
                    influence = np.dot([u_i+u_ii, v_i-v_ii, w_i+w_ii], panel_normal[i,j])
                    A_w[num_span*i+j, num_span*a+b] = influence

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
                    
    # Assemble full matrices and Vectors (CONSTANT)
    A_top = np.concatenate((A_b, A_w), axis=1)
    A_bot = np.concatenate((A_wb, A_ww), axis=1)
    A = np.concatenate((A_top, A_bot), axis=0)

    B_bot = np.concatenate((B_b, B_w), axis=1)
    B = np.concatenate((np.zeros((num_chord*num_span,((num_wake+num_chord)*num_span))), B_bot), axis=0)
    
    print('Done')
    
    ########## AERO-STRUCTURE COUPLING ##########
    print("Starting FSI analysis...")
    flutter_flag = 0
    flutter_speed = 0
    for V_inf in range(V_start,V_end+1, 2):
        # Location of Elastic Axis
        elastic_collocation = panel_collocation.copy()
        for i in range(num_chord):
            for j in range(num_span):
                elastic_collocation[i,j,0] = elastic_collocation[i,j,0] - (e_a + elastic_collocation[i,j,1]*np.tan(sweep/180*np.pi))
                
        # Heave downwash mapping
        T_zeros = np.zeros((num_chord*num_span, 3*num_span))
        T_1 = np.zeros((num_chord*num_span, 3*num_span))
        T_wake = np.zeros((num_wake*num_span, 9*num_span))

        for i in range(num_chord):
            for j in range(num_span):
                if j == 0:
                    # Heave downwash
                    T_1[i*num_span, 0] = 0.5
                else:
                    T_1[(i*num_span)+j, j*3-3] = 0.5
                    T_1[(i*num_span)+j, j*3] = 0.5

        T_1 = np.concatenate((T_zeros, T_1, T_zeros), axis=1)
        T_heave = np.concatenate((T_1, T_wake), axis=0)

        # Pitch velocity mapping
        T_2 = np.zeros((num_chord*num_span, 3*num_span))

        for i in range(num_chord):
            for j in range(num_span):
                if j == 0:
                    T_2[i*num_span, 1] = 0.5*np.sin(sweep*np.pi/180) * -elastic_collocation[i,j,0]
                    T_2[i*num_span, 2] = 0.5*np.cos(sweep*np.pi/180) * -elastic_collocation[i,j,0]
                else:
                    T_2[(i*num_span)+j, j*3-2] = 0.5*np.sin(sweep*np.pi/180) * -elastic_collocation[i,j,0]
                    T_2[(i*num_span)+j, j*3+1] = 0.5*np.sin(sweep*np.pi/180) * -elastic_collocation[i,j,0]
                    
                    T_2[(i*num_span)+j, j*3-1] = 0.5*np.cos(sweep*np.pi/180) * -elastic_collocation[i,j,0]
                    T_2[(i*num_span)+j, j*3+2] = 0.5*np.cos(sweep*np.pi/180) * -elastic_collocation[i,j,0]

        T_2 = np.concatenate((T_zeros, T_2, T_zeros), axis=1)
        T_pitch = np.concatenate((T_2, T_wake), axis=0)

        # Pitch angle downwash mapping
        T_3 = np.zeros((num_chord*num_span, 3*num_span))

        for i in range(num_chord):
            for j in range(num_span):
                if j == 0:
                    T_3[i*num_span, 1] = 0.5*np.sin(sweep*np.pi/180)
                    T_3[i*num_span, 2] = 0.5*np.cos(sweep*np.pi/180)
                else:
                    T_3[(i*num_span)+j, j*3-2] = 0.5*np.sin(sweep*np.pi/180)
                    T_3[(i*num_span)+j, j*3+1] = 0.5*np.sin(sweep*np.pi/180)
                    
                    T_3[(i*num_span)+j, j*3-1] = 0.5*np.cos(sweep*np.pi/180)
                    T_3[(i*num_span)+j, j*3+2] = 0.5*np.cos(sweep*np.pi/180)

        T_3 = np.concatenate((T_3, T_zeros, T_zeros), axis=1)
        T_aoa = -V_inf * np.concatenate((T_3, T_wake), axis=0)

        # Full Downwash Mapping
        T = T_heave + T_pitch + T_aoa
        
        # Compute forces
        Phi_1 = rho*V_inf/delta_c + rho*V_inf*np.sin(sweep/180*np.pi)/delta_b + rho/dt
        Phi_2 = -rho*V_inf/delta_c
        Phi_3 = -rho*V_inf*np.sin(sweep/180*np.pi)/delta_b
        Phi_4 = -rho/dt
        area = delta_c * delta_b
        
        elastic_quarterchord = panel_quarterchord.copy()
        for i in range(num_chord):
            for j in range(num_span):
                elastic_quarterchord[i,j,0] = elastic_quarterchord[i,j,0] - (e_a + elastic_quarterchord[i,j,1]*np.tan(sweep/180*np.pi))
        
        Phi_new = Phi_1*np.eye((num_span*num_chord))
        Phi_old = Phi_4*np.eye((num_span*num_chord))

        for i in range(num_span*num_chord):
            if i >= num_span:
                Phi_new[i, i-num_span] = Phi_2
                
        for j in range(num_span*num_chord):
            if j%num_span != 0:
                Phi_new[j,j-1] = Phi_3
                
        Phi_new = Phi_new*area
        Phi_old = Phi_old*area
        
        force_mapping = np.zeros((3*num_span, num_span*num_chord))

        for i in range(num_span):
            for j in range(num_chord):
                force_mapping[3*i,i+j*num_span] = 0.5
                force_mapping[3*i+1,i+j*num_span] = L_e/8    
                force_mapping[3*i+2,i+j*num_span] = 0.5 * -elastic_quarterchord[j,i,0]

                if i < num_span-1:
                    force_mapping[3*i,i+j*num_span+1] = 0.5
                    force_mapping[3*i+1,i+j*num_span+1] = L_e/8     
                    force_mapping[3*i+2,i+j*num_span+1] = 0.5 * -elastic_quarterchord[j,i+1,0]

        Psi_1 = np.matmul(force_mapping, Phi_new)
        Psi_2 = np.matmul(force_mapping, Phi_old)

        Psi_1 = np.concatenate((Psi_1, np.zeros((3*num_span, num_span*num_wake))),axis=1)
        Psi_2 = np.concatenate((Psi_2, np.zeros((3*num_span, num_span*num_wake))),axis=1)

        Psi_1 = np.concatenate((Psi_1, np.zeros((2*3*num_span, (num_chord+num_wake)*num_span))), axis=0)
        Psi_2 = np.concatenate((Psi_2, np.zeros((2*3*num_span, (num_chord+num_wake)*num_span))), axis=0)


        Matrix_11 = np.concatenate((A, -T), axis=1)
        Matrix_12 = np.concatenate((-Psi_1, N_1), axis=1)
        Matrix_1 = np.concatenate((Matrix_11, Matrix_12), axis=0)

        Matrix_21 = np.concatenate((B, np.zeros(((num_chord+num_wake)*num_span, 3*3*num_span))), axis=1)
        Matrix_22 = np.concatenate((Psi_2, -N_2), axis=1)
        Matrix_2 = np.concatenate((Matrix_21, Matrix_22), axis=0)

        # External Downwash Mapping
        W = -V_inf*np.sin(AoA)*np.ones(num_span*num_chord)
        W_zeros = np.zeros(num_span*num_wake+num_span*3*3)
        W = np.concatenate((W, W_zeros), axis=0)
        
        # Eigenvalue Problem
        Matrix_eigen = np.matmul(inv(Matrix_1), Matrix_2)
        w, v = eig(Matrix_eigen)

        # Convert discrete to continuous eigenvalues
        s = np.log(w)/dt
        s_real = np.real(s)
        s_im = np.imag(s)

        p = s_im.argsort()
        s_im = s_im[p]
        s_real = s_real[p]
        
        speed = V_inf * np.ones_like(s_real)

        print("Velocity: " + str(V_inf) + " m/s" + "\t" + "max eigenvalue real: " + str(max(s_real)))

        if flutter_flag == 0 and max(s_real) > 0:
            flutter_flag = 1
            flutter_speed = V_inf
            print("Flutter occured.")

        if V_inf == V_start:
            plt.scatter(s_real, s_im, color='r', marker='x')
        elif V_inf == V_end:
            plt.scatter(s_real, s_im, color='b', marker='x')
        else:
            plt.scatter(s_real, s_im, color='k', marker='x')
        
    plt.xlabel("Real(s)")
    plt.ylabel("Im(s)")
    plt.axvline(0, color='grey', linestyle="--")
    
    print("Plotting eigenvalues")
    if flutter_flag == 1:
        print("Flutter speed: " + str(flutter_speed) + "m/s")
    elif flutter_flag == 0:
        print("No flutter.")
    print("Simulation complete.")

    plt.show()

    
    return