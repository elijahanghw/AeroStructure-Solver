import numpy as np
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.linalg import eig

from aerodynamics_mesh import Wing
from aerodynamics_mesh import Wake
from vortex_ring import VRTXRING

def UVLM_solver(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_inf, AoA, rho, frequency, heave_amplitude, pitch_amplitude, e_a, boundary_condition, dt, time_steps):
    print('Setting up AIC matrices...')
    # Freestream Conditions
    AoA = AoA*np.pi/180
    
    # Aerodynamics panels
    delta_c = chord_length/num_chord
    delta_b = semi_span/num_span
    
    wing = Wing(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_inf, dt)
    panel_vortex, panel_collocation, panel_normal, panel_quarterchord = wing.bound_vortex_ring()
    
    wake = Wake(chord_length, semi_span, num_span, num_wake, sweep, V_inf, dt)
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
                    
                    if boundary_condition == 'half':
                        sym_collocation = panel_collocation[i,j].copy()
                        sym_collocation[1] = -sym_collocation[1]
                        u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
                        
                    elif boundary_condition == 'full':
                        u_ii, v_ii, w_ii = 0, 0, 0
                    
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
                    
                    if boundary_condition == 'half':
                        sym_collocation = panel_collocation[i,j].copy()
                        sym_collocation[1] = -sym_collocation[1]
                        u_ii, v_ii, w_ii = VRTXRING(sym_collocation, pt1, pt2, pt3, pt4, 1)
                        
                    elif boundary_condition == 'full':
                        u_ii, v_ii, w_ii = 0, 0, 0    
                    
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
    
    # Time marching solution
    print('Starting time marching solution...')
    CL = []
    time = []
    # Wake update matrices and vectors
    Gamma_old = np.zeros(((num_chord+num_wake)*num_span))
    pitch_axis = e_a * chord_length
    for T in range(time_steps):
        # Compute downwash velocity due to wing motion
        downwash = np.zeros(num_chord*num_span)
        
        theta_dot = [0, pitch_amplitude*frequency*np.cos(frequency*(T+1)*dt), 0]
        theta = pitch_amplitude*np.sin(frequency*(T+1)*dt)
        
        theta_rotation = np.array([[np.cos(theta), 0, -np.sin(theta)],
                              [0, 1, 0],
                              [np.sin(theta), 0, np.cos(theta)]])
        V_theta = -np.matmul(theta_rotation, [V_inf, 0, 0])
        
        V_h = [0, 0, heave_amplitude*frequency*np.cos(frequency*(T+1)*dt)]

        for i in range(num_chord):
            for j in range(num_span):
                OmegaxR = np.cross(theta_dot, panel_collocation[i,j]-pitch_axis)
                U = (V_theta[0] + OmegaxR[0] + V_h[0])
                V = (V_theta[1] + OmegaxR[1] + V_h[1])
                W = (V_theta[2] + OmegaxR[2] + V_h[2])
                
                downwash[num_span*i+j] = np.dot([U, V, W], panel_normal[i,j])
                
        # Compute Gamma
        Downwash = np.concatenate((downwash, np.zeros(num_wake*num_span)), axis=0)
        
        RHS = np.matmul(B,Gamma_old) + Downwash

        Gamma_new = np.matmul(inv(A),RHS)
        Gamma_b_new = Gamma_new[:num_chord*num_span]
        
        Gamma_b_old = Gamma_old[:num_chord*num_span]
        
        tau_i = np.array([1,0,0])
        tau_j = np.array([np.sin(sweep/180*np.pi),np.cos(sweep/180*np.pi),0])

        # Compute forces
        Phi_1 = rho*V_inf/delta_c + rho*V_inf*np.sin(sweep/180*np.pi)/delta_b + rho/dt
        Phi_2 = -rho*V_inf/delta_c
        Phi_3 = -rho*V_inf*np.sin(sweep/180*np.pi)/delta_b
        Phi_4 = -rho/dt
        area = delta_c * delta_b
        
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
        
        delta_L = np.matmul(Phi_new, Gamma_b_new) + np.matmul(Phi_old, Gamma_b_old)
        
        Lift = sum(delta_L)
        
        cl = Lift/(0.5*V_inf**2*rho*chord_length*semi_span)
        CL.append(cl)
        time.append((T+1)*dt)
        
        Gamma_old = Gamma_new
        
    ## Theordosens function (For validation) 
    # k = frequency*chord_length/(2*V_inf)

    # C = 1 - (0.165/(complex(1,-0.0455/k))) - (0.335/(complex(1, -0.3/k)))
    # C_mag = abs(C)
    # C_phase = np.angle(C)

    # CL_theodorsen = []
    # for T in range(time_steps):
    #     alpha = pitch_amplitude*np.sin(frequency*(T+1)*dt)
    #     alpha_dot = pitch_amplitude*frequency*np.cos(frequency*(T+1)*dt)
    #     alpha_ddot = -pitch_amplitude*frequency**2*np.sin(frequency*(T+1)*dt)
    #     h_ddot = -heave_amplitude*frequency**2*np.sin(frequency*(T+1)*dt)
        
    #     alpha_c = pitch_amplitude*np.sin(frequency*(T+1)*dt + C_phase)
    #     alpha_dot_c = pitch_amplitude*frequency*np.cos(frequency*(T+1)*dt + C_phase)
    #     h_dot_c = heave_amplitude*frequency*np.cos(frequency*(T+1)*dt + C_phase)
        
    #     L_nc = rho*np.pi*chord_length**2/4*(V_inf*alpha_dot - (e_a*chord_length-chord_length/2)*alpha_ddot - h_ddot)
    #     L_c = 2*np.pi*rho*chord_length/2*V_inf*C_mag*(V_inf*alpha_c - ((e_a*chord_length-chord_length/2)-chord_length/4)*alpha_dot_c - h_dot_c)
        
    #     Lift_theodorsen = L_nc + L_c
    #     CL_theodorsen.append(Lift_theodorsen/(0.5*V_inf**2*rho*chord_length))
        
    plt.plot(time, CL)
    # plt.plot(time, CL_theodorsen, color='r')
    # plt.legend(('UVLM', 'Theodorsen'), loc='upper right')
    # plt.xlabel(r'$\omega t$')
    # plt.ylabel(r'$C_l$')
    plt.show()
        
    return


def UVLM_preprocessing(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V, boundary_condition, dt):
    # Define Wing
    wing = Wing(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V, dt)
    panel_vortex, panel_collocation, panel_normal, panel_quarterchord = wing.bound_vortex_ring()
    
    # Define Flat Wake
    wake = Wake(chord_length, semi_span, num_span, num_wake, sweep, V, dt)
    wake_vortex = wake.wake_vortex_ring()
    
    print(wake_vortex)
    
    ## Plotting geometry for visualization
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    for i in range(num_chord+1):
        plt.plot(wing.geometry_x[i,:], wing.geometry_y[i,:], color='k')
        plt.plot(wing.bound_vortex_x[i,:], wing.bound_vortex_y[i,:], color='r')
    
    for j in range(num_span+1):
        plt.plot(wing.geometry_x[:,j], wing.geometry_y[:,j], color='k')
        plt.plot(wing.bound_vortex_x[:,j], wing.bound_vortex_y[:,j], color='r')
        plt.plot(wake.wake_vortex_x[:,j], wake.wake_vortex_y[:,j], color='r', linestyle='--')
        
        
    for row in panel_collocation:
        for col in row:
            plt.scatter(col[0], col[1], color='r', marker='x')
            
    for k in range(num_wake+1):
        plt.plot(wake.wake_vortex_x[k,:], wake.wake_vortex_y[k,:], color='r', linestyle='--')
            
        
    plt.xlim(-1,(chord_length+semi_span*np.tan(sweep/180*np.pi)+V*dt)*3)    
    plt.ylim(-1,semi_span*1.1)
    
    plt.xlabel('chord')
    plt.ylabel('span') 
    
    plt.legend(['Wing', 'Vortex', 'Wake'])    
    plt.show()
    
    return