import numpy as np
from numpy.linalg import norm

class Wing():
    def __init__(self, chord_length, semi_span, num_chord, num_span, num_wake, sweep, V, dt):
        # Define Basic Variables
        self.chord_length = chord_length
        self.semi_span = semi_span
        self.num_chord = num_chord
        self.num_span = num_span
        self.num_wake = num_wake
        self.sweep = sweep/180*np.pi
        self.V = V
        self.dt = dt
        self.panel_chord = chord_length/num_chord
        self.panel_span = semi_span/num_span

        # Define vertices of bound vortex ring wrt. flat plate geometry (x -> chord, y -> span)
        self.geometry_x = np.zeros((num_chord+1, num_span+1))
        self.geometry_y = np.zeros((num_chord+1, num_span+1))
        self.bound_vortex_x = np.zeros((num_chord+1, num_span+1))
        self.bound_vortex_y = np.zeros((num_chord+1, num_span+1))
        for i in range(num_chord+1):
            for j in range(num_span+1):
                self.geometry_y[i,j] = j*self.semi_span/self.num_span
                self.geometry_x[i,j] = i*self.chord_length/self.num_chord + j*self.semi_span/self.num_span*np.tan(self.sweep)
                
                self.bound_vortex_y[i,j] = j*self.semi_span/self.num_span
                if i == num_chord:
                    self.bound_vortex_x[i,j] = i*self.chord_length/self.num_chord + j*self.semi_span/self.num_span*np.tan(self.sweep) + V*dt

                else:    
                    self.bound_vortex_x[i,j] = i*self.chord_length/self.num_chord + j*self.semi_span/self.num_span*np.tan(self.sweep) + 1/4*self.panel_chord

    def bound_vortex_ring(self):
        panel_vortex = []
        panel_collocation = []
        panel_normal = []
        panel_quarterchord = []
        for i in range(self.num_chord):
            for j in range(self.num_span):
                # Computation of vortex ring points for panel i,j
                pt1 = [self.bound_vortex_x[i,j], self.bound_vortex_y[i,j], 0]
                pt2 = [self.bound_vortex_x[i,j+1], self.bound_vortex_y[i,j+1], 0]
                pt3 = [self.bound_vortex_x[i+1,j+1], self.bound_vortex_y[i+1,j+1], 0]
                pt4 = [self.bound_vortex_x[i+1,j], self.bound_vortex_y[i+1,j], 0]

                panel_vortex.append([pt1, pt2, pt3, pt4])

                # Computation of collocation point
                collocation_pt = [i*self.chord_length/self.num_chord + (j*self.semi_span/self.num_span + 1/2*self.panel_span)*np.tan(self.sweep) + 3/4*self.panel_chord, j*self.semi_span/self.num_span + 1/2*self.panel_span , 0]
                panel_collocation.append(collocation_pt)

                # Computation of normal vector
                A_vector = np.array(pt3) - np.array(pt1)
                B_vector = np.array(pt2) - np.array(pt4)
                AxB = np.cross(A_vector,B_vector)
                panel_normal.append(AxB/norm(AxB))

                # # Computation of quarter chord
                quarter_chord = [i*self.chord_length/self.num_chord + (j*self.semi_span/self.num_span + 1/2*self.panel_span)*np.tan(self.sweep) + 1/4*self.panel_chord, j*self.semi_span/self.num_span + 1/2*self.panel_span , 0]
                panel_quarterchord.append(quarter_chord)

        panel_vortex = np.array(panel_vortex)
        panel_vortex = np.reshape(panel_vortex, (self.num_chord,self.num_span,4,3))

        panel_collocation = np.array(panel_collocation)
        panel_collocation = np.reshape(panel_collocation, (self.num_chord,self.num_span,3))

        panel_normal = np.array(panel_normal)
        panel_normal = np.reshape(panel_normal, (self.num_chord,self.num_span,3))

        panel_quarterchord = np.array(panel_quarterchord)
        panel_quarterchord = np.reshape(panel_quarterchord, (self.num_chord,self.num_span,3))
        
        return panel_vortex, panel_collocation, panel_normal, panel_quarterchord

class Wake():
    def __init__(self, chord_length, semi_span, num_span, num_wake, sweep, V, dt):
        # Define Basic Variables
        self.chord_length = chord_length
        self.semi_span = semi_span
        self.num_span = num_span
        self.num_wake = num_wake
        self.sweep = sweep*np.pi/180
        self.V = V
        self.dt = dt
        self.panel_span = self.semi_span/self.num_span

        # Define vertices of wake vortex ring wrt. flat plate geometry (x -> chord, y -> span)
        self.wake_vortex_x = np.zeros((num_wake+1, num_span+1))
        self.wake_vortex_y = np.zeros((num_wake+1, num_span+1))
        for k in range(num_wake+1):
            for j in range(num_span+1):
                self.wake_vortex_x[k,j] = self.chord_length + j*self.semi_span/self.num_span*np.tan(self.sweep) + (k+1)*V*dt
                self.wake_vortex_y[k,j] = j*self.semi_span/self.num_span

    def wake_vortex_ring(self):
        wake_vortex = []
        for k in range(self.num_wake):
            for j in range(self.num_span):
                # Computation of vortex ring points for panel i,j
                pt1 = [self.wake_vortex_x[k,j], self.wake_vortex_y[k,j], 0]
                pt2 = [self.wake_vortex_x[k,j+1], self.wake_vortex_y[k,j+1], 0]
                pt3 = [self.wake_vortex_x[k+1,j+1], self.wake_vortex_y[k+1,j+1], 0]
                pt4 = [self.wake_vortex_x[k+1,j], self.wake_vortex_y[k+1,j], 0]

                wake_vortex.append([pt1, pt2, pt3, pt4])

        wake_vortex = np.array(wake_vortex)
        wake_vortex = np.reshape(wake_vortex, (self.num_wake,self.num_span,4,3))
        return wake_vortex








