from UVLM import UVLM_solver
from UVLM import UVLM_preprocessing
from classical_flutter import classical_flutter

## Geometry ##
chord_length = 1.8228
semi_span = 6.096
num_chord = 8
num_span = 16
num_wake = 5
sweep = 10   # degrees

## UVLM Parameters ##
# V_inf = 25
# AoA = 0
# rho = 1
# frequency = 17.5    # rad/s
# heave_amplitude = 0.01
# pitch_amplitude = 0.01
# e_a = 0.7    # pitch axis (fraction of chord)
# boundary_condition = 'half' #half or full

## Flutter Simulation Parameters ##
V_start = 100
V_end = 180
AoA = 0
rho = 1

mu = 35.71
EI = 9.77e6
GJ = 0.99e6
I_0 = 8.64
CG = 0.43*chord_length
e_a = 0.33*chord_length
e_m = e_a - CG
alpha = 0.001



## General Simulation Parameters ##
time_steps = 157
dt = 0.01


# UVLM_preprocessing(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V, boundary_condition, dt)
#UVLM_solver(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_inf, AoA, rho, frequency, heave_amplitude, pitch_amplitude, e_a, boundary_condition, dt, time_steps)
classical_flutter(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_start, V_end, AoA, rho, mu, EI, GJ, I_0, e_m, e_a, dt, alpha)