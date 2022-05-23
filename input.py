from UVLM import UVLM_solver
from UVLM import UVLM_preprocessing

## Geometry ##
chord_length = 1
semi_span = 6
num_chord = 8
num_span = 10
num_wake = 20
sweep = 5   # degrees

## Aerodynamics Properties ##
V_inf = 25
AoA = 0
rho = 1

## Wing kinematics (for pure aerodyanmics only) ##
frequency = 17.5    # rad/s
heave_amplitude = 0.01
pitch_amplitude = 0.01
e_a = 0.7    # pitch axis (fraction of chord)

## Simulation Parameters ##
boundary_condition = 'half' #half or full
time_steps = 157
dt = 0.01


# UVLM_preprocessing(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V, boundary_condition, dt)
UVLM_solver(chord_length, semi_span, num_chord, num_span, num_wake, sweep, V_inf, AoA, rho, frequency, heave_amplitude, pitch_amplitude, e_a, boundary_condition, dt, time_steps)