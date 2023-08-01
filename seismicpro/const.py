"""Package-level constants"""

from .stacking_velocity import StackingVelocity


# Custom headers optionally added by SeismicPro to those from SEG-Y file
HDR_FIRST_BREAK = 'FirstBreak'
HDR_TRACE_POS = 'TRACE_POS_IN_SURVEY'


# Default stacking velocity for spherical divergence correction and velocity spectrum calculation.
# Estimated as the mean stacking velocity among variety of surveys.
DEFAULT_STACKING_VELOCITY = StackingVelocity(
    times      = [0.,  600., 1200., 1800., 2400., 3000., 3600., 4200., 4800., 5400., 6000.], # ms
    velocities = [1594.3, 1844.5, 2213.6, 2634.5, 3102.3, 3741. , 4068.2, 4245.7, 4367.2, 4479.7, 4559.5] # m/s
)
