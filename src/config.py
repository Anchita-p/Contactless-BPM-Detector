"""Pipeline timing constants shared by the tracker (which sizes its rolling
buffer) and the signal pipeline (which slices windows out of it)."""

TARGET_FS = 30.0            # fixed FFT/filter sampling rate
WINDOW_DURATION_S = 10.0    # sliding analysis window length in seconds

# Trailing-mean length for the RGB temporal normalization. Dividing by a
# rolling mean of length W is a moving-average high-pass whose corner must sit
# below the 0.7 Hz band edge or it eats the low end of the pulse band; 3 s
# leaves the band intact. It must also stay shorter than WINDOW_DURATION_S,
# otherwise it degenerates into the whole-window normalization POS already
# performs internally and becomes a no-op.
NORM_WINDOW_S = 3.0

# The buffer holds the analysis window plus the normalization lead-in, so
# every sample in the emitted window has a full trailing mean behind it.
BUFFER_DURATION_S = WINDOW_DURATION_S + NORM_WINDOW_S
