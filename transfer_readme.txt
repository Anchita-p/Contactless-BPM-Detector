SUPERSEDED -- see README.md
===========================

This file described the project when it was a single-algorithm (POS only)
pipeline with no user interface, paused pending a camera test. All of that has
since been built and the setup instructions here were out of date, so they were
removed rather than left to mislead someone setting up on a new machine.

    Setup, launch and usage:  README.md
    Architecture and every tunable constant:  ARCHITECTURE.md

The one note worth carrying forward, because it is still open:

    The SNR confidence gate (src/bpm_calc.py, SNR_THRESHOLD = 15.0) has never
    been cleared by a real recorded clip on the original development machine.
    data/test_clip.mp4 peaks at 10.7 dB against an 11.76 dB gate -- about 1 dB
    short. The threshold is set above the level pure noise reaches, so lowering
    it to make that clip pass would admit noise as pulse. Confirming the gate on
    a machine with a better camera and steadier lighting is still the outstanding
    validation step, and it is what the Accuracy Check tab exists to settle.
