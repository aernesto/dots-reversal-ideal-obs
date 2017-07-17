# Aim of this file is to create a few classes and methods to test
# my ideas (Adrian)

class Experiment(object):  # inheriting from object = new-style Python class
    def __init__(self, states=[-1,1], obs_noise):
        self.states = states
        self.obs_noise = obs_noise

class Observer(Experiment):
    def __init__(self, snr, alpha=1, beta=1, dt):
        self.snr = snr
        self.alpha = alpha
        self.beta = beta
        self.dt = dt  # TODO: check that dt divides all possible trial
        # durations set in the Experiment

class Experimenter(Experiment):
    def __init__(self, h, T):
        self.h = h
        self.T = T

class ExpterTrial(Experimenter):
    def __init__(self):

class ObserverTrial(Observer):


