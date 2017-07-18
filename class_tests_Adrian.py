# Aim of this file is to create a few classes and methods to test
# my ideas (Adrian)

class Experiment(object):
    def __init__(self, states = [-1,1], stim_noise, trial_durations,
                 trial_number, outputs = 'perf_acc_last_cp'):
        self.states = states
        self.stim_noise = stim_noise  # TODO: extend this to a set of values
        self.trial_durations = trial_durations  # for now an integer in msec.
        self.trial_number = trial_number
        self.outputs = outputs

    def launch(self):


class ExpTrial(object):
    def __init__(self):

class Stimulus(object):
    def __init__(self):

class IdealObs(object):
    def __init__(self, alpha=1, beta=1, dt, expt, prior_states = [.5,.5]):
        self.obs_noise = expt.stim_noise
        self.alpha = alpha
        self.beta = beta
        self.dt = dt  # in msec
        # TODO: check that dt divides all possible trial
        self.expt = expt  # reference to Experiment object
        self.prior_states = prior_states
        # durations set in the Experiment

class ObsTrial(object):


# Test code
#1.
Expt = Experiment([-1,1],1,2000,1,'perf_acc_last_cp')
Observer = IdealObs(dt=1, expt=Expt)
Expt.launch()



