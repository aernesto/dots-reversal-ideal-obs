# Aim of this file is to create a few classes and methods to test
# my ideas (Adrian)

# Main documentation for this file is the Wiki:
#   https://github.com/aernesto/dots-reversal-ideal-obs/wiki/Python-classes-and-methods
import numpy as np

# Overarching class
class Experiment(object):
    def __init__(self, setof_stim_noise, setof_trial_dur, setof_h, tot_trial,
                 outputs='perf_acc_last_cp', states=np.array([-1, 1])):
        self.states = states
        self.setof_stim_noise = setof_stim_noise
        self.setof_trial_dur = setof_trial_dur  # for now an integer in msec.
        self.tot_trial = tot_trial
        self.outputs = outputs
        self.setof_h = setof_h
        self.results = []

    def launch(self, observer):
        for trial_idx in range(self.trial_number):
            curr_exp_trial = ExpTrial(self)
            curr_obs_trial = ObsTrial(curr_exp_trial, observer)
#            curr_obs_trial.inference()
            curr_exp_trial.save()
            curr_obs_trial.save()

# Corresponds to single trial
class ExpTrial(object):
    def __init__(self, expt):
        self.expt = expt
        self.stim = np.ones(self.expt.setof_trial_dur)

    def save(self):
        print('stimulus is:')
        print(self.stim)


#class Stimulus(object):
#    def __init__(self, exp_trial):
#        self.exp_trial = exp_trial

# Level 2
class IdealObs(object):
    def __init__(self, dt, expt, prior_states=np.array([.5, .5]),
                 prior_h=np.array([1, 1])):
        self.expt = expt  # reference to Experiment object
        self.obs_noise = self.expt.setof_stim_noise
        self.prior_h = prior_h
        self.dt = dt  # in msec
        # TODO: check that dt divides all possible trial
        self.prior_states = prior_states
        # durations set in the Experiment


class ObsTrial(object):
    def __init__(self, exp_trial, observer):
        self.exp_trial = exp_trial
        self.observer = observer
        self.observations = np.ones(self.exp_trial.expt.setof_trial_dur)

#    def inference(self):
#        return self.observations

    def save(self):
        print('observations are:')
        print(self.observations)

# Test code
#1.
Expt = Experiment(setof_stim_noise=1, setof_trial_dur=5, setof_h=1,
                  trial_number=1, outputs='perf_acc_last_cp')
Observer = IdealObs(dt=1, expt=Expt)
Expt.launch(Observer)
