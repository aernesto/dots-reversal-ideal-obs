# Aim of this file is to create a few classes and methods to test
# my ideas (Adrian)

import numpy as np

class Experiment(object):
    def __init__(self, stim_noise, trial_durations,
                 trial_number, outputs = 'perf_acc_last_cp', states = (-1,1)):
        self.states = states
        self.stim_noise = stim_noise  # TODO: extend this to a set of values
        self.trial_durations = trial_durations  # for now an integer in msec.
        self.trial_number = trial_number
        self.outputs = outputs

    def launch(self, observer):
        for trial_idx in range(self.trial_number):
            curr_exp_trial = ExpTrial(self)
            curr_obs_trial = ObsTrial(curr_exp_trial, observer)
#            curr_obs_trial.inference()
            curr_exp_trial.save()
            curr_obs_trial.save()

class ExpTrial(object):
    def __init__(self, expt):
        self.expt = expt
        self.stim = np.ones(self.expt.trial_durations)

    def save(self):
        print('stimulus is:')
        print(self.stim)


#class Stimulus(object):
#    def __init__(self, exp_trial):
#        self.exp_trial = exp_trial

class IdealObs(object):
    def __init__(self, dt, expt, prior_states = (.5,.5), alpha=1, beta=1):
        self.expt = expt  # reference to Experiment object
        self.obs_noise = self.expt.stim_noise
        self.alpha = alpha
        self.beta = beta
        self.dt = dt  # in msec
        # TODO: check that dt divides all possible trial
        self.prior_states = prior_states
        # durations set in the Experiment


class ObsTrial(object):
    def __init__(self, exp_trial, observer):
        self.exp_trial = exp_trial
        self.observer = observer
        self.observations = np.ones(self.exp_trial.expt.trial_durations)

#    def inference(self):
#        return self.observations

    def save(self):
        print('observations are:')
        print(self.observations)

# Test code
#1.
Expt = Experiment(stim_noise=1,trial_durations=5,
                  trial_number=1,outputs='perf_acc_last_cp')
Observer = IdealObs(dt=1, expt=Expt)
Expt.launch(Observer)



