# Aim of this file is to create a few classes and methods to test
# my ideas (Adrian)

# Main documentation for this file is the Wiki:
# https://github.com/aernesto/dots-reversal-ideal-obs/wiki/Python-classes-and-methods
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
        for trial_idx in range(self.tot_trial):
            h = 1
            duration = self.setof_trial_dur
            stim_noise = self.setof_stim_noise
            trial_number = trial_idx
            init_state = self.states[0]
            curr_exp_trial = ExpTrial(self, h, duration, stim_noise,
                                      trial_number, init_state)
            curr_stim = Stimulus(curr_exp_trial)
            curr_obs_trial = ObsTrial(observer, curr_exp_trial, curr_stim)
            curr_obs_trial.infer()
#            curr_exp_trial.save()
#            curr_obs_trial.save()
        self.save()

    def save(self):
        print('temporary string')  # temporary

    def parallel_launch(self):
        return 0  # temporary


# Corresponds to single trial constants
class ExpTrial(object):
    def __init__(self, expt, h, duration, stim_noise, trial_number, 
                 init_state):
        self.true_h = h
        self.duration = duration
        self.stim_noise = stim_noise
        self.trial_number = trial_number
        self.init_state = init_state
        self.cp_times = self.gen_cp(self.duration, self.true_h)
        self.end_state = self.compute_endstate(self.init_state,
                                               self.cp_times.size)
        self.expt = expt
        self.tot_trial = self.expt.tot_trial

    def compute_endstate(self, init_state, ncp):
        return 1
        
#    def save(self):
#        print('stimulus is:')
#        print(self.stim)

    def gen_cp(self, duration, true_h):
        # TODO: import MATLAB Code
        return np.array([2, 3])


class Stimulus(object):
    def __init__(self, exp_trial):
        self.stim = self.gen_stim()
        self.exp_trial = exp_trial
        self.trial_number = self.exp_trial.trial_number
    
    def gen_stim(self):
        return np.ones(self.exp_trial.duration)


# Level 2
class IdealObs(object):
    def __init__(self, dt, expt, prior_states=np.array([.5, .5]),
                 prior_h=np.array([1, 1])):
        self.prior_h = prior_h
        self.dt = dt  # in msec
        # TODO: check that dt divides all possible trial
        self.prior_states = prior_states
        self.expt = expt  # reference to Experiment object
        self.obs_noise = self.expt.setof_stim_noise


class ObsTrial(object):
    def __init__(self, observer, exp_trial, stimulus):
        self.llr = []
        self.decision = 0
        self.exp_trial = exp_trial
        self.obs_noise = self.exp_trial.stim_noise
        self.trial_number = self.exp_trial.trial_number
        self.obs = self.gen_obs()
        self.observer = observer
        self.stimulus = stimulus
        
    def gen_obs(self):
        return self.stimulus.stim
        
    def infer(self):
        # TODO: import MATLAB code
        self.llr = np.ones(self.exp_trial.duration)
        self.decision = 1

#    def save(self):
#        print('observations are:')
#        print(self.observations)


# Test code
Expt = Experiment(setof_stim_noise=1, setof_trial_dur=5, setof_h=1,
                  tot_trial=1)
Observer = IdealObs(dt=1, expt=Expt)
Expt.launch(Observer)
