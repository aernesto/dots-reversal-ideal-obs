# Aim of this file is to create a few classes and methods to test
# my ideas (Adrian)

# Main documentation for this file is the Wiki:
# https://github.com/aernesto/dots-reversal-ideal-obs/wiki/Python-classes-and
# -methods

# Comment practice: In this script, we write comments before the targeted
# instruction

import numpy as np


# Overarching class
class Experiment(object):
    def __init__(self, setof_stim_noise, setof_trial_dur, setof_h, tot_trial,
                 outputs='perf_acc_last_cp', states=np.array([-1, 1]),
                 exp_prior=np.array([.5,.5])):
        self.states = states
        self.setof_stim_noise = setof_stim_noise
        self.setof_trial_dur = setof_trial_dur  # for now an integer in msec.
        self.tot_trial = tot_trial
        self.outputs = outputs
        self.setof_h = setof_h
        self.results = []
        self.exp_prior = exp_prior  # TODO: check entries >=0 and sum to 1
        # Corresponds to 25 frames/sec (for stimulus presentation)
        self.exp_dt = 40

    # function that switches the environment state that is given as argument
    def switch(self, H):
        try:
            # might be more elegant to use elseif syntax below
            if H in self.states:
                if H == self.states[0]:
                    return self.states[1]
                else:
                    return self.states[0]
            else:
                raise ValueError("Error in argument H: must be an element of "
                                 "Experiment.states")
        except AttributeError as err:
            print(err.args)

    def launch(self, observer):
        for trial_idx in range(self.tot_trial):
            h = 1
            duration = self.setof_trial_dur
            stim_noise = self.setof_stim_noise
            trial_number = trial_idx
            if np.random.uniform() < self.exp_prior[0]:
                init_state = self.states[0]
            else:
                init_state = self.states[1]
            curr_exp_trial = ExpTrial(self, h, duration, stim_noise,
                                      trial_number, init_state)
            curr_stim = Stimulus(curr_exp_trial)
            curr_obs_trial = ObsTrial(observer, curr_exp_trial, curr_stim)
            curr_obs_trial.infer()
        # curr_exp_trial.save()
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
        self.expt = expt
        self.true_h = h
        self.duration = duration
        self.stim_noise = stim_noise
        self.trial_number = trial_number
        self.init_state = init_state
        self.cp_times = self.gen_cp(self.duration, self.true_h)
        self.end_state = self.compute_endstate(self.cp_times.size)
        self.tot_trial = self.expt.tot_trial

    def compute_endstate(self, ncp):
        # the fact that the last state equals the initial state depends on
        # the evenness of the number of change points.
        if ncp % 2 == 0:
            return self.init_state
        else:
            return self.expt.switch(self.init_state)

    #    def save(self):
    #        print('stimulus is:')
    #        print(self.stim)

    # the following is the likelihood used to generate stimulus values,
    #  given the true state H of the environment
    def lh(self, H):
        # try clause might be redundant (because switch method does it)
        try:
            if H in self.expt.states:
                return np.random.normal(H, self.stim_noise)
            else:
                raise ValueError("Error in argument H: must be an element of "
                                 "Experiment.states")
        except AttributeError as err:
            print(err.args)

    '''
    generates poisson train of duration milliseconds with rate true_h in Hz, 
    using the Gillespie algorithm.
    
    print statements are only there for debugging purposes
    '''
    def gen_cp(self, duration, true_h):
        # TODO: Generate a warning if >1 ch-pt occur in Experiment.exp_dt window
        # print('launching gen_cp')

        # convert duration into seconds.
        secdur = duration / 1000.0
        # print('secdur = '), secdur
        '''
        pre-allocate ten times the mean array size 
        for speed, will be shrinked after computation
        '''
        nEntries = int(np.ceil(10 * true_h * secdur))
        # print('allocated entries = '), nEntries

        t = np.zeros((nEntries, 1))
        totalTime = 0
        eventIdx = -1

        while totalTime < secdur:
            sojournTime = np.random.exponential(1. / true_h)
            totalTime += sojournTime
            eventIdx += 1
            t[eventIdx] = totalTime

        # trim unused nodes, and maybe last event if occurred beyond secdur

        # print t[0:10]
        lastEvent, idxLastEvent = t.max(0), t.argmax(0)
        # print 'lastEvent = ', lastEvent, 'idxLastEvent = ', idxLastEvent

        if lastEvent > secdur:
            idxLastEvent -= 1

        if idxLastEvent == -1:
            t = np.zeros((0, 1))
        else:
            t = t[0:int(idxLastEvent) + 1]

        return t


class Stimulus(object):
    def __init__(self, exp_trial):
        self.exp_trial = exp_trial
        self.trial_number = self.exp_trial.trial_number
        self.stim = self.gen_stim()

    def gen_stim(self):
        binsize = self.exp_trial.expt.exp_dt  # in msec

        # number of bins, i.e. number of stimulus values to compute
        nbins = (self.exp_trial.duration - 1) / binsize

        # stimulus vector to be filled by upcoming while loop
        stimulus = np.zeros((nbins, 1))

        # loop variables
        bin_nb = 1
        last_envt = self.exp_trial.init_state
        cp_idx = 0

        while bin_nb < nbins:
            stim_idx = bin_nb - 1  # index of array entry to fill in

            # check environment state in current bin
            curr_time = (bin_nb - 1) * binsize  # in msec

            if curr_time < self.exp_trial.cp_times[cp_idx]:
                new_envt = last_envt
            else:
                new_envt = self.exp_trial.expt.switch(last_envt)

            # compute likelihood to generate stimulus value
            stimulus[stim_idx] = self.exp_trial.lh(new_envt)

            # update variables for next iteration
            last_envt = new_envt
            cp_idx += 1
            bin_nb += 1

        return stimulus

# Level 2
class IdealObs(object):
    def __init__(self, dt, expt, prior_states=np.array([.5, .5]),
                 prior_h=np.array([1, 1])):
        try:
            if (expt.setof_trial_dur % dt) == 0:
                self.dt = dt  # in msec
            else:
                raise AttributeError("Error in arguments: the observer's time"
                                     "step size "
                                     "'dt' "
                                     "does not divide "
                                     "the trial durations 'setof_trial_dur'")
        except AttributeError as err:
            print(err.args)

        self.prior_h = prior_h
        self.prior_states = prior_states
        self.expt = expt  # reference to Experiment object
        self.obs_noise = self.expt.setof_stim_noise


class ObsTrial(object):
    def __init__(self, observer, exp_trial, stimulus):
        self.observer = observer
        self.exp_trial = exp_trial
        self.stimulus = stimulus
        self.llr = []
        self.decision = 0
        self.obs_noise = self.exp_trial.stim_noise
        self.trial_number = self.exp_trial.trial_number
        self.obs = self.gen_obs()

    def gen_obs(self):
        return self.stimulus.stim

    def infer(self):
        # TODO: import MATLAB code
        self.llr = np.ones(self.exp_trial.duration)
        self.decision = 1


# def save(self):
#        print('observations are:')
#        print(self.observations)


# Test code
Expt = Experiment(setof_stim_noise=1, setof_trial_dur=5, setof_h=1,
                  tot_trial=1)
Observer = IdealObs(dt=1, expt=Expt)
Expt.launch(Observer)
