import matplotlib.pyplot as plt
import numpy as np
import scipy
import datetime
import dataset

plt.rcdefaults()

# Debug mode
debug = True


def printdebug(debugmode, string=None, vartuple=None):
    """
    prints string, varname and var for debug purposes
    :param debugmode: True or False
    :param string: Custom message useful for debugging
    :param vartuple: Tuple (varname, var), where:
        :varname: string representing name of variable to display
        :var: actual Python variable to print on screen
    :return:
    """
    if debugmode:
        print('-------------------------')
        if string is None:
            pass
        else:
            print(string)
        if vartuple is None:
            pass
        else:
            print(vartuple[0], '=', vartuple[1])
        print('-------------------------')


def gen_cp_discrete(duration, true_h):
    """
    Generates the change point times for a discrete time environment
    If H_1 = H_plus and H_2 = H_minus, then t=2 is a change point time
    :param duration: an integer, representing the number of time steps of the trial.
            The observer will make duration observations in a single trial.
    :param true_h: true hazard rate. It is a probability between 0 and 1, since it is the transition probability
            of the discrete-time Markov Chain H_n.
    :return: Numpy array of (numpy) integers
    """
    cp_times = []
    t = 1  # time step
    while t <= duration:
        if np.random.uniform() < true_h:
            cp_times += [t]
        t += 1
    return np.array(cp_times)


class Experiment(object):
    """
    Overarching class for our simulations. Contains all the necessary meta-data, like
    the number of trials per condition, and the fact that the stimulus will be modeled
    in discrete or continuous time.
    """

    def __init__(self, setof_stim_noise, exp_dt, setof_trial_dur, setof_h,
                 tot_trial, states=np.array([-1, 1]),
                 exp_prior=np.array([.5, .5]), discrete_time=True):
        self.states = states
        self.setof_stim_noise = setof_stim_noise
        self.setof_trial_dur = setof_trial_dur  # for now an integer in msec.
        self.tot_trial = tot_trial
        #         self.outputs = outputs
        self.setof_h = setof_h
        self.results = []
        self.exp_prior = exp_prior  # TODO: check that entries >=0 and sum to 1
        self.discrete_time = discrete_time
        if not self.discrete_time:
            # exp_dt = 40 msec corresponds to 25 frames/sec (for stimulus presentation)
            try:
                # check that exp_dt divides all the trial durations
                if ((self.setof_trial_dur % exp_dt) != 0).sum() == 0:
                    self.exp_dt = exp_dt  # in msec - or in time unit for discrete time
                else:
                    raise AttributeError("Error in arguments: the Experiment's time"
                                         "step size "
                                         "'exp_dt' "
                                         "does not divide "
                                         "the trial durations 'setof_trial_dur'")
            except AttributeError as err:
                print(err.args)

    # function that switches the environment state that is given as argument
    def switch(self, envstate):
        try:
            # might be more elegant to use elseif syntax below
            if envstate in self.states:
                if envstate == self.states[0]:
                    return self.states[1]
                else:
                    return self.states[0]
            else:
                raise ValueError("Error in argument H: must be an element of "
                                 "Experiment.states")
        except AttributeError as err:
            print(err.args)

    def launch(self, **kwargs):
        """
        This is the main method of the object Experiment. It loops over conditions and trials (creating them),
        and performs the inference process with the ObsTrial.infer() method.
        :param observern: IdealObs object representing the ideal observer model who performs the task
        :return:
        """
        if

        # Start exhaustive loop on parameters
        global_nb = 1  # counter for output to the user via console
        for h in self.setof_h:
            for duration in self.setof_trial_dur:
                for stim_noise in self.setof_stim_noise:
                    for trial_idx in range(self.tot_trial):
                        # Debugging strings
                        if global_nb % 1000 == 0:
                            printdebug(debugmode=debug, string="entering trial",
                                       vartuple=('', global_nb))
                        global_nb += 1
                        trial_number = trial_idx
                        '''
                        We now define the different signaling conditions over which we will loop
                        We create a dict named signal_conditions with two keys:
                        :key state_signal: value = tuple (signal, reliability), where
                            :signal: Boolean indicating presence or absence of signal about state at the end of the trial
                            :reliability: Float between 0 and 1 indicating the reliability of the signal
                        :key cp_signal: value = tuple (signal, reliability), where
                            :signal: Boolean indicating presence or absence of signals about each change point time
                            :reliability: tuple (p_change, p_nochange), where
                                :p_change: float between 0 and 1 indicating reliability for presence of change point
                                :p_nochange: float between 0 and 1 indicating reliability for absence of change point
                        
                        '''
                        signal_conditions = {'state_signal':(),
                                             'cp_signal':()}
                        for signal in ['no_signal','state_signal','cp_signal','both signals']:
                            if signal == 'no_signal':

                            state_signal, cp_signal =
                            # select initial true environment state for current trial
                            if np.random.uniform() < self.exp_prior[0]:
                                init_state = self.states[0]
                            else:
                                init_state = self.states[1]
                            printdebug(debugmode=not debug, string="about to create ExpTrial object")

                            # set rng to random seed
                            np.random.seed(None)
                            # generate random seed, but force it to be an integer (easier to save)
                            seed = np.random.randint(1000000000000, dtype=int)

                            # create the Stimulus object
                            printdebug(debugmode=not debug, string="about to create Stimulus object")
                            curr_stim = Stimulus(self, state_signal, cp_signal, h, duration, stim_noise,
                                                 trial_number, init_state, seed=seed)

                            # create ObsTrial object
                            printdebug(debugmode=not debug, string="about to create ObsTrial object")
                            curr_obs_trial = ObsTrial(curr_stim,
                                                      observer.prior_states, observer.prior_h)
                            printdebug(debugmode=not debug, string="about to launch infer method")
                            curr_obs_trial.infer(save2db=True)


class ExpTrial(object):
    def __init__(self, expt, h, duration, stim_noise, trial_number,
                 init_state, seed):
        self.expt = expt
        self.true_h = h
        self.duration = duration  # msec
        self.stim_noise = stim_noise
        self.trial_number = trial_number
        self.init_state = init_state
        self.cp_times = gen_cp_discrete(self.duration, self.true_h)
        self.end_state = self.compute_endstate(self.cp_times.size)
        self.tot_trial = self.expt.tot_trial
        self.seed = seed

    def compute_endstate(self, ncp):
        # the fact that the last state equals the initial state depends on
        # the evenness of the number of change points.
        if ncp % 2 == 0:
            return self.init_state
        else:
            return self.expt.switch(self.init_state)

    def randlh(self, envstate):
        # try clause might be redundant (because switch method does it)
        try:
            if envstate in self.expt.states:
                return np.random.normal(envstate, self.stim_noise)
            else:
                raise ValueError("Error in argument H: must be an element of "
                                 "Experiment.states")
        except ValueError as err:
            print(err.args)


class Stimulus(ExpTrial):
    def __init__(self, state_signal, cp_signal, expt, h, duration, stim_noise, trial_number,
                 init_state, seed):
        super().__init__(expt, h, duration, stim_noise, trial_number,
                         init_state, seed)
        self.state_signal = state_signal
        self.cp_signal = cp_signal
        self.binsize = self.expt.exp_dt  # in msec

        # number of bins, i.e. number of stimulus values to compute
        # the first bin has 0 width and corresponds to the stimulus presentation
        # at the start of the trial, when t = 0.
        # So for a trial of length T = N x exp_dt msecs, there will be an observation
        # at t = 0, t = exp_dt, t = 2 x exp_dt, ... , t = T
        self.nbins = int(self.exp_trial.duration / self.binsize) + 1

        self.stim = self.gen_stim()

    def gen_stim(self):

        # stimulus vector to be filled by upcoming while loop
        stimulus = np.zeros(self.nbins)

        ncp = self.exp_trial.cp_times.size  # number of change points

        # loop variables
        last_envt = self.exp_trial.init_state
        next_cp_idx = 0
        non_passed = True

        for bin_nb in np.arange(self.nbins):
            # exact time in msec, of current bin
            curr_time = bin_nb * self.binsize

            # Control flow setting current environment
            if ncp == 0:  # no change point
                curr_envt = last_envt
            else:
                next_cp = self.exp_trial.cp_times[next_cp_idx]  # next change point time in msec
                if curr_time < next_cp:  # current bin ends before next cp
                    curr_envt = last_envt
                else:  # current bin ends after next cp
                    if non_passed:
                        curr_envt = self.exp_trial.expt.switch(last_envt)
                        if next_cp_idx < ncp - 1:
                            next_cp_idx += 1
                        else:
                            non_passed = False  # last change point passed
                    else:
                        curr_envt = last_envt

                        #             print('time, envt', curr_time, curr_envt)
            # compute likelihood to generate stimulus value
            stimulus[bin_nb] = self.exp_trial.randlh(curr_envt)

            # update variables for next iteration
            last_envt = curr_envt

        return stimulus


class IdealObs(object):
    def __init__(self, expt, prior_states=np.array([.5, .5]), prior_h=np.array([1, 1])):
        self.expt = expt  # reference to Experiment object
        self.dt = dt
        self.prior_h = prior_h
        self.prior_states = prior_states  # TODO: check that prior_states is a stochastic vector

    # the following is the likelihood used by the ideal observer
    # H = assumed state of the environment
    # x = point at which to evaluate the pdf
    def lh(self, envstate, x, obs_noise):
        try:
            if envstate in self.expt.states:
                return scipy.stats.norm(envstate, obs_noise).pdf(x)
            else:
                raise ValueError("Error in argument H: must be an element of "
                                 "Experiment.states")
        except ValueError as err:
            print(err.args)


class ObsTrial(IdealObs):
    def __init__(self, stimulus,
                 prior_states=np.array([.5, .5]),
                 prior_h=np.array([1, 1])):
        super().__init__(stimulus.expt, prior_states, prior_h)
        self.stimulus = stimulus
        self.llr = np.zeros(self.stimulus.nbins)
        self.decision = 0
        self.obs_noise = self.exp_trial.stim_noise
        self.trial_number = self.exp_trial.trial_number
        # artificial observations for testing purposes
        #         self.obs = np.array([0.7, -0.2, -2, 3.6])
        self.obs = self.gen_obs()
        self.dbname = dbname
        self.marg_gamma = None
        self.marg_gamma_feedback = None

    def gen_obs(self):
        return self.stimulus.stim

    def infer(self, save2db):
        #  initialize variables
        envp = self.expt.states[1]
        envm = self.expt.states[0]
        joint_plus_new = np.zeros(self.stimulus.nbins)
        joint_plus_current = np.copy(joint_plus_new)
        joint_minus_new = np.copy(joint_plus_new)
        joint_minus_current = np.copy(joint_plus_new)
        priorprec = self.prior_h.sum()

        # get first observation
        x = self.obs[0]

        # First time step
        # compute joint posterior after first observation: P_{t=0}(H,a=0) --- recall first obs at t=0
        joint_minus_current[0] = self.lh(envm, x, self.obs_noise) * self.prior_states[0]
        joint_plus_current[0] = self.lh(envp, x, self.obs_noise) * self.prior_states[1]

        normcoef = joint_plus_current[0] + joint_minus_current[0]
        joint_plus_current[0] = joint_plus_current[0] / normcoef
        joint_minus_current[0] = joint_minus_current[0] / normcoef

        # pursue algorithm if interrogation time is greater than 0
        if self.stimulus.duration == 0:
            print('trial has duration 0 msec')
            exit(1)
            # todo: find a way to exit the function

        for j in np.arange(self.stimulus.nbins - 1):
            # make an observation
            x = self.obs[j + 1]

            # compute likelihoods
            xp = self.lh(envp, x, self.obs_noise)
            xm = self.lh(envm, x, self.obs_noise)

            # update the boundaries (with 0 and j change points)
            ea = 1 - alpha / (j + priorprec)
            eb = (j + alpha) / (j + priorprec)
            joint_plus_new[0] = xp * ea * joint_plus_current[0]
            joint_minus_new[0] = xm * ea * joint_minus_current[0]
            joint_plus_new[j + 1] = xp * eb * joint_minus_current[j]
            joint_minus_new[j + 1] = xm * eb * joint_plus_current[j]

            # update the interior values
            if j > 0:
                vk = np.arange(2, j + 2)
                ep = 1 - (vk - 1 + alpha) / (j + priorprec)  # no change
                em = (vk - 2 + alpha) / (j + priorprec)  # change

                joint_plus_new[vk - 1] = xp * (np.multiply(ep, joint_plus_current[vk - 1]) +
                                               np.multiply(em, joint_minus_current[vk - 2]))
                joint_minus_new[vk - 1] = xm * (np.multiply(ep, joint_minus_current[vk - 1]) +
                                                np.multiply(em, joint_plus_current[vk - 2]))

            # sum probabilities in order to normalize
            normcoef = joint_plus_new.sum() + joint_minus_new.sum()

            # if last iteration of the for loop, special computation for feedback observer
            if j == self.stimulus.nbins - 2:
                jpn = joint_plus_new.copy()
                jmn = joint_minus_new.copy()
                if self.stimulus.end_state == envm:  # feedback is H-
                    # update the boundaries (with 0 and j change points)
                    ea = 1 - alpha / (j + priorprec)
                    eb = (j + alpha) / (j + priorprec)
                    jpn[0] = 0
                    jmn[0] = xm * ea * joint_minus_current[0]
                    jpn[j + 1] = 0
                    jmn[j + 1] = xm * eb * joint_plus_current[j]

                    # update the interior values
                    if j > 0:
                        vk = np.arange(2, j + 2)
                        ep = 1 - (vk - 1 + alpha) / (j + priorprec)  # no change
                        em = (vk - 2 + alpha) / (j + priorprec)  # change

                        jpn[vk - 1] = 0
                        jmn[vk - 1] = xm * (np.multiply(ep, joint_minus_current[vk - 1]) +
                                            np.multiply(em, joint_plus_current[vk - 2]))

                    joint_plus_feedback = jpn.copy()
                    joint_minus_feedback = jmn / jmn.sum()

                else:
                    # update the boundaries (with 0 and j change points)
                    ea = 1 - alpha / (j + priorprec)
                    eb = (j + alpha) / (j + priorprec)
                    jpn[0] = xp * ea * joint_plus_current[0]
                    jmn[0] = 0
                    jpn[j + 1] = xp * eb * joint_minus_current[j]
                    jmn[j + 1] = 0

                    # update the interior values
                    if j > 0:
                        vk = np.arange(2, j + 2)
                        ep = 1 - (vk - 1 + alpha) / (j + priorprec)  # no change
                        em = (vk - 2 + alpha) / (j + priorprec)  # change

                        jpn[vk - 1] = xp * (np.multiply(ep, joint_plus_current[vk - 1]) +
                                            np.multiply(em, joint_minus_current[vk - 2]))
                        jmn[vk - 1] = 0

                    joint_minus_feedback = jmn.copy()
                    joint_plus_feedback = jpn / jpn.sum()

            joint_plus_current = joint_plus_new / normcoef
            joint_minus_current = joint_minus_new / normcoef

        # compute marginals over change point count if last iteration
        self.marg_gamma = joint_plus_current + joint_minus_current
        self.marg_gamma_feedback = joint_plus_feedback + joint_minus_feedback

        if save2db:
            self.save2db(seed=self.stimulus.seed)

    def save2db(self, seed):
        dict2save = dict()
        dict2save['commit'] = '14df86f2dc071450cf73012df5d4337290b3e51b'
        dict2save['path2file'] = 'sims_learning_rate/scripts/feedback_effect_1.py'
        dict2save['discreteTime'] = True
        dict2save['trialNumber'] = int(self.exp_trial.trial_number)
        dict2save['hazardRate'] = round(float(self.exp_trial.true_h), 3)
        printdebug(debugmode=not debug, vartuple=("duration",
                                                  self.exp_trial.duration))
        printdebug(debugmode=not debug, vartuple=("type",
                                                  type(self.exp_trial.duration)))
        dict2save['trialDuration'] = int(self.exp_trial.duration)
        dict2save['SNR'] = round(float(2 / self.obs_noise), 3)
        dict2save['seed'] = seed
        printdebug(debugmode=not debug,
                   vartuple=("seed has type", type(seed)))
        printdebug(debugmode=not debug, vartuple=("seed", seed))
        printdebug(debugmode=not debug, vartuple=("initial state",
                                                  self.exp_trial.init_state))
        printdebug(debugmode=not debug, vartuple=("type",
                                                  type(self.exp_trial.init_state)))
        dict2save['initialState'] = int(self.exp_trial.init_state)
        dict2save['endState'] = int(self.exp_trial.end_state)
        printdebug(debugmode=not debug, vartuple=("alpha",
                                                  self.prior_h[0]))
        printdebug(debugmode=not debug, vartuple=("type",
                                                  type(self.prior_h[0])))
        dict2save['alpha'] = float(self.prior_h[0])
        dict2save['beta'] = float(self.prior_h[1])

        # compute and store time since last change point
        if self.exp_trial.cp_times.size > 0:
            time_last_cp = self.exp_trial.duration - self.exp_trial.cp_times[-1]
        else:
            time_last_cp = self.exp_trial.duration
        printdebug(debugmode=not debug, vartuple=("time last CP",
                                                  time_last_cp))
        printdebug(debugmode=not debug, vartuple=("type",
                                                  type(time_last_cp)))
        dict2save['timeLastCp'] = int(time_last_cp)

        # compute and store mean and stdev of marginals over CP counts
        mean_gamma = np.dot(self.marg_gamma, np.arange(len(self.marg_gamma)))
        stdev_gamma = np.sqrt(np.dot(self.marg_gamma, np.arange(len(self.marg_gamma)) ** 2) - mean_gamma ** 2)
        mean_gamma_feedback = np.dot(self.marg_gamma_feedback, np.arange(len(self.marg_gamma_feedback)))
        stdev_gamma_feedback = np.sqrt(np.dot(self.marg_gamma_feedback,
                                              np.arange(len(self.marg_gamma_feedback)) ** 2) - mean_gamma_feedback ** 2)

        dict2save['meanFeedback'] = mean_gamma_feedback
        dict2save['meanNoFeedback'] = mean_gamma
        dict2save['meandiff'] = mean_gamma_feedback - mean_gamma
        dict2save['absmeandiff'] = abs(mean_gamma_feedback - mean_gamma)
        dict2save['stdevFeedback'] = stdev_gamma_feedback
        dict2save['stdevNoFeedback'] = stdev_gamma
        dict2save['stdevdiff'] = stdev_gamma_feedback - stdev_gamma
        dict2save['absstdevdiff'] = abs(stdev_gamma_feedback - stdev_gamma)

        # save dict to SQLite db
        db = dataset.connect('sqlite:///' + dbname + '.db')
        table = db['feedback']
        table.insert(dict2save)

        # # write heavy data to file
        # heavydict = dict()
        # heavydict['seed'] = seed
        # heavydict['marginal-gamma'] = self.marg_gamma
        # # heavydict['post-var-h'] =
        # # heavydict['post-mean-h'] =
        # filename = dbname + '.pickle'
        # with open(filename, 'wb') as handle:
        #     pickle.dump(heavydict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    # SET PARAMETERS
    printdebug(debugmode=debug,
               string="debug prints activated")
    # list of hazard rates to use in sim
    hazard_rates = [0.01]
    hstep = round(0.05, 3)
    hh = hstep
    while hh < 0.26:  # true value is 0.51
        hazard_rates += [hh]
        hh += hstep
    printdebug(debugmode=not debug, vartuple=("hazard_rates", hazard_rates))
    # hyper-parameters of Beta prior
    alpha = 1
    beta = 1

    # list of SNRs to use in sim
    SNR = []
    stimstdev = []
    snrstep = round(0.2, 3)
    snr = snrstep
    while snr < 1.1:  # true value is 3.01
        stdev = 2.0 / snr
        SNR += [snr]
        stimstdev += [stdev]
        snr += snrstep

    # time step. Should be one for discrete time
    dt = 1  # for discrete time

    # numpy array of trial durations to use in sim
    trial_durations = [50]
    tdstep = 100
    td = tdstep
    while td < 501:  # true value is 2001
        trial_durations += [td]
        td += tdstep
    trial_durations = np.array(trial_durations, dtype=int)

    # total number of trials per condition
    nTrials = 100

    # boolean variables telling script what to plot
    singleTrialOutputs = [True, True, True]
    multiTrialOutputs = [True, True]

    # filenames for saving data
    dbname = 'true_6'

    printdebug(debugmode=not debug, string="about to create expt object")
    Expt = Experiment(setof_stim_noise=stimstdev, exp_dt=dt, setof_trial_dur=trial_durations,
                      setof_h=hazard_rates, tot_trial=nTrials)
    printdebug(debugmode=not debug, string="Expt object created")
    Observer = IdealObs(expt=Expt, prior_h=np.array([alpha, beta]))
    printdebug(debugmode=not debug, string="Observer object created")
    aa = datetime.datetime.now().replace(microsecond=0)
    printdebug(debugmode=not debug, string="initial time stored")
    printdebug(debugmode=not debug, string="about to execute the launch method")
    Expt.launch(Observer)

    bb = datetime.datetime.now().replace(microsecond=0)
    print('total elapsed time in hours:min:sec is', bb - aa)
