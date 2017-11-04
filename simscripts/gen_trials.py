"""
=== AIM ===

Aim of this script is to create hidden variable trials for each condition
The hidden variable is the state of the environment. A trial, here, is a
realization of this continuous-time stochastic process.

=== SQLite ===

Each trial is stored in an SQLite database file. The columns of the latter
are:
Column0: comm -- Commit number for the current script
Column1: dur -- Trial duration in seconds
Column2: snr -- SNR as positive float
Column3: h -- hazard rate in Herz
Column4: seed -- integer seed used to generate the trial
Column5: numb -- bin number of last change point (integer)
Column6: bwidth -- bin width, in msec

=== Bin numbering ===

A note about the bin numbering:
    For a trial duration of 6 seconds and a binwidth of 250 msec,
    bin 0 is (5750,6000) msec
    bin 1 is (5500, 5750] msec
    ...
    etc.
"""
import numpy as np
import dataset
import datetime


def time2bin(time, binwidth, duration, minim):
    """
    Converts a time point (float) in sec to a bin number (integer or None).
    Note, a bin window is right inclusive, left exclusive.

    the function will behave as follows:
    time2bin(5.990, 250, 6.000, 4.000) = 0
    time2bin(6.000, 250, 6.000, 4.000) = None
    time2bin(4.000, 250, 6.000, 4.000) = None
    time2bin(4.001, 250, 6.000, 4.000) = 7
    time2bin(5.750, 250, 6.000, 4.000) = 1
    time2bin(5.300, 250, 6.000, 4.000) = 2

    :param time: time of event, float
    :param binwidth: integer, in msec
    :param duration: float, trial duration in sec
    :param minim: minimum time below which the function returns None
    :return: integer representing the bin in which the time falls, or None if time is out of range
    """
    # round all floats to 3 decimal points, and cast binwidth to integer
    time = round(time, 3)
    duration = round(duration, 3)
    minim = round(minim, 3)
    binwidth = int(binwidth)

    if time <= minim or time >= duration:
        return None

    t = int((duration - time) * 1000)
    return t // binwidth


'''
Say we want to go up to 2 sec in the past, with bins of width 250msec and a trial duration of 6 seconds.
Then, 
bin 0 = (5.750, 6.000)
bin 1 = (5.500, 5.750]
bin 2 = (5.250, 5.500]
...
bin 7 = (4.000, 4.250]

Therefore, the min argument to the function above is 4.000
'''

# the following dict just makes it more convenient for the developer to
# access the appropriate column name in the database
column_names = {'commit': 'comm',
                'trial_duration': 'dur',
                'snr': 'snr',
                'h': 'h',
                'seed': 'seed',
                'bin_number': 'numb',
                'bin_width': 'bwidth'}

ntrials = 500                     # true value is 500 or 1000
# maximum number of iterations allowed in the upcoming while loop
# factor 8 is the number of bins
maxtrials = 1000 * 8 * ntrials
bin_width = 250                 # in msec
trial_duration = [6.000]        # in sec
min_time = [4.000]
hazard = [0.1, .5, 1, 1.5, 2]
snrs = [0.5, 1, 1.5, 2]
commit_number = 'b614a5eed6a8b84d514b3f808034a0534b3e2b5c'


if __name__ == "__main__":
    # name of SQLite db
    dbname = 'true_1'
    # create connection to SQLite db
    db = dataset.connect('sqlite:///' + dbname + '.db')
    # get handle for specific table of the db
    table = db['crossover']

    # the following dict keeps track of how many trials have been recorded, for each bin
    # the recording stops when the trial count hits ntrials, or when the while loop has exceeded maxtrials
    bin_counts = {'0': 0,
                  '1': 0,
                  '2': 0,
                  '3': 0,
                  '4': 0,
                  '5': 0,
                  '6': 0,
                  '7': 0}

    aa = datetime.datetime.now().replace(microsecond=0)

    # nested loops over parameters
    for T in trial_duration:
        for snr in snrs:
            for h in hazard:
                n = 0  # counts iterations for upcoming while loop
                while n < maxtrials:
                    n += 1
                    # generate random integer seed to simulate independent trials
                    np.random.seed(None)
                    seed = np.random.randint(1000000000)
                    np.random.seed(seed)

                    # generate the CP times from the trial by successively sampling
                    # from an exponential distribution
                    cp_time = 0

                    # maybe the following line is not needed, but PyCharm syntax checker was complaining
                    last_cp_time = cp_time

                    while cp_time < T:
                        dwell_time = np.random.exponential(1. / h)
                        last_cp_time = cp_time
                        cp_time += dwell_time

                    bin_nb = time2bin(last_cp_time, bin_width, trial_duration[0], min_time[0])

                    # go to next iteration of while loop without saving the data, if bin_nb is None
                    # or if this bin is already full. But if bin is full, check all other bins and
                    # if all bins are full, exit while loop.
                    # Otherwise, increment the count and pursue the loop
                    if bin_nb is None or bin_counts[str(bin_nb)] >= ntrials:
                        if sum(bin_counts.values()) == ntrials * 8:
                            break
                        else:
                            continue
                    else:
                        bin_counts[str(bin_nb)] += 1

                    # Save info to database
                    table.insert({column_names['commit']: commit_number,
                                  column_names['trial_duration']: trial_duration[0],
                                  column_names['snr']: snr,
                                  column_names['h']: h,
                                  column_names['seed']: seed,
                                  column_names['bin_number']: bin_nb,
                                  column_names['bin_width']: bin_width})

                print('outer while loop exited')
                print('final trial counts in each bin are')
                print(bin_counts)

    bb = datetime.datetime.now().replace(microsecond=0)
    print('total elapsed time in hours:min:sec is', bb - aa)
