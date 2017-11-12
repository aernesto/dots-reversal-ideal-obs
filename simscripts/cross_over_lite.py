"""
====
GENERAL AIM
====
This script aims at reproducing the cross-over effect, for the continuous-time
model, and for an ideal-observer who uses a delta prior over the hazard rate

The idea is to compute percentage correct as a function of the time since the
last change point, for several hazard rate values.

We assume for now that the observer knows the true hazard rate.

====
ALGORITHM
====
1. For i in DB rows DO:
2.  read parameters
3.  run inference while creating environment on the fly
4.  write to DB
"""
import numpy as np
import dataset
import datetime

# the following dict just makes it more convenient for the developer to
# access the appropriate column name in the database
column_names = {'commit': 'comm',
                'trial_duration': 'dur',
                'snr': 'snr',
                'h': 'h',
                'seed': 'seed',
                'bin_number': 'numb',
                'bin_width': 'bwidth',
                'init_state': 'init',
                'end_state': 'end',
                'decision': 'dec',
                'correctness': 'correct'}


def read_param(db_table, row_num):
    """
    extracts parameters from a given row in database table
    :param db_table: table object from dataset module
    :param row_num: id value from the column field for the row to read
    :return: tuple of parameters (trial_duration, snr, h, seed)
    """
    row = db_table.find_one(id=row_num)
    final_tuple = (row[column_names['trial_duration']],
                   row[column_names['snr']],
                   row[column_names['h']],
                   row[column_names['seed']])
    return final_tuple


def generate_cp(t, s, hazard):
    """
    generate the CP times for a trial by successively sampling
    from an exponential distribution
    :param t: trial duration in seconds
    :param s: seed value for numpy.random.seed()
    :param hazard: hazard rate in Hz
    :return: list (possibly empty) of change point times (strictly smaller than duration)
    """
    np.random.seed(s)
    cp_list = []
    cp_time = 0

    while cp_time < t:
        dwell_time = np.random.exponential(1. / hazard)
        cp_time += dwell_time
        cp_list += [cp_time]

    if len(cp_list) == 0:
        return np.array(cp_list)
    else:
        return np.array(cp_list[:-1])


def run_sde(cp_times, hazard, mm, trial_duration):
    """
    SDE for optimal evidence accumulation with known hazard rate.
    :param cp_times: numpy array of change point times
    :param hazard: hazard rate in Hz
    :param mm: SNR for normalized equation
    :param trial_duration: trial duration in seconds
    :return: tuple (init_state, end_state, decision, correctness), where
        :init_state: initial state of environment for current trial
        :end_state: end state of environment for current trial
        :decision: choice of ideal observer at end of trial
        :correctness: correctness of ideal observer's choice (Boolean)
    """
    rho = np.sqrt(2*mm)
    dt = 0.01
    sqrt_dt_rho = np.sqrt(dt)*rho
    num_steps = int(trial_duration / dt)  # number of time steps
    y = np.zeros(num_steps)  # evidence
    np.random.seed(None)  # randomize seed of rng
    init = np.random.choice([-1, 1])  # initial state of the environment, with flat prior
    state = init
    current_time = 0
    cp_index = 0
    last_cp = cp_times[cp_index]

    keep_if = True
    for i in range(num_steps - 1):
        current_time += dt

        if (current_time >= last_cp) and keep_if:  # change hidden state if change point was crossed
            state *= -1
            if cp_index < len(cp_times) - 1:
                cp_index += 1
                last_cp = cp_times[cp_index]
            elif cp_index == len(cp_times) - 1:
                keep_if = False

        y[i+1] = y[i] + dt * (np.sign(state) * mm - 2 * hazard * np.sinh(y[i])) + sqrt_dt_rho * np.random.normal()

    initial_state = int(init)
    endstate = int(state)
    final_decision = int(np.sign(y[-1]))
    bool_dec = int(np.sign(y[-1]) == np.sign(state))

    return initial_state, endstate, final_decision, bool_dec


if __name__ == "__main__":
    # name of SQLite db
    dbname = 'test_true_2'
    # create connection to SQLite db
    db = dataset.connect('sqlite:///' + dbname + '.db')
    # get handle for specific table of the db
    table = db['crossover']
    aa = datetime.datetime.now().replace(microsecond=0)

    for row_id in range(80000):
        duration, snr, h, seed = read_param(table, row_id + 1)
        m = 2 * snr
        # last = generate_cp(duration, seed, h)
        # print('row id', row_id + 1)
        # print('last cp', last[-1])
        # curr_row = table.find_one(id=(row_id + 1))
        # print('bin value', curr_row[column_names['bin_number']])
        init_state, end_state, decision, correctness = run_sde(generate_cp(duration, seed, h), h, m, duration)

        # Save info to database
        table.update({column_names['init_state']: init_state,
                      column_names['end_state']: end_state,
                      column_names['decision']: decision,
                      column_names['correctness']: correctness,
                      'id': row_id + 1}, ['id'])
    bb = datetime.datetime.now().replace(microsecond=0)
    print('total elapsed time in hours:min:sec is', bb - aa)
