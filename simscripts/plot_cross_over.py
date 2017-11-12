"""
This script allows to plot two curves, using data stored in an SQLite
database. Each curve is percentage correct as function of time since last change point.
"""
import dataset
import matplotlib.pyplot as plt


def plot_cross_over(pair1, pair2, data, columnnames, fignum):
    """
    Generate two curves (one per parameter pair)
    Plot curves in order to visualize the cross-over effect
    :param pair1: 2-element tuple (snr, h)
        :snr: float
        :h: float in Hz
    :param pair2: like pair1, for the second curve
    :param data: database object from dataset module
    :param columnnames: dict containing column names from db

            columnnames = {'commit': 'comm',
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
    :param fignum: figure number (integer)
    :return: produces cross-over plot for the two given curves
    """
    x = list(range(8))
    result1 = data.query('SELECT AVG({}) AS perf FROM crossover WHERE \
                         {} = {} AND {} = {} \
                         GROUP BY {} \
                         ORDER BY {} DESC;'.format(columnnames['correctness'],
                                                   columnnames['snr'],
                                                   pair1[0],
                                                   columnnames['h'],
                                                   pair1[1],
                                                   columnnames['bin_number'],
                                                   columnnames['bin_number']))
    curve1 = []
    for row_data in result1:
        curve1 += [row_data['perf']]
    result2 = data.query('SELECT AVG({}) AS perf FROM crossover WHERE \
                             {} = {} AND {} = {} \
                             GROUP BY {} \
                             ORDER BY {} DESC;'.format(columnnames['correctness'],
                                                       columnnames['snr'],
                                                       pair2[0],
                                                       columnnames['h'],
                                                       pair2[1],
                                                       columnnames['bin_number'],
                                                       columnnames['bin_number']))
    curve2 = []
    for row_data in result2:
        curve2 += [row_data['perf']]

    plt.figure(fignum)
    plt.plot(x, curve1, x, curve2, linewidth=3.0)
    plt.title('cross-over 500 trials per bin')
    plt.legend(['snr ' + str(pair1[0]) + '; h ' + str(pair1[1]),
                'snr ' + str(pair2[0]) + '; h ' + str(pair2[1])])
    plt.xlabel('bin number, counting backward from end of trial')
    plt.ylabel('percentage correct')
    plt.show()


if __name__ == "__main__":
    # name of SQLite db
    dbname = 'test_true_2'
    # create connection to SQLite db
    db = dataset.connect('sqlite:///' + dbname + '.db')
    # get handle for specific table of the db
    table = db['crossover']

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

    plot_cross_over((1, 0.1), (1, 2), db, column_names, 1)
    plot_cross_over((2, 0.1), (2, 2), db, column_names, 2)
