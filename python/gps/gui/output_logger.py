"""
GPS output logger
used to log data when GUI is off
"""
# Needed for typechecks
import numpy as np
from gps.algorithm.algorithm_badmm import AlgorithmBADMM
from gps.algorithm.algorithm_mdgps import AlgorithmMDGPS

class GPSOutputLogger(object):
    def __init__(self, hyperparams):
        self._hyperparams = hyperparams
        self._log_filename = self._hyperparams['log_filename']
        self.log_text('\n')
        self.log_text(self._hyperparams['info'])
        self._first_update = True

    def log_text(self, text):
        with open(self._log_filename, 'a') as f:
            f.write(text + '\n')

    # Iteration update functions
    def update(self, itr, algorithm, pol_sample_lists):
        """
        After each iteration, update the iteration data output, the cost plot
        """
        if self._first_update:
            self._output_column_titles(algorithm)
            self._first_update = False

        costs = [np.mean(np.sum(algorithm.prev[m].cs, axis=1)) for m in range(algorithm.M)]
        self._update_iteration_data(itr, algorithm, costs, pol_sample_lists)

    def _output_column_titles(self, algorithm):
        """
        Setup iteration data column titles: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        self.log_text(self._hyperparams['experiment_name'])
        if isinstance(algorithm, AlgorithmMDGPS) or isinstance(algorithm, AlgorithmBADMM):
            condition_titles = '%3s | %8s %12s' % ('', '', '')
            itr_data_fields  = '%3s | %8s %12s' % ('itr', 'avg_cost', 'avg_pol_cost')
        else:
            condition_titles = '%3s | %8s' % ('', '')
            itr_data_fields  = '%3s | %8s' % ('itr', 'avg_cost')
        for m in range(algorithm.M):
            condition_titles += ' | %8s %9s %-7d' % ('', 'condition', m)
            itr_data_fields  += ' | %8s %8s %8s' % ('  cost  ', '  step  ', 'entropy ')
            if isinstance(algorithm, AlgorithmBADMM):
                condition_titles += ' %8s %8s %8s' % ('', '', '')
                itr_data_fields  += ' %8s %8s %8s' % ('pol_cost', 'kl_div_i', 'kl_div_f')
            elif isinstance(algorithm, AlgorithmMDGPS):
                condition_titles += ' %8s' % ('')
                itr_data_fields  += ' %8s' % ('pol_cost')
        self.log_text(condition_titles)
        self.log_text(itr_data_fields)

    def _update_iteration_data(self, itr, algorithm, costs, pol_sample_lists):
        """
        Update iteration data information: iteration, average cost, and for
        each condition the mean cost over samples, step size, linear Guassian
        controller entropies, and initial/final KL divergences for BADMM.
        """
        avg_cost = np.mean(costs)
        if pol_sample_lists is not None:
            test_idx = algorithm._hyperparams['test_conditions']
            # pol_sample_lists is a list of singletons
            samples = [sl[0] for sl in pol_sample_lists]
            pol_costs = [np.sum(algorithm.cost[idx].eval(s)[0])
                    for s, idx in zip(samples, test_idx)]
            itr_data = '%3d | %8.2f %12.2f' % (itr, avg_cost, np.mean(pol_costs))
        else:
            itr_data = '%3d | %8.2f' % (itr, avg_cost)
        for m in range(algorithm.M):
            cost = costs[m]
            step = np.mean(algorithm.prev[m].step_mult * algorithm.base_kl_step)
            entropy = 2*np.sum(np.log(np.diagonal(algorithm.prev[m].traj_distr.chol_pol_covar,
                    axis1=1, axis2=2)))
            itr_data += ' | %8.2f %8.2f %8.2f' % (cost, step, entropy)
            if isinstance(algorithm, AlgorithmBADMM):
                kl_div_i = algorithm.cur[m].pol_info.init_kl.mean()
                kl_div_f = algorithm.cur[m].pol_info.prev_kl.mean()
                itr_data += ' %8.2f %8.2f %8.2f' % (pol_costs[m], kl_div_i, kl_div_f)
            elif isinstance(algorithm, AlgorithmMDGPS):
                # TODO: Change for test/train better.
                if test_idx == algorithm._hyperparams['train_conditions']:
                    itr_data += ' %8.2f' % (pol_costs[m])
                else:
                    itr_data += ' %8s' % ("N/A")
        self.log_text(itr_data)

