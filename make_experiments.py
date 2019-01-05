
# coding: utf-8

# In[1]:


from bayesian_benchmarks.data import regression_datasets
from bayesian_benchmarks.database_utils import Database

import itertools
import os


# In[29]:


def make_experiment_combinations(combinations: list):
    """
    The product of all combinations of arguments.
    :param combinations: A list of dictionaries, each with a list of args
    :return: A list of dictionaries for all combinations
    """
    fields = []
    vals = []
    for p in combinations:
        for k in p:
            fields.append(k)
            vals.append(p[k])

    ret = []
    for a in itertools.product(*vals):
        d = {}

        for f, arg in zip(fields, a):
            d[f] = arg

        ret.append(d)

    return ret


def make_local_jobs(script: str, experiments: list, overwrite=False):
    """
    Writes a file of commands to be run in in series on a single machine, e.g. 
    #!/usr/bin/env bash
    python run_regression --split=0
    python run_regression --split=1
    etc. 
    
    If overwrite=True then a new file is written with a shebang, otherwise lines are appended. 

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    if overwrite:
        with open('local_run', 'w') as f:
            f.write('#!/usr/bin/env bash\n\n')

    with open('local_run', 'a') as f:
        for e in experiments:
            s = 'python {}.py '.format(script)
            for k in e:
                s += '--{}={} '.format(k, e[k])
            s += '\n'
            f.write(s)
            
def make_condor_jobs(script: str, experiments: list, overwrite=False):
    """
    Writes a condor submission file, and also creates the executable if necessary. Preamble for the 
    exectable (e.g. for setting up the python environment) should go in 'preamble.txt.txt'. Preamble
    for the condor submission should go in condor_preamble.txt.txt.txt.

    If overwrite=True then a new file is written with the condor preamble from condor_preamble.txt.txt,
    otherwise lines are appended. 

    :param script: name of python script to run
    :param experiments: list of dictionaries of args
    :return: None
    """
    
    cmds = ['cd /home/kaw293/',
            '. /home/kaw293/miniconda3/etc/profile.d/conda.sh',
            'conda activate gpytorch']
    for num_gpus in [1, 2, 4, 6, 8]: 
        job_dir = '/home/kaw293/jobs_{}'.format(num_gpus)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
            
        for exp in experiments:
            python_cmd = ' '.join(['python {}.py'.format(script)] 
                                  + ['--{}={}'.format(k, exp[k]) for k in exp]
                                  + ['--num_gpus={}'.format(num_gpus)])
            name ='{}_{}'.format(exp['dataset'], exp['split'])
            sbatch_settings = [
                '#!/bin/bash',
                '#SBATCH -J {}_{}'.format(name, num_gpus),
                '#SBATCH -o {}_{}.out'.format(name, num_gpus),
                '#SBATCH -e {}_{}.err'.format(name, num_gpus),
                '#SBATCH -N 1',
                '#SBATCH -n 1',
                '#SBATCH --mem=16000',
                '#SBATCH --partition=default_gpu',
                '#SBATCH --gres=gpu:{}'.format(num_gpus)]
            
            sub_file= os.path.join(job_dir, "{}_{}.sub".format(name, num_gpus))
            if not os.path.isfile(sub_file):
                with open(sub_file, 'w') as f:
                    for sset in sbatch_settings:
                        f.write(sset + "\n")
                    f.write(' && '.join(cmds + [python_cmd]))
                    
def remove_already_run_experiments(table, experiments):
    res = []

    with Database() as db:
        for e in experiments:
            if len(db.read(table, ['test_loglik'], e)) == 0:
                res.append(e)

    s = 'originally {} experiments, but {} have already been run, so running {} experiments'
    print(s.format(len(experiments), len(experiments) - len(res), len(res)))
    return res

#################################################
models = ['exact_gp']

############# Regression
combinations = []
datasets = ['wilson_'+name for name in 
            ['elevators', 'kin40k', 'protein', 'keggdirected',
             'slice', 'keggundirected', '3droad', 
             'song', 'buzz', 'tamielectric'
            ]
           ]

combinations.append({'dataset' : datasets})
combinations.append({'split' : range(5)})
combinations.append({'model' : models})
experiments = make_experiment_combinations(combinations)
#experiments = remove_already_run_experiments('regression', experiments)

make_condor_jobs('/home/kaw293/bayesian_benchmarks/bayesian_benchmarks/tasks/regression', experiments, overwrite=True)
