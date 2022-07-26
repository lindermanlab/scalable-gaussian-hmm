# Create and execute job submission script according to parameters

import os
import argparse
from datetime import datetime

def mkdir(dir):
    """Make a directory if it does not yet exist."""
    if not os.path.exists(dir):
        os.mkdir(dir)

DATADIR = os.environ['DATADIR']
TEMPDIR = os.environ['TEMPDIR']
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

py_script_name = 'fit_fish0_137'

SBATCH_ARGS = {
    'partition': dict(
        type=str, default='dev',
        help='Requested partition. Use `sh_part` to get system limits.'),
    'time': dict(
        type=str, default='dev',
        help='Requested partition. Use `sh_part` to get system limits.'),
    'mem_per_cpu':dict(
        type=str, default='6G',
        help='Max amount of real memory per CPU required by the job.'),
}

# To specify an arrayble arg, add option `nargs='*'`
SCRIPT_ARGS = {
    'method': dict(
        type=str, choices=['pmap','vmap'], #required=True,
        help='Parallelization approach. NB: If using pmap, must run on node with multiple cores.'),
    'states':dict(
        default=['5'], nargs='*',
        help='Number of HMM states to fit. Arrayble.'),
    'batch_size': dict(
        default=['8'], nargs='*',
        help='# batches loaded at once into memory. Arrayable.'),
    'frames_per_batch': dict(
        default=['100000'], nargs='*',
        help='# frames per batch. Arrayable.'),
    'train': dict(
        type=float, default=0.8,
        help='If >1, number of files in dataset to train over. If [0, 1), fraction of dataset to train over.'),
    'test': dict(
        type=float, default=0.2,
        help='If >1, number of files in dataset to test over. If [0, 1), fraction of dataset to test over.'),
    'iters': dict(
        default=['30'], nargs='*',
        help='Number of EM iterations to run. Arrayable.'),
    'seed': dict(
        type=int, default=2207011837,
        help='PRNG seed to split data and intialize HMM.'),
    'max_frames_per_day': dict(
        type=int, default=0,
        help='For debugging. Truncate # frames available in each file/day to this value.'),
        # Must make default value 0 (instead of -1). Otherwise, if argument '-1'
        # used, the argparse for fit_fish0_137 script interprets the negative
        # number as an argument and throws "expected one argument" error
}

parser = argparse.ArgumentParser(
    description='Create and execute job submission script.')

for arg_dict in [SBATCH_ARGS, SCRIPT_ARGS]:
    for flag, kwargs in arg_dict.items():
        parser.add_argument('--{}'.format(flag), **kwargs)
    
# Additional arguments
parser.add_argument(
    '--mprof', action='store_true',
    help='If specified, profile memory usage with memory_profiler.')
parser.add_argument(
    '--norun', action='store_true',
    help='If specified, only generate job scripts but do not run')

ARRAYABLE_SCRIPT_KEYS = [flag for flag, flag_opts in SCRIPT_ARGS.items()
                         if flag_opts.get('nargs', None) == '*']

def parse_and_split_args(args):
    """Parses input arguments into seperate Slurm and pyscript dictionaries.
    In particular, this function handles any python args specified as arrays.

    Parameters
        args: dict or argparse.Namespace

    Returns
        sb_args: dict of sbatch args
        py_args: dict of python script args
        iter: iterable or None. If not None, this will be for looped over
    """

    # Add all slurm sbatch argument inputs
    sb_args = {k: args[k] for k in SBATCH_ARGS.keys()}
    
    # Add all Python script argument inputs, EXCEPT those that might be arrayed
    py_args = {k: args[k] for k in set(SCRIPT_ARGS.keys())-set(ARRAYABLE_SCRIPT_KEYS)}

    is_batch_size_and_pmap = lambda k: (k=='batch_size' and args['method']=='pmap')
    # -----------------------------------------------------
    # Identify which script arguments are actually arrayed
    # -----------------------------------------------------
    arrayed_keys = []
    for k in ARRAYABLE_SCRIPT_KEYS:
        args[k] = list(map(int, args[k]))    # Convert arrayable values into ints

        if len(args[k]) == 1:                # If single value specified instead of array
            py_args[k] = args[k][0]          #      Add arg and value directly to py_args
            
            if is_batch_size_and_pmap(k):
                sb_args['mincpus'] = py_args[k]
        else:
            arrayed_keys.append(k)           # Add key to list of arrayed_keys

    # -------------------------------------------------
    # If there are arrayed elements, handle them here!
    # -------------------------------------------------
    def make_arr(key):
        # Remove all whitespaces from sbatch directive (i.e. [10,20] instead of [10, 20])
        sb_args['array'] = str(args[key]).replace(" ", "")
        py_args[key] = '$SLURM_ARRAY_TASK_ID'
        return

    def make_itr(key):
        return zip([key]*len(args[key]), args[key])

    # 0 arguments specified as an array
    if len(arrayed_keys) == 0:
        iterable = None

    # 1 argument specified as an array
    elif len(arrayed_keys) == 1:
        k = arrayed_keys[0]

        # If arraying over batch_size and pmapping, MUST use for loop
        # since batch_size controls SBATCH arg `mincpus``
        # if k == 'batch_size' and args['method'] == 'pmap':
        if is_batch_size_and_pmap(k):
            iterable = make_itr('batch_size')
        else:
            make_arr(k)
            iterable = None

    # >2 arguments specified as arrays
    elif len(arrayed_keys) >= 2:
        # Sort arrayed_keys from shortest to longest arg values
        len_arrays = [len(args[k]) for k in arrayed_keys]
        _, arrayed_keys = zip(*sorted(zip(len_arrays, arrayed_keys)))
        arrayed_keys = list(arrayed_keys)

        # Let Slurm array over the argument with the most args, UNLESS
        # that arg happens to be 'batch_size', then choose next most arg
        k = arrayed_keys.pop(-1) if not is_batch_size_and_pmap(arrayed_keys[-1]) \
            else arrayed_keys.pop(-2)
        make_arr(k)

        if len(arrayed_keys) == 1:
            iterable = make_itr(arrayed_keys[0])
        else:
            # NOT YET TESTED IN FOR LOOP
            from itertools import product
            return product(map(make_itr, arrayed_keys))

    return sb_args, py_args, iterable

def generate_script(prefix, sb_args, py_args, logdir=TEMPDIR, mprof=False):
    """Generate job script for fitting HMM.

    Params
        fpath: str. Path to job submission file
        args: dict. 
        sbatch_args: dict of SBATCH directive arguments and values.
        array: tuple, (arrayed_arg, array_values)
    """
    fpath = os.path.join(SCRIPTDIR, '.jobs', "{}.job".format(prefix))
    with open(fpath, 'w') as f:
        f.writelines("#!/bin/bash\n")

        for k, v in sb_args.items():
            if k == 'mem_per_cpu':
                f.writelines("#SBATCH --mem-per-cpu={}\n".format(v))
            else:
                f.writelines("#SBATCH --{}={}\n".format(k, v))
        f.writelines("#SBATCH --output={}\n".format(os.path.join(logdir, prefix+'.%j.out')))
        f.writelines("#SBATCH --error={}\n".format(os.path.join(logdir, prefix+'.%j.err')))
        f.writelines('\n')

        # Set up environment
        f.writelines('source ~/.bashrc\n')
        f.writelines('module purge > /dev/null\n')
        f.writelines('ml python/3.9\n')
        f.writelines('venv activate kf-cpu\n')

        f.writelines('cd {}\n'.format(SCRIPTDIR))
        f.writelines('\n')

        # Echo all parameters into output file
        f.writelines('echo JOB ID $SLURM_JOB_ID, TASK_ID $SLURM_ARRAY_TASK_ID\n')
        for k, v in sb_args.items():
            f.writelines('echo {} {}\n'.format(k, v))
        f.writelines('\n')

        # Execute code
        if mprof:
            f.writelines('mpfile={}\n'.format(os.path.join(logdir, prefix+'.$SLURM_JOB_ID')))
            f.writelines('mprof run -M -o $mpfile --backend psutil_pss \\\n')
            f.writelines('\t{} \\\n'.format(py_script_name))
        else:
            f.writelines('python {}.py \\\n'.format(py_script_name))

        for k, v in py_args.items():
            f.writelines('\t--{} {} \\\n'.format(k,v))
        f.writelines('\t--log_dir {}\\\n'.format(logdir))
        f.writelines('\t--log_prefix {}\\\n'.format(prefix))
        
        if mprof:
            f.writelines('\n')
            f.writelines('mprof peak $mpfile.mprof                      # Print peak memory usage\n')
            f.writelines('mprof plot $mpfile.mprof -o $mpfile.png -s    # Make figure (-o), with slope (-s)\n')
    return fpath

def main():
    # `vars()` converts argparse object into dict
    args = vars(parser.parse_args())    
    sb_args, py_args, iterable = parse_and_split_args(args)
    
    timestamp = datetime.now().strftime("%y%m%d%H%M")
    if iterable:
        for i, (k, v) in enumerate(iterable):
            # NOT YET TESTED for itertools.product!!
            py_args[k] = v
            if k=='batch_size' and py_args['method']=='pmap':
                sb_args['mincpus'] = v
            prefix = "{}_{}-{}".format(timestamp, i, py_args['method'])
            job_path = generate_script(prefix, sb_args, py_args, mprof=args['mprof'])

            if args['norun']:
                print(f"--norun specified. jobscript generated:\n\t{job_path}")
            else:
                os.system("sbatch {}".format(jobpath))
    else:
        prefix = "{}-{}".format(timestamp, py_args['method'])
        jobpath = generate_script(prefix, sb_args, py_args, mprof=args['mprof'])
        
        if args['norun']:
            print(f"--norun specified. jobscript generated:\n\t{job_path}")
        else:
            os.system("sbatch {}".format(jobpath))

if __name__ == '__main__':
    main()