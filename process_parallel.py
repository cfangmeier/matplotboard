#!env/bin/python
from os import listdir
from os.path import join, isdir
import argparse
import subprocess

import multiprocessing
from multiprocessing.pool import Pool

from merge import merge_files


def run_job(job_number, executable, files):
    file_list = f'file_list_{job_number:02d}.txt'
    with open(file_list, 'w') as f:
        f.write("\n".join(files))

    output_filename = f'output_{job_number:02d}.root'
    ret = subprocess.run([executable, '-s', '-F', file_list,
                          '-o', output_filename])
    retcode = ret.returncode
    if retcode != 0:
        raise RuntimeError(f'Job {job_number} encountered errors!'
                           f'(retcode: {retcode}), check log file.')
    return (job_number, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument

    add('executable')
    add('--jobs', '-j', type=int, default=multiprocessing.cpu_count())
    add('--dir', '-d', default='./data')
    add('--mergeinto', '-m', default='output.root')
    args = parser.parse_args()

    if not isdir(args.dir):
        raise ValueError(f'Directory {args.dir} does not exist')

    files = sorted([join(args.dir, fname) for fname in listdir(args.dir) if fname[-5:] == '.root'])

    files_per_job = len(files) // args.jobs
    job_files = [files[i::args.jobs] for i in range(args.jobs)]
    output_files = []

    def job_callback(args):
        job_num, job_file = args
        print(f'job {job_num} finished')
        output_files.append(job_file)

    with Pool(args.jobs) as pool:
        print(f'Starting {args.jobs} processes to process {len(files)} files')
        results = []
        for i, files in enumerate(job_files):
            results.append(pool.apply_async(run_job, (i, args.executable, files),
                           callback=job_callback))
        for result in results: result.get()
        pool.close()
        pool.join()
        print('Finished processing nTuples.')
    print('Begin merging job files')
    merge_files(output_files, args.mergeinto)
