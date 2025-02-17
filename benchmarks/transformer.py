import argparse
import logging
import os
import random
import subprocess
import tempfile
import shutil

from hfta.workflow import (attach_args as attach_workflow_args, workflow,
                           rearrange_runner_kwargs, extract_logging_level)


def main(args):

  def trial(
      B=None,
      device=None,
      prec=None,
      epochs=None,
      iters_per_epoch=None,
      outdir=None,
      env_map=None,
  ):
    cmd = [
        'python',
        'main.py',
        '--dataset',
        args.dataroot,
        '--epochs',
        str(epochs),
        '--iters-per-epoch',
        str(iters_per_epoch),
        '--device',
        device,
    ]
    if outdir is not None:
      cmd.extend(['--outf', outdir])
    else:
      tmp_path = tempfile.mkdtemp()
      cmd.extend(['--outf', tmp_path])

    if prec == 'amp' and device == 'cuda':
      cmd.append('--amp')

    num_hps = max(B, 1)
    hyperparam_strs = {
        'lr': [str(random.uniform(0.1, 10)) for _ in range(num_hps)],
        'gamma': [str(random.uniform(0.3, 0.99)) for _ in range(num_hps)],
        'step_size': [str(int(random.uniform(3, 20))) for _ in range(num_hps)],
    }

    for flag, vals in hyperparam_strs.items():
      if B > 0:
        cmd.extend(['--{}'.format(flag)] + vals)
      else:
        cmd.extend(['--{}'.format(flag), vals[0]])

    if B > 0:
      cmd.append('--hfta')

    succeeded = True
    try:
      subprocess.run(
          cmd,
          stdout=subprocess.DEVNULL if outdir is None else open(
              os.path.join(outdir, 'stdout.txt'),
              'w',
          ),
          stderr=subprocess.DEVNULL if outdir is None else open(
              os.path.join(outdir, 'stderr.txt'),
              'w',
          ),
          check=True,
          cwd=os.path.join(
              os.path.abspath(os.path.expanduser(os.path.dirname(__file__))),
              '../examples/transformer/',
          ),
          env=env_map,
      )
    except subprocess.CalledProcessError as e:
      logging.warning("An error has occurred during this trial, "\
                      "but the experiment could have still succeeded")
      logging.warning(e)

    # Issue 15: Plan B reliability issue with Transformer on TPU 
    # Currently when this is run on torch/xla, the script would finish
    # fine but some tear-down routine of TPU causes an abnormal exit.
    # While this does not affect the results, it does make it hard to
    # decide if the run succeeded. As a results, we rely on the existence
    # of timing.csv (dumped at the end of main.py) to determine the success
    # of the run
    if outdir is not None:
      succeeded = os.path.exists(os.path.join(outdir, "timing.csv"))
    else:
      succeeded = os.path.exists(os.path.join(tmp_path, "timing.csv"))
      shutil.rmtree(tmp_path)

    return succeeded

  if workflow(
      trial_func=trial,
      device=args.device,
      device_model=args.device_model,
      outdir_prefix=args.outdir_root,
      precs=args.precs,
      modes=args.modes,
      enable_dcgm=args.enable_dcgm,
      enable_tpu_profiler=args.enable_tpu_profiler,
      tpu_profiler_waittime=10,
      tpu_profiler_duration=10,
      epochs=args.epochs,
      iters_per_epoch=args.iters_per_epoch,
      concurrent_runner_kwargs=args.concurrent_runner_kwargs,
      mps_runner_kwargs=args.mps_runner_kwargs,
      hfta_runner_kwargs=args.hfta_runner_kwargs,
      mig_runner_kwargs=args.mig_runner_kwargs,
  ):
    logging.info('Done!')
  else:
    logging.error('Failed!')


def attach_args(
    parser=argparse.ArgumentParser('Transformer Benchmark Workflow')):
  parser.add_argument(
      '--outdir_root',
      type=str,
      required=True,
      help='output root dir',
  )
  parser.add_argument(
      '--epochs',
      type=int,
      default=5,
      help='number of epochs',
  )
  parser.add_argument(
      '--iters-per-epoch',
      type=int,
      default=1000,
      help='number of iterations per epochs',
  )
  parser.add_argument(
      '--dataroot',
      type=str,
      required=True,
      help='path to the shapenet parts dataset',
  )
  parser = attach_workflow_args(parser)
  return parser


if __name__ == '__main__':
  args = attach_args().parse_args()
  rearrange_runner_kwargs(args)
  logging.basicConfig(level=extract_logging_level(args))
  args.outdir_root = os.path.abspath(os.path.expanduser(args.outdir_root))
  args.dataroot = os.path.abspath(os.path.expanduser(args.dataroot))
  main(args)
