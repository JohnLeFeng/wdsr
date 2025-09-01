"""
This script convert WDSR model to OpenVINO format based on trainer.py
"""
import argparse
import importlib
import logging
import os
import sys

import torch
import openvino as ov

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dataset',
      help='Dataset name.',
      default="div2k",
      type=str,
  )
  parser.add_argument(
      '--scale',
      help='Scale factor for image super-resolution.',
      default=2,
      type=int)
  parser.add_argument(
      '--lr_patch_size',
      help='Number of pixels in height or width of LR patches.',
      default=48,
      type=int)
  parser.add_argument(
      '--ignored_boundary_size',
      help='Number of ignored boundary pixels of LR patches.',
      default=2,
      type=int)
  parser.add_argument(
      '--num_patches',
      help='Number of sampling patches per image for training.',
      default=100,
      type=int)
  parser.set_defaults(
      train_batch_size=16,
      eval_batch_size=1,
      image_mean=0.5,
      num_channels=3,
  )
  parser.add_argument(
      '--model',
      help='Model name.',
      default="wdsr",
      type=str,
  )
  parser.add_argument(
      '--job_dir',
      help='Directory to write checkpoints and export models.',
      default="X",
      type=str,
  )
  parser.add_argument(
      '--ckpt',
      help='File path to load checkpoint.',
      default=None,
      type=str,
  )
  parser.add_argument(
      '--override_epoch',
      help='Override epoch number when loading from checkpoint.',
      default=None,
      type=int,
  )
  parser.add_argument(
      '--eval_only',
      default=True,
      action='store_true',
      help='Running evaluation only.',
  )
  parser.add_argument(
      '--eval_datasets',
      help='Dataset names for evaluation.',
      default="div2k set5 bsds100 urban100",
      type=str,
      nargs='+',
  )
  # Experiment arguments
  parser.add_argument(
      '--save_checkpoints_epochs',
      help='Number of epochs to save checkpoint.',
      default=1,
      type=int)
  parser.add_argument(
      '--keep_checkpoints',
      help='Keepining intermediate checkpoints.',
      default=False,
      action='store_true')
  parser.add_argument(
      '--train_epochs',
      help='Number of epochs to run training totally.',
      default=10,
      type=int)
  parser.add_argument(
      '--log_steps',
      help='Number of steps for training logging.',
      default=100,
      type=int)
  parser.add_argument(
      '--random_seed',
      help='Random seed for TensorFlow.',
      default=None,
      type=int)
  # Performance tuning parameters
  parser.add_argument(
      '--opt_level',
      help='Number of GPUs for experiments.',
      default='O0',
      type=str)
  parser.add_argument(
      '--sync_bn',
      default=False,
      action='store_true',
      help='Enabling apex sync BN.')
  # Verbose
  parser.add_argument(
      '-v',
      '--verbose',
      action='count',
      default=0,
      help='Increasing output verbosity.',
  )
  parser.add_argument('--local_rank', default=0, type=int)
  parser.add_argument('--node_rank', default=0, type=int)
  parser.add_argument('-ih', '--input_height', type = int, default = 512,
                      help='Optional. Height of input.')
  parser.add_argument('-iw', '--input_width', type = int, default = 512,
                      help='Optional. Width of input.')
  parser.add_argument('-ds', '--do_static', default = True, action=argparse.BooleanOptionalAction,
                      help='Optional. Whether to execute text encoder.')

  # Parse arguments
  args, _ = parser.parse_known_args()
  logging.basicConfig(
      level=[logging.WARNING, logging.INFO, logging.DEBUG][args.verbose],
      format='%(asctime)s:%(levelname)s:%(message)s')

  model_module = importlib.import_module('models.' +
                                         args.model if args.model else 'models')
  model_module.update_argparser(parser)
  params = parser.parse_args()
  logging.critical(params)

  torch.backends.cudnn.benchmark = True

  params.distributed = False
  params.master_proc = True
  wdsr_model, criterion, optimizer, lr_scheduler, metrics = model_module.get_model_spec(
      params)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  wdsr_model = wdsr_model.to(device)
  criterion = criterion.to(device)

  if params.ckpt or os.path.exists(os.path.join(params.job_dir, 'latest.pth')):
    checkpoint = torch.load(
        params.ckpt or os.path.join(params.job_dir, 'latest.pth'),
        map_location=torch.device('cpu')
        )
    try:
      wdsr_model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
    except RuntimeError as e:
      logging.critical(e)
    latest_epoch = checkpoint['epoch']
    logging.critical('Loaded checkpoint from epoch {}.'.format(latest_epoch))
    if params.override_epoch is not None:
      latest_epoch = params.override_epoch
      logging.critical('Overrode epoch number to {}.'.format(latest_epoch))
  else:
    latest_epoch = 0

  HEIGHT = args.input_height
  WIDTH = args.input_width

  torch_inputs = torch.randn((1, 3, HEIGHT, WIDTH))

  if (args.do_static):
    WDSR_OV_MODEL_PATH = "./ov_models/FP16/{}_{}x_{}x{}.xml".format(args.model, args.scale, HEIGHT, WIDTH)
    ov_inputs = [1, 3, HEIGHT, WIDTH]
  else:
    WDSR_OV_MODEL_PATH = "./ov_models/FP16/{}_{}x_dyn.xml".format(args.model, args.scale)
    ov_inputs = [1, 3, -1, -1]


  wdsr_model.eval()
  with torch.no_grad():
    ov_model = ov.convert_model(wdsr_model, example_input=torch_inputs, input=ov_inputs)
    prep = ov.preprocess.PrePostProcessor(ov_model)
    prep.input(0).tensor().set_layout(ov.Layout("NCHW"))
    prep.input(0).preprocess().scale([255, 255, 255])
    ov_model = prep.build()
    ov.save_model(ov_model, WDSR_OV_MODEL_PATH)

  logging.critical("{} Model converted to IR and saved to {}".format(args.model, WDSR_OV_MODEL_PATH))

if __name__ == '__main__':
    sys.exit(main())