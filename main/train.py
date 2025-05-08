import os
import argparse
import __init_path

from core.config import cfg, update_config

parser = argparse.ArgumentParser(description='Train HOITG')
parser.add_argument('--resume_training', action='store_true', help='resume training')
parser.add_argument('--gpu', type=str, default='0,1', help='assign multi-gpus by comma concat')
parser.add_argument('--dataset', type=str, default='behave', choices=['behave', 'intercap'], help='dataset')
parser.add_argument('--exp', type=str, default='', help='assign experiments directory')
parser.add_argument('--checkpoint', type=str, default='', help='model path for resuming')


# Organize arguments
args = parser.parse_args()
update_config(dataset_name=args.dataset.lower(), exp_dir=args.exp, ckpt_path=args.checkpoint)

from core.config import logger
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
logger.info(f"Work on GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
logger.info(f"Args: {args}")
logger.info(f"Cfg: {cfg}")

logdir = cfg.log_dir
# Prepare trainer and tester
from core.hoitgbase import Trainer, Tester
from train_utils import save_checkpoint, check_data_parallel
trainer = Trainer(args, load_dir=cfg.MODEL.weight_path,log_dirr=logdir)
tester = Tester(args)


# Initalize evaluation history
if hasattr(trainer, 'eval_history'):
    tester.eval_history = trainer.eval_history


# Train HOITG
logger.info(f"===> Start training...")
for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
    trainer.run(epoch)
    
    # Validate HOITG
    if epoch % 10 == 1 or epoch == cfg.TRAIN.end_epoch:
        tester.run(epoch, current_model=trainer.model)    
        tester.save_history(tester.eval_history)

    # Save checkpoint of HOITG
    if epoch % 10 == 1 or epoch == cfg.TRAIN.end_epoch:
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': check_data_parallel(trainer.model.state_dict()),
            'optim_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
            'train_log': trainer.loss_history,
            'test_log': tester.eval_history
        }, epoch)