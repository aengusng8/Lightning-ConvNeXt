""" This main entrance of the whole project.
    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and 
    DInterface can be seen as transparent to all your args.    
"""
import os
import pandas as pd
import pytorch_lightning as pl
import argparse
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import precision_score

from data import DInterface, dataset
from model.model_interface import get_model

def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opt():

    parser = argparse.ArgumentParser()
    # Basic Training Control
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--model', default='convnext_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--train_csv_path', type=str,
                        help='dataset path')
    parser.add_argument('--test_csv_path', type=str,
                        help='dataset path')

    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--lr', type=float, default=4e-3, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')

    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)

    args = parser.parse_args()

    return args


def main(args):
    pl.seed_everything(args.seed)
    
    dataset_train = dataset.Dataset(image_path=args.data_path, 
                              df=pd.read_csv(args.train_csv_path),
                              mode="train",
                              args=args)
    dataset_val = dataset.Dataset(image_path=args.data_path, 
                              df=pd.read_csv(args.train_csv_path),
                              mode="val",
                              args=args)
    dataset_test = dataset.Dataset(image_path=args.data_path, 
                              df=pd.read_csv(args.test_csv_path),
                              mode="test",
                              args=args)
    data = DInterface(args, dataset_train, dataset_val, dataset_test)

    model = get_model(args=args, model=args.model, num_classes=args.nb_classes)
    
    # trainer = Trainer.from_argparse_args(args)
    trainer = pl.Trainer(gpus=3, 
                         accelerator='dp',
                        #  precision=16,
                    
                     logger=WandbLogger("./"),)
                    #  callbacks=[early_stop_callback],
                    #  checkpoint_callback=checkpoint_callback)
    trainer.fit(model, data)

if __name__ == '__main__':
    args = parse_opt()
    main(args)