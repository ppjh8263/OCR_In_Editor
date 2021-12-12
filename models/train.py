import os
import json
import logging
import argparse
import wandb

from modules.models.loss import E2ELoss
from modules.models.model import OCRModel
from modules.logger.logger import Logger
from modules.data.loader import ICDARDataLoader
from modules.trainer.trainer import Trainer
from modules.models.metric import icdar_metric

import wandb

logging.basicConfig(level=logging.DEBUG, format='')


def main(config, resume):
    train_logger = Logger()

    # load data
    train_dataloader = ICDARDataLoader(config).train()
    val_dataloader = ICDARDataLoader(config).val() if config['validation']['validation_split'] > 0 else None

    # initial model
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = OCRModel(config)
    model.summary()
<<<<<<< HEAD
    wandb.init(
        project='fots',
        name='Shared_res50'
    )
    
    wandb.log(config)
    loss = E2ELoss()
    trainer = Trainer(model, loss, icdar_metric, resume, config, train_dataloader, val_dataloader, train_logger, wandb=wandb)
=======
    
    wandb.init(
        project='final_project',
        name='asdf'
    )
    
    loss = E2ELoss()
    trainer = Trainer(model, loss, icdar_metric, resume, config, train_dataloader, val_dataloader, train_logger, wandb)
>>>>>>> 96939bd6414909256d36a8641d4ab33ec1cfea7f
    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='MLT-OCR')
    parser.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    config = json.load(open('config.json'))
    if args.resume:
        logger.warning('Warning: --config overridden by --resume')

    main(config, args.resume)
