#! /usr/bin/env python3

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dm4gnc.config.config import Config
from dm4gnc.config.parser import ConfigParser
from dm4gnc.utils.random_seed import random_seed
from dm4gnc.utils import ExperimentLogger
from dm4gnc.data.dataset import Dataset
from dm4gnc.pipeline.pipeline_manager import PipelineManager

def main():
    # load config
    parser = ConfigParser.create_parser()
    args = parser.parse_args()

    config = Config.from_file(args.config_path)
    config = ConfigParser.parse_args_and_merge_config(args)

    config.validate()

    print(config)

    random_seed(config.seed)
    
    # 初始化logger
    logger = ExperimentLogger(config)

    # load dataset
    dataset_builder = Dataset(
        data_path=config.data_path,
        name=config.dataset,
        imb_level=config.imb_level,
        shuffle_seed=config.seed
    )
    dataset = dataset_builder.load_dataset()
    print(dataset)

    # run stages
    pipeline_manager = PipelineManager(config, dataset, logger=logger)
    pipeline_manager.run()
    
    # 完成日志记录
    logger.finalize()


if __name__ == "__main__":
    main()