import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
import math
import shutil
import pickle as pkl
import copy
import yaml
import pdb
from tqdm import tqdm
from typing import List, Dict

import transformers
from transformers.adapters import AdapterConfig
from transformers import PfeifferConfig, HoulsbyConfig, ParallelConfig, CompacterConfig
from transformers.adapters.configuration import DynamicAdapterFusionConfig
import transformers.adapters.composition as ac

from continual_learner import ContinualLearner

logger = logging.getLogger(__name__)


ADAPTER_MAP = {
    'pfeiffer': PfeifferConfig,
    'houlsby': HoulsbyConfig,
    'parallel': ParallelConfig,
    'compacter': CompacterConfig,
}

SUPPORTED_ADAPTER_METHODS = ['vanilla', 'fusion']
SUPPORTED_FUSION_METHODS = ['bert-fusion', 'weighted-composition']

class AdapterHandler:

    def __init__(self, adapter_config, args,device):

        self.args = args
        self.device = device
        self.adapter_config = adapter_config

        individual_adapter_config = AdapterConfig.load(adapter_config['adapter_type'])
        config_dict = individual_adapter_config.to_dict()
        config_dict['reduction_factor'] = adapter_config['reduction_factor']
        self.individual_adapter_config = AdapterConfig.from_dict(config_dict)
        
        logger.info("Adding Adapter layers, initialized with configuration:")
        logger.info(str(individual_adapter_config))

    def add_adapter(self, model: ContinualLearner, task_key: str):
        model.get_backbone_transformer().add_adapter(task_key, config=self.individual_adapter_config)
        model.to(self.device)
        logger.info("Added {} Adapter".format(task_key))

    def add_adapters_to_model(self, model: ContinualLearner, task_keys: List[str]):
        for task_key in task_keys:
            self.add_adapter(model, task_key)
        logger.info("Added Adapters for tasks: {}".format(', '.join(task_keys)))
        logger.info("Total Adapter params = {:.2f}M ({:.2f}% of full encoder)".format(self.get_total_adapter_params(model)*10**-6,
                                                                      self.get_total_adapter_params(model)/(model.get_total_encoder_params() - self.get_total_adapter_params(model))*100.0))

    def activate_adapter_for_training(self, task_key: str, model: ContinualLearner):
        model.get_backbone_transformer().train_adapter(task_key)
        logger.info("Activated {} Adapter for training".format(task_key))

    def activate_adapter_for_eval(self, task_key: str, model: ContinualLearner):
        model.get_backbone_transformer().set_active_adapters(task_key)
        logger.info("Set active {} Adapter".format(task_key))

    def get_active_adapters(self, model: ContinualLearner):
        return model.get_backbone_transformer().active_adapters

    def delete_adapter(self, model: ContinualLearner, task_key: str):
        model.get_backbone_transformer().delete_adapter(task_key)
        logger.info("Deleted {} Adapter from model".format(task_key))

    def save_adapter(self, model: ContinualLearner, task_key: str, experiment_output_dir: str):
        adapter_save_dir = os.path.join(experiment_output_dir, 'adapters', task_key)
        if not os.path.exists(adapter_save_dir):
            os.makedirs(adapter_save_dir)
        model.get_backbone_transformer().save_adapter(adapter_save_dir, task_key)
        logger.info("Saved {} Adapter to {}".format(task_key, adapter_save_dir))

    def load_adapter(self, model: ContinualLearner, task_key: str, adapter_dir: str):
        model.get_backbone_transformer().load_adapter(adapter_name_or_path=adapter_dir, load_as=task_key)
        logger.info("Loaded {} Adapter from path {}".format(task_key, adapter_dir))

    def unfreeze_adapter(self, model: ContinualLearner, task_key: str):
        model.get_backbone_transformer().apply_to_adapter_layers(lambda i, layer: layer.enable_adapters(ac.Stack(task_key), True, False))
        logger.info("Adapter {} has been un-frozen".format(task_key))

    def get_total_adapter_params(self, model: ContinualLearner):
        return sum([p.numel() for n, p in model.named_parameters() if 'adapter' in n])

    def get_trainable_adapter_params(self, model: ContinualLearner):
        return sum([p.numel() for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad == True])

class AdapterFusionHandler(AdapterHandler):

    def __init__(self, adapter_config, args):

        super(AdapterFusionHandler, self).__init__(adapter_config=adapter_config,
                                                   args=args)

        fusion_config = adapter_config['fusion_config']
        self.fusion_config = fusion_config
        self.fusion_method = fusion_config['fusion_method']

        self.init_fusion_config = dict(DynamicAdapterFusionConfig())
        self.init_fusion_config.update(fusion_config)

        logger.info("Creating AdapterFusionHandler, with fusion method: {}".format(self.fusion_method))

    def add_adapter_fusion(self, model: ContinualLearner, adapter_names: List[str]):
        adapter_setup = ac.Fuse(*adapter_names)
        model.get_backbone_transformer().add_adapter_fusion(adapter_setup, 
                                                            config=self.init_fusion_config)
        logger.info("Added AdapterFusion for tasks: {}".format(', '.join(adapter_names)))
        model.to(self.device)

    def train_adapter_fusion(self, model: ContinualLearner, adapter_names: List[str], unfreeze_adapters: bool = False):
        adapter_setup = ac.Fuse(*adapter_names)
        model.get_backbone_transformer().train_adapter_fusion(adapter_setup, unfreeze_adapters)
        logger.info("Activated AdapterFusion({}) for training".format(",".join(adapter_names)))

def create_adapter_handler(adapter_config: Dict, args, device) -> AdapterHandler:

    adapter_method = adapter_config['method']
    assert adapter_method in SUPPORTED_ADAPTER_METHODS
    if adapter_method == 'vanilla':
        return AdapterHandler(adapter_config=adapter_config, args=args,device=device)
    elif adapter_method == 'fusion':
        return AdapterFusionHandler(adapter_config=adapter_config, args=args,device=device)

