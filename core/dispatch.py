# Copyright 2022 Stream Computing Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
    dispatch engine or dataset just in time
"""

import os
import importlib
import logging
from typing import Any, Dict, List
import json

log = logging.getLogger("Hs_mlperf")

def get_accuracy_checker(dataset_name: str):
    AccuracyChecker = importlib.import_module("datasets." + dataset_name + ".test_accuracy")
    AccuracyChecker = getattr(AccuracyChecker, "AccuracyChecker")
    return AccuracyChecker()

def load_engine(hardware_type: str):
    """
    Use python3 dyanmic load for input hardware type
    Args: str

    Returns: engine()
    """
    log.info("Loading Engine: {}".format(hardware_type))

    engine = importlib.import_module(
        "engines." + hardware_type + ".engine_" + hardware_type.lower()
    )
    engine = getattr(engine, "Engine" + hardware_type)
    return engine()

def load_stc_instance(instance_type: str):
    """
    Use python3 dyanmic load for input hardware type
    Args: str

    Returns: instance()
    """
    log.info("Loading Stc Run Instance: {}".format(instance_type))

    instance = importlib.import_module(
        "engines.STC." + instance_type.lower()
    )
    instance = getattr(instance, instance_type[0].upper() + instance_type[1:])
    return instance


def load_dataset(config: Dict[str, Any]):
    """
    Load related dataset class with workload file
    Args: Dict

    Returns: Dataloader()
    """

    dataset_name = config["dataset_name"]
    log.info("Loading Dataset: {}".format(dataset_name))

    DataLoader = importlib.import_module("datasets." + dataset_name + ".data_loader")
    DataLoader = getattr(DataLoader, "DataLoader")
    db = DataLoader(config)
    return db


def load_workload(tasks: List[str], works_path: str) -> List[Dict[str, Any]]:
    """
    Return a list of dictionary with workload

    Args: List[str]

    Returns: List[dic]
    """
    if len(tasks) > 1 and "all" in tasks:
        log.warning("[ all ] should not be grouped with other tasks")

    workloads_result = []

    modules_dir = (works_path)

    wd_filter = lambda f: not f.startswith(('_', '.')) and f.endswith('.json')
    wds = [os.path.splitext(fn)[0] for fn in filter(wd_filter, os.listdir(modules_dir))]
    if tasks and 'all' not in tasks:
        wds = [t for t in tasks if t in set(tasks) and set(wds)]

    for fn in wds:
        with open(os.path.join(modules_dir, f"{fn}.json"), "r") as f:
            workloads_result.append(json.load(f))
    
    for name in set(tasks) - set(wds):
        log.error(
            "Task name: [ {} ] was not found, please check your task name".format(
                name
            )
        )
    log.info("Loading {} workloads".format(len(workloads_result)))

    return workloads_result
