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

"""
from audioop import mul
from json.encoder import py_encode_basestring
import logging
from os.path import join
import random
from math import ceil
from fractions import Fraction

import numpy as np
from datasets import data_loader
from tqdm import tqdm
from typing import List, Dict, AnyStr
import importlib

from toolutils.common_utils import split_expression

INPUT_TYPE = {
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
    "SHORT": np.int16,
    "INT": np.int32,
    "LONG": np.int64,
    "HALF": np.float16,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
    "BOOL": np.bool_,
}

log = logging.getLogger("FAKE_DATA")


class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        self.config = config
        self.cur_bs = 1e9
        self.input_range = config.get("input_range", "")
        self.items = config.get("best_batch", 1000) * 1000
        self.batch_num = int(self.items / self.cur_bs)

    def name(self):
        return "fake_dataset"

    def generate_fake_data(self, sample_id):
        input_shape = self.config["input_shape"]
        input_names = self.config["inputs"].split(",")
        input_types = self.config["input_type"].split(",")
        input_type_dict = {key: val for key, val in zip(input_names, input_types)}

        return self.get_fake_samples_regular(self.sample_lens[sample_id], input_shape, input_type_dict)

    def rebatch(self, new_bs, skip=True, low=None, high=None, base_batch=None):
        log.info("Rebatching batch size to: {} ...".format(new_bs))

        if self.cur_bs == new_bs and skip and not low and not high:
            return

        self.base_batch = base_batch or new_bs
        if not (low and high):
            self.cur_bs = new_bs
            low, high = new_bs, new_bs
        else:
            self.cur_bs = -1

        step = 0
        self.sample_lens = []
        random.seed(2014)
        while step < self.items:

            random_batch = random.randint(low, high)

            step += random_batch

            if step > self.items:
                random_batch = self.items - step + random_batch

            self.sample_lens.append(Fraction(random_batch, self.base_batch))

        self.batch_num = len(self.sample_lens)

    def get_samples(self, sample_id):
        np.random.seed(sample_id)
        return self.generate_fake_data(sample_id), []

    def _get_random_data(
        self, inputs: Dict[AnyStr, List[int]], custom_function: str, input_range: Dict[AnyStr, List[int]]
    ):
        res = {}
        if custom_function:
            custom = importlib.import_module(custom_function[:-3].replace("/", "."))
            custom = getattr(custom, "create_data")
            return custom(inputs)

        for key, val in inputs.items():
            res[key] = np.random.uniform(low=input_range[key][0], high=input_range[key][1], size=val)
        return res

    def get_fake_samples_regular(self, simple_list: int, shape: dict, input_type: dict) -> dict:
        data = {}
        input_type = {key: val.upper() for key, val in input_type.items()}
        if self.config["custom_function"] or (input_type and self.input_range):
            custom_input = {}
            for key, val in shape.items():
                # val is generate shape
                new_shape = []
                for a in val:
                    if type(a) == str:
                        raw_str = a
                        # 按照四则运算和括号分隔
                        for name in split_expression(a):
                            if name not in self.config:
                                log.error(
                                    f"[Fake_Dataset use expression is {raw_str}, but name: {name}. not in configs_names: {list(self.config)}]"
                                )
                            a = a.replace(name, f'self.config["{name}"]')
                        a = ceil(eval(a) * simple_list)
                    new_shape.append(a)

                custom_input[key] = new_shape

            data = self._get_random_data(custom_input, self.config["custom_function"], self.input_range)
            for key, val in shape.items():
                data[key] = data[key].astype(INPUT_TYPE[input_type[key]])
            return data

        else:
            raise ValueError("Please provide input type")

    def get_total_batch(self, index):
        base = self.base_batch
        res = 0
        for i in range(index):
            res += ceil(base * self.sample_lens[i])

        return res
