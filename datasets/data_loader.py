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
dataset related classes and methods
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import numpy as np
from toolutils.common_utils import isTf

log = logging.getLogger("Dataset")

INPUT_TYPE = {"UINT8": np.uint8, "FLOAT32": np.float32, "LONG": np.long, "INT32": np.int32, "INT64": np.int64}


class Dataset:
    def __init__(self, config):
        self.config = config
        self.cur_bs = 1
        self.batched_data = []
        self.labels = []
        self.sample_lens = []
        self.items = 0
        self.batch_num = 0
        self.input_type_dict = {}
        self.remap = config.get("dataset_map", {})
        self.rev_remap = {val: key for key, val in self.remap.items()}
        for i, input in enumerate(config["inputs"].split(",")):
            self.input_type_dict[input] = config["input_type"].split(",")[i].upper()
        print(self.input_type_dict)

    def name(self):
        raise NotImplementedError("Dataset:name")

    def get_item_count(self):
        return self.items

    def get_batch_count(self):
        return self.batch_num

    def get_total_batch(self, index):
        return sum(self.sample_lens[:index])

    def preprocess(self):
        return

    def benchmark_resample(self, batch_size, model: str, fix_val=None):

        if model == "batch_random":
            high = int(fix_val)
            self.rebatch(-1, skip=True, low=1, high=high, base_batch=batch_size)
            return
        if model == "batch_fix":
            self.rebatch(fix_val if fix_val < self.items else self.items, skip=True, base_batch=batch_size)
            return
        if model == "batch_total":
            self.rebatch(self.items, skip=True, base_batch=batch_size)
            return

        raise NotImplementedError(
            "Dataset:benchmark_resample not support this model {}, \
                for now just support [random, fix_batch, total]".format(
                model
            )
        )

    def get_sequence_lens(self,):
        input_seqs_lens = []
        for name in self.config["inputs"].split(","):
            input_size = self.config["input_shape"][name]
            if len(input_size) != 2:
                log.error("ERROR: [sequence input must be 2 dims. but this input is:{}:{}]".format(name, input_size))
                raise ValueError("ERROR: [sequence input must be 2 dims. but this input is: \
                         {}:{}]".format(name, input_size))

            input_seqs_lens.append(input_size[1])
        return input_seqs_lens

    def get_image_format(self, config):
        image_format = "NCHW"
        if config.get("layout", 0):
            image_format = config.get("layout", 0)
        elif isTf(config["framework"]):
            image_format = "NHWC"
        else:
            image_format = "NCHW"
        return image_format

    def get_samples(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID is out of range")

        new_feeds = {}
        for input_tensor_name, input_value in self.batched_data[sample_id].items():
            raw_name = self.remap[input_tensor_name]
            input_tensor_dtype = self.input_type_dict[raw_name].lower()
            new_feeds[raw_name] = np.array(input_value, dtype=input_tensor_dtype)

        return new_feeds, self.labels[sample_id]



    def rebatch(self, new_bs, skip=True, low=None, high=None):
        raise NotImplementedError("Dataset:rebatch")

    def get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if input_type:
            i = 0
            for key, val in shape.items():
                val = [val[0] * batch_size] + val[1:]
                data[key] = np.random.random(size=val).astype(INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")

