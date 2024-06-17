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
from cmath import isnan
import logging
from unittest.util import _MIN_DIFF_LEN

import numpy as np
import os
from typing import Dict
from scipy import spatial
import pickle
from tqdm import tqdm
import random

log = logging.getLogger("TestAccuracy")


def cal_diff_percent(diff):

    cnt_1000, cnt_200, cnt_100, cnt_20, cnt_10 = 0, 0, 0, 0, 0
    for val in tqdm(diff):
        if val > 0.001:
            cnt_1000 += 1
        if val >= 0.005:
            cnt_200 += 1
        if val >= 0.01:
            cnt_100 += 1
        if val >= 0.05:
            cnt_20 += 1
        if val >= 0.1:
            cnt_10 += 1

    return cnt_1000 / len(diff), cnt_200 / len(diff), cnt_100 / len(diff), cnt_20 / len(diff), cnt_10 / len(diff)


def draw_all_diff(ori_outs, cur_outs, file_name):

    ori_data = ori_outs.flatten()
    cur_data = cur_outs.flatten()

    new_ori_data = []
    new_cur_data = []
    """
    cal diff compare the data except Nan inf
    """
    nan_count, inf_count = 0, 0
    max_total = int(1e7)
    total_lens = min(max_total, len(ori_data))
    if total_lens == max_total:
        sample_ids = random.sample(range(len(ori_data)), max_total)
    else:
        sample_ids = range(len(ori_data))

    for i in tqdm(sample_ids):
        if np.isnan(cur_data[i]) and not np.isnan(ori_data[i]):
            nan_count += 1
        if np.isinf(cur_data[i]) and not np.isinf(ori_data[i]):
            inf_count += 1
        if np.isnan(ori_data[i]) or np.isinf(ori_data[i]) or np.isnan(cur_data[i]) or np.isinf(cur_data[i]):
            continue
        new_ori_data.append(ori_data[i])
        new_cur_data.append(cur_data[i])

    # if all data is nan | inf, to be continue
    if not new_ori_data:
        new_ori_data.append(0)
        new_cur_data.append(1e9)

    new_ori_data = np.array(new_ori_data)
    new_cur_data = np.array(new_cur_data)

    diff = np.array(new_ori_data) - np.array(new_cur_data)
    abs_diff = np.abs(diff)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    max_val, min_val = float(max(new_cur_data)), float(min(new_cur_data))
    rms_val = float(np.sqrt(np.mean(np.power(new_cur_data.astype("float32"), 2))))

    rel_diff = abs_diff / np.abs(new_ori_data)
    rel_diff[np.isnan(rel_diff)] = 0.0
    rel_diff[np.isinf(rel_diff)] = 0.0
    max_rel_diff = float(np.max(rel_diff))
    mean_rel_diff = float(np.mean(rel_diff))

    p1000, p200, p100, p20, p10 = cal_diff_percent(diff)
    rp1000, rp200, rp100, rp20, rp10 = cal_diff_percent(rel_diff)

    cos_sim = float(1 - spatial.distance.cosine(new_ori_data, new_cur_data))

    res = {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "cos_sim": cos_sim,
        "max_val": max_val,
        "min_val": min_val,
        "rms_val": rms_val,
        "Nan_percent": nan_count / len(ori_data),
        "Inf_percent": inf_count / len(ori_data),
        "total_count": len(ori_data),
        "d0.001_percent": [p1000, rp1000],
        "d0.005_percent": [p200, rp200],
        "d0.01_percent": [p100, rp100],
        "d0.05_percent": [p20, rp20],
        "d0.1_percent": [p10, rp10],
    }

    log.info(
        "Max Abs Diff: {}, Mean Abs Diff: {}, Max Rel Diff: {}, Mean Rel Diff : {}, Cos Sim : {}, Max Val : {} \
              Min Val : {}, RMS Val : {}, Nan Percent : {}, Inf Percent : {}, Total count : {}".format(
            max_abs_diff,
            mean_abs_diff,
            max_rel_diff,
            mean_rel_diff,
            cos_sim,
            max_val,
            min_val,
            rms_val,
            res["Nan_percent"],
            res["Inf_percent"],
            res["total_count"],
        )
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 8))
    plt.cla()

    lens = min(new_ori_data.shape[0], 1000)
    plt.subplot(1, 3, 1)
    plt.yscale("log")
    plt.hist(diff, bins=lens, alpha=0.5, label="Diff", range=(diff.min(), diff.max()))
    plt.xlabel("Diff Distribute")

    plt.subplot(1, 3, 2)
    plt.yscale("log")
    plt.hist(new_ori_data, bins=lens, alpha=0.5, label="CPU", range=(new_ori_data.min(), new_ori_data.max()))
    plt.xlabel("CPU Result")

    plt.subplot(1, 3, 3)
    plt.yscale("log")
    new_cur_data = new_cur_data.astype("float32")
    plt.hist(new_cur_data, bins=lens, alpha=0.5, label="NPU", range=(new_cur_data.min(), new_cur_data.max()))
    plt.xlabel("NPU Result")

    plt.savefig(file_name, dpi=300)
    return res


class AccuracyChecker:
    def __init__(self):
        self.configs = None
        self.dataloader = None
        self.compiled_model = None
        self.batch_size = -1

    def update(self, engine, model_info):
        self.configs = model_info
        self.batch_size = engine.get_loaded_batch_size()
        self.updata_engine(engine)
        self.input_range = model_info["input_range"]
        self._update_output_dir()

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def _update_output_dir(self):
        self.output_dir = os.path.abspath(
            "reports/{}/".format(self.compiled_model.hardware_type) + self.configs["model"]
        )
        self.cpu_output_dir = os.path.abspath("reports/CPU/" + self.configs["model"])

    def updata_engine(self, engine):
        self.compiled_model = engine
        self._update_output_dir()
        engine.configs = self.configs
        self._create_report_path()

    def _create_report_path(self):
        # set report dir and init report
        self._update_output_dir()
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cpu_output_dir, exist_ok=True)

    def get_data_path(self):
        self._update_output_dir()
        if len(str(self.batch_size)) > 150:
            suffix_name = hash(str(self.batch_size))
        else:
            suffix_name = str(self.batch_size)
        return "{}.pkl".format(self.dataloader.name() + "-" + str(suffix_name))

    def get_cpu_data_filepath(self):
        return self.cpu_output_dir + "/" + self.get_data_path()

    def get_vendor_data_filepath(self):
        return self.output_dir + "/" + self.get_data_path()

    def get_output_dir(self):
        self._update_output_dir()
        return self.output_dir

    def calculate_diff(self) -> Dict[str, float]:
        print("---------------------------")
        print(self.configs, self.dataloader.name())
        """
        Return a dictionary of Mean Diff, Std Diff and Max Diff

        Args: None

        Returns: Dict[str, float]
        """
        cpu_data_filepath = self.get_cpu_data_filepath()
        vendor_data_filepath = self.get_vendor_data_filepath()

        if not os.path.exists(cpu_data_filepath):
            log.error("Fetch CPU Data Failed")
            return {}
        outputs = self.configs["outputs"].split(",")

        with open(vendor_data_filepath, "rb") as f:
            vendor_data = pickle.load(f)

        with open(cpu_data_filepath, "rb") as f:
            cpu_data = pickle.load(f)

        digit = all([isinstance(val, int) for val in vendor_data.keys()])
        res = {}
        image_names = []
        for i, name in enumerate(outputs):
            name = i if digit else name
            image_name = "b@b" + str(i) if digit else name
            # some model output node name is "xxxx/xxxx/xxx"
            image_name = image_name.replace("/", "-")
            if name not in cpu_data or name not in vendor_data:
                log.error(f"ERROR: [config output_name {name} not match model output name {list(cpu_data.keys())}.]")
                return {}
            ans = draw_all_diff(
                cpu_data[name],
                vendor_data[name],
                self.output_dir + "/" + self.configs["model"] + "-{}".format(image_name) + ".png",
            )
            image_names.append(self.configs["model"] + "-{}".format(image_name) + ".png")
            res[outputs[i]] = ans
        res["image_names"] = image_names
        return res

    def calculate_acc(self, data_percent):
        raise NotImplementedError("Dataset: caculate_acc")
