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
 local DDK AIE runtime
"""
import logging
import time
import numpy as np
from stc_ddk import stc_aie
from stc_ddk import tools
from .backend import Backend
from run_engine import engine
import pickle

logger = logging.getLogger("Hs_mlperf")


class Thread_ddk(Backend):
    def __init__(self):
        self.engine_type = "thread_ddk"
        self.worker = None

    def __del__(self,):
        print("thread ddk free")
        self.unload()

    def load(self, local_file, remote_path, thread_num):
        self.local_file = local_file
        self.remote_path = remote_path
        self.thread_num = thread_num
        # thread number 是否需要？
        self.worker = engine(local_file, thread_num)
        self.model = stc_aie.STCGraph(local_file)
        self.output_names = self.worker.get_output_names()
        self.input_names = self.worker.get_inputs()

        self.suffix = ":0" if list(self.input_names.keys())[0][-2:] == ":0" else ""

        return True

    def unload(self):
        if self.worker:
            del self.worker
            self.worker = None

    def inference(self, tvm_inputs, repeat=1, endpoint=None, npu_version="npu-v1"):

        self.suffix = "" if list(tvm_inputs.keys())[0][-2:] == self.suffix else self.suffix
        keys = list(tvm_inputs.keys())
        new_input = {}
        for key in keys:
            name = key + self.suffix
            new_input[name] = tvm_inputs[key]
        
        start = time.time()
        # with open("model_input.pkl", "wb") as f:
        #     pickle.dump(new_input, f)
        res = self.worker.run(new_input)
        self.latency = time.time() - start

        return {(name.rstrip(":0") if self.suffix else name): val for name, val in zip(self.output_names, res)}

    def benchmark(self, best_batch):
        self.unload()

        print("stc_benchmark run_params : model_path [{}], thread_num [{}], batchs [{}]".format(self.local_file, self.thread_num, str(best_batch)))
        qps = tools.stc_benchmark(self.local_file, thread_num=self.thread_num, 
                                    batchs=str(best_batch))
        avg_latency = 1 / qps

        self.load(self.local_file, self.remote_path, self.thread_num)
        return qps, avg_latency

    def get_perf(self):
        return {"duration": self.latency}
