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

logger = logging.getLogger("Hs_mlperf")


class Local_ddk(Backend):
    def __init__(self):
        self.engine_type = "local_ddk"
        self.worker = None

    def __del__(self,):
        print("local ddk free")
        self.unload()

    def load(self, local_file, remote_path, thread_num):
        self.local_file = local_file
        self.thread_num = thread_num
        # thread number 是否需要？
        self.model = stc_aie.STCGraph(local_file)
        self.output_names = self.model.get_output_names()
        self.input_names = self.model.get_inputs()

        self.suffix = ":0" if list(self.input_names.keys())[0][-2:] == ":0" else ""
        self.worker = stc_aie.STCGraphExec(self.model)

        return True

    def unload(self):
        if self.worker:
            self.worker.release()
            del self.worker
            self.worker = None

    def _inference(self, input, repeat):
        ddk_input, res = [], {}

        self.suffix = "" if list(input.keys())[0][-2:] == self.suffix else self.suffix

        for key, val in input.items():
            ddk_input.append(stc_aie.STCTensor(key + self.suffix, val))

        # check input name match

        for name in self.input_names.keys():
            name = name[:len(name)-len(self.suffix)]
            if name not in list(input.keys()):
                logger.error(
                    f"ERROR: [local ddk input name mismatch. \
                    input_name : {name} not in model parse name: {list(input.keys())}]"
                )
                return None
        start = time.time()
        for _ in range(repeat):
            ddk_output = self.worker.run(ddk_input)
        duration = time.time() - start
        if len(ddk_output) != len(self.output_names):
            logger.error(
                f"ERROR: [local ddk output lens mismatch. \
                        ddk_output lens: {len(ddk_output)}; set lens: {len(self.output_names)}]"
            )
            return None

        for name, val in zip(self.output_names, ddk_output):
            res[name.rstrip(":0") if self.suffix else name] = val
        return res, duration

    def inference(self, tvm_inputs, repeat=1, endpoint=None, npu_version="npu-v1"):
        ddk_input, res = [], {}
        self.latency = 0
        # 求倍数，如果大于

        self.suffix = "" if list(tvm_inputs.keys())[0][-2:] == self.suffix else self.suffix

        base_map = {}
        multi = 0
        for key, val in self.input_names.items():
            name = key[:len(key)-len(self.suffix)]
            model_shape = val[0]
            base_map[name] = [-1, -1]

            for i, (a, b) in enumerate(zip(model_shape, tvm_inputs[name].shape)):
                if a == b:
                    continue

                base_map[name] = [i, a]
                if not multi:
                    multi = (b + a - 1) // a
                    continue
                
                if multi != (b + a - 1) // a:
                    # print(model_shape, tvm_inputs[name].shape)
                    # print(i, a, b, multi)
                    logger.error(f"ERROR: [input batch mismatch, {i, a, b, multi}]")

        res = {}
        for i in range(max(multi, 1)):
            cell_input = {}
            for key, val in tvm_inputs.items():
                if key not in base_map:
                    # some model can optimization some no use input
                    continue
                if base_map[key][0] == -1:
                    cell_input[key] = val
                else:
                    axis = base_map[key][0]
                    base_batch = base_map[key][1]
                    val_array = [a for a in range(i*base_batch, min((i+1)*base_batch, val.shape[axis]))]
                    # cell_input[key] = np.ascontiguousarray(np.take(val, val_array, axis=axis))
                    cell_input[key] = np.take(val, val_array, axis=axis)

            ans, latency = self._inference(cell_input, repeat)
            self.latency += latency
            for key, val in ans.items():
                if key not in res:
                    res[key] = val
                else:
                    res[key] = np.concatenate((res[key], val), axis=0)

        return res

    def benchmark(self, best_batch):
        self.worker.release()

        print("stc_benchmark run_params : model_path [{}], thread_num [{}], batchs [{}]".format(self.local_file, self.thread_num, str(best_batch)))
        qps = tools.stc_benchmark(self.local_file, thread_num=self.thread_num, 
                                    batchs=str(best_batch))
        avg_latency = 1 / qps

        self.worker = stc_aie.STCGraphExec(self.model)
        return qps, avg_latency

    def get_perf(self):
        return {"duration": self.latency}
