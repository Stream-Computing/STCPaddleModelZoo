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
 local stc runtime
"""
import logging
import stc
from .backend import Backend

logger = logging.getLogger("Hs_mlperf")


class Local(Backend):
    def __init__(self):
        self.engine_type = "Local"
        self.worker = None

    def __del__(self,):
        print("local free")
        self.unload()

    def load(self, local_file, remote_path, thread_num):
        self.local_file = local_file
        self.thread_num = thread_num
        self.worker = stc.STCInference(local_file, thread_num=thread_num)
        return True

    def unload(self):
        if self.worker:
            del self.worker
            self.worker = None

    def inference(self, tvm_inputs, repeat=1, endpoint=None, npu_version="npu-v1"):
        try:
            return self.worker.Run(input_dict=tvm_inputs, repeat=repeat)
        except:
            logger.error("ERROR: [local stc runtime error.]")
            return None


    def get_perf(self):
        latency = self.worker.GetLatency()
        return {"duration": latency / 1000}
