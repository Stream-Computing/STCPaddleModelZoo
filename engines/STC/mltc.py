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
    mltc local run
"""
import logging
from mltc.module import Module
import time
from .backend import Backend

logger = logging.getLogger("MLTC_RUNNER")


class Mltc(Backend):
    def __init__(self):
        self.engine_type = "Mltc"
        self.worker = None

    def __del__(
        self,
    ):
        print("local free")
        self.unload()

    def load(self, local_file, remote_path, thread_num):
        self.local_file = local_file
        self.thread_num = thread_num
        self.worker = Module()
        self.worker.load_from_stcobj(local_file + ".stcobj")
        self.output_names = [val[0] for val in self.worker.kernel_info[0].outputs]
        return True

    def unload(self):
        if self.worker:
            del self.worker
            self.worker = None

    def inference(self, inputs, repeat=1, endpoint=None, npu_version="npu-v2"):
        self.start = time.time()
        res = self.worker.run(inputs, arch=npu_version)
        self.end = time.time()

        return {name: val for name, val in zip(self.output_names, res)}

    def get_perf(self):
        latency = self.end - self.start
        return {"duration": latency}
