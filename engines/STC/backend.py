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
abstract backend class
"""

# pylint: disable=unused-argument,missing-docstring


class Backend(object):
    def __init__(self):
        self.engine_type = 'UnKnown'
        self.need_reload = False
        self.need_quant = False

    def load(self, local_file, thread_num = 4):
        raise NotImplementedError("Backend:version")

    def unload(self):
        return True

    def inference(self, tvm_inputs, repeat = 1, endpoint = None, npu_version = "npu-v1"):
        raise NotImplementedError("Backend:compile")

    def get_model_info(self):
        pass

    def get_perf(self):
        pass
