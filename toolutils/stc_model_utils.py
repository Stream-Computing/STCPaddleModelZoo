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
    stc model utils
"""
import os
import pathlib
import logging
import subprocess

from toolutils.common_utils import isRelay, isParams
from .common_utils import check_ret


log = logging.getLogger("Hs_mlperf")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@check_ret("model format convert failed")
def model_convert(model_info):
    if len(model_info["model_path"].split(",")) != 2:
        return True
    # check relay format
    relay, params = model_info["model_path"].split(",")
    if isRelay(params.split(".")[-1]) and isParams(relay.split(".")[-1]):
        relay, params = params, relay
    converted_model_path = "toolutils/converted_models" + "/{}.{}".format(model_info["model"], "onnx")
    trans_cmd = ["stc_ddk.relay2onnx", "--relay", relay, "--params", params, "--save_path", converted_model_path]

    subprocess.call(trans_cmd)

    if os.path.exists(converted_model_path):
        log.info("Convertion Done")

        # update relay2onnx onnx use sim model
        # cancel to use sim model on 20230713
        model_info["model_path"] = converted_model_path
        model_info["model_format"] = model_info["framework"] = "onnx"
        return True
    return False
