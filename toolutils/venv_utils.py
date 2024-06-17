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
    venv utils
"""

import os
import sys
import logging
import subprocess
import virtualenv
from .common_utils import check_ret


log = logging.getLogger("Hs_mlperf")

@check_ret("Activate virtualenv Failed, Please Check...")
def activate_venv(name, workload, in_verv_dir) -> bool:

    install_cmd1 = (
        "torch" if workload["framework"] == "Pytorch" else workload["framework"].lower()
    )
    install_cmd = install_cmd1 + "=={}".format(workload["nnf_version"][1:])

    if os.path.exists("venv/" + "requirements.txt"):
        log.info("Activating Virtual Env for " + name)

        venv_dir = in_verv_dir or os.path.join("venv", name)
        activate_file = os.path.join(venv_dir, "bin", "activate_this.py")
        if not os.path.exists(activate_file):
            log.info("venv not exist, Creating Virtual Env for " + name)
            virtualenv.create_environment(venv_dir)

            exec(open(activate_file).read(), {"__file__": activate_file})
            pip_path = os.path.join(venv_dir, "bin", "pip3")
            subprocess.call(
                [pip_path, "install", "-r", "requirements.txt",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            )
            subprocess.call(
                [pip_path,"install","-r","venv/" + "requirements.txt",
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            )
            subprocess.call(
                [pip_path, "install", install_cmd,
                    "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            )
        else:
            
            exec(open(activate_file).read(), {"__file__": activate_file})
            """
            just in case install failed in pre-run.
            """

            pip_path = os.path.join(venv_dir, "bin", "pip3")

            # get venv has pip package
            res = subprocess.run([pip_path, "list"], capture_output=True, encoding="utf-8")
            out_list = res.stdout
            has_package = set(
                [node.split(" ")[0] for node in out_list.split("\n") if len(node)]
            )

            with open("requirements.txt", "r") as f:
                file = f.read().splitlines()

            with open("venv/" + "requirements.txt", "r") as f:
                file += f.read().splitlines()

            for name in file:
                if name and name[0] == "#":
                    continue
                if name.split("==")[0] and (name.split("==")[0] not in has_package):
                    subprocess.call(
                        [pip_path, "install", name, "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
                    )

            subprocess.call(
                [pip_path, "install", install_cmd, "--extra-index-url", "https://download.pytorch.org/whl/cpu"]
            )
            if hasattr(sys, "real_prefix"):
                return True
            else:
                return False
    return True


def deactivate_venv(prev_sys_path, real_prefix, old_os_path):
    sys.path[:0] = prev_sys_path  # will also revert the added site-packages
    sys.prefix = real_prefix
    os.environ["PATH"] = old_os_path
