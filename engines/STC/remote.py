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
    remote stc runtime.
    use ssh to connect remote server, send model file, and start server
    and use https to inference
"""
import os
import time
import random
import logging
import requests
import pickle
import paramiko
from scp import SCPClient
import numpy as np

from datetime import datetime, timezone, timedelta
from subprocess import Popen, PIPE

from toolutils.common_utils import port_check, ip_check, MyThread

from .backend import Backend

logger = logging.getLogger("Hs_mlperf")


class Remote(Backend):
    def __init__(self, ip, user, password):
        # pass
        if not ip_check(ip):
            logger.error("Remote input ip:{}, not available".format(ip))
            return None

        # random to choice a port, and retry 3times
        for _ in range(3):
            self.port = random.randint(8000, 9000)
            if not port_check(ip, self.port):
                logger.info("Remote will used this ip and port. [{}:{}]".format(ip, self.port))
                break
        else:
            logger.error("Remote random choice 3 times, but do not find a available port")
            return None

        self.ip = ip
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh.connect(ip, username = user, password = password, port = '22')
        except:
            logger.error("Remote ssh connect failed, may be username or password is wrong.")
            return None

        self.engine_type = "Remoat:" + self.ip

    def __del__(self,):
        print("remote free")
        self.unload()

    def ssh_function(self, cmd_start, cmd_next=None):
        try:
            if not cmd_next:
                stdin, stdout, stderr = self.ssh.exec_command(cmd_start, get_pty=True)
            else:
                stdin, stdout, stderr = self.ssh.exec_command(cmd_start, get_pty=True)
                stdin, stdout, stderr = self.ssh.exec_command(cmd_next, get_pty=True)
        except:
            logger.error("error: remote server start error. error log : [{}]".format(stdout.read().decode()))
            # logger.error("error: remote server start error. error log : []")
        finally:
            res = stdout.read().decode()
            return "stopped", None

    def load(self, local_file, remote_path, thread_num):

        # self.endpoint = "172.16.30.118:8442"
        # return True
        test = "raw"
        self.local_file = local_file
        self.thread_num = thread_num
        now = datetime.now()
        tzinfo = timezone(timedelta(hours=8), name="BKG")
        now = now.astimezone(tzinfo)
        
        self.remote_header = remote_path
        self.file_path = os.path.join(remote_path, now.strftime("%Y%m%d_%H%M%S"))

        self.ssh_function("mkdir -p {}".format(self.file_path))

        print("local file path: {}, remote path: {}".format(local_file, self.file_path))
        self.scp = SCPClient(self.ssh.get_transport())
        self.scp.put(local_file, recursive=True, remote_path=self.file_path)
        if os.path.isdir(local_file):
            self.file_path += "/" + local_file.split("/")[-1]

        self.endpoint = self.ip + ":" + str(self.port)
        docker_cmd, cmd_start = self._gen_cmd(self.file_path, self.port, thread_num, type=test)

        print("Remote: remote run cmd : {}".format(cmd_start))
        # 创建线程
        if test == "docker":
            self.t = MyThread(self.ssh_function, (docker_cmd, cmd_start))
        else:
            self.t = MyThread(self.ssh_function, ("export STC_SET_DEVICES='4,5,6,7';" + cmd_start, ))
        self.t.setDaemon(True)
        self.t.start()

        # start server and check model correct
        time.sleep(20)
        check = Popen("curl http://{}".format(self.endpoint), stdin=PIPE, stdout=PIPE, shell=True)
        res = check.stdout.read()

        if self.t.get_res()[0] == "stopped":
            logger.error("ERROR: model load error. Remote start error.")
            return False
        return True

    def _gen_cmd(self, file_path, port, thread_num, type="n_docker"):
        if type == "docker":
            docker_cmd = "docker run -it --net=host -v {}:/simple_tensorflow_serving/models/stc_model \
                -v /usr/local/hpe:/usr/local/hpe \
                -p {}:8500 \
                --device=/dev/stc0 --device=/dev/stc0c0 \
                --device=/dev/stc0c1 --device=/dev/stc0c2 --device=/dev/stc0c3 --device=/dev/stc0ctrl \
                streamcomputing/simple_tensorflow_serving:py37 /bin/bash".format(file_path, port)
            remote_start_cmd = "simple_tensorflow_serving"
            file_path = "./models/stc_model"
            print(docker_cmd)
        else:
            remote_start_cmd = "python3 /home/zhangyunfei/simple_tensorflow_serving/simple_tensorflow_serving/server.py"
            docker_cmd = ""
        run_cmd = remote_start_cmd + " --model_base_path={} --model_platform={} --port={} --thread_num={}".format(
            file_path, "stc", port, thread_num)
        return docker_cmd, run_cmd

    def unload(self):
        try:
            if self.ssh:
                if self.file_path.find(self.remote_header) != -1:
                    logger.info("remove remote stcobj file")
                    self.ssh_function("rm -rf {}".format(self.file_path))
                self.ssh.close()
        except:
            logger.error("error: remote server stop error. ")

    def inference(self, tvm_inputs, repeat=1, endpoint=None, model_version=3, npu_version="npu-v1"):
        input_data = {"model_name": "default", "model_version": model_version, "data": {}, "repeat": repeat}
        for key, val in tvm_inputs.items():
            input_data["data"][key] = val.tolist()

        try:
            res = requests.post("http://" + (endpoint or self.endpoint), json=input_data).json()
        except Exception as e:
            save_file = "error_remote_input.pkl"
            with open(save_file, "wb") as f:
                pickle.dump(input_data, f)
            logger.error(
                "Remote run error, [input file at: {}, remote endpoint is: {}]".format(
                    save_file, endpoint or self.endpoint
                )
            )
            return {}
        result = {}
        for i, node in enumerate(res["signature"]):
            dtype = node["dtype"]
            name = node["name"]
            result[name] = np.array(res["data"][i]).astype(dtype)

        self.perf = {"duration": res["duration"], "signature": res["signature"]}
        return result

    def benchmark(self, best_batch):
        logger.info("Remote model not support benchmark")
        return -1, -1

    def get_perf(self):
        return self.perf
