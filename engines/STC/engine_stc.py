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
backend of stc
"""

import os
import shutil
import subprocess
import atexit
import logging
import numpy as np
import time
import json
from pathlib import Path

from engines import engine
from tqdm import tqdm
import sys
from core.dispatch import load_stc_instance

from toolutils.common_utils import check_ret, isOnnx, split_expression
from toolutils.mltc_utils import convert_mltc_dtype


sys.path.append(os.path.dirname(__file__))
log = logging.getLogger("Hs_mlperf")
log.setLevel(logging.INFO)


class EngineSTC(engine.Engine):
    TARGET = "stc_tc"

    def __init__(self, compile_dir="./engines/STC/mix_tmp/"):
        self.frontend_util = None
        self.need_quant = False
        self.hardware_type = "STC"
        self.stc_dtype = "float16"
        self.object_suffix = ".stcobj"
        self.tmpdir = compile_dir
        self.remote_path = "~/models/"
        self.dataset_name_batch = {}
        self.tmpfiles = set()
        self.pid = None
        atexit.register(self.__del)
        self.stc_infer = None

    def __del(self):
        if self.frontend_util is not None:
            self.tmpfiles.update(self.frontend_util.gc())
        for tmpfile in self.tmpfiles:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)

    def version(self):
        return "2.3"

    def get_loaded_batch_size(self):
        if self.best_batch is None:
            log.error("There is no loaded_batch_size. Please call pre_optimize to infer the loaded_batch_size.")
        return self.best_batch

    def update_compile_data(self, info, npu_version="npu-v1"):
        self.best_batch = info["best_batch"]
        self.npu_version = npu_version

    def repeat_batch(self, input: dict, repeat: int):
        res = {}
        cell_batch = 0
        for key, val in input.items():
            if not cell_batch:
                cell_batch = val.shape[0]
            res[key] = val.repeat(repeat, axis=0)
        return res, cell_batch

    def benchmark(self, dataloader, percent=100, pure_mode=False, npu_version="npu-v1"):
        remote_latencies = []
        latencies = []
        max_loop = 100
        print("start benchmark.....")
        if not pure_mode:
            # use dataloader data to cal real performance
            total = ((max(percent, 0) * dataloader.get_batch_count()) // 100) or 1
            total = min(total, max_loop)
            print(pure_mode, "total rounds", total)
            for i in tqdm(range(total)):
                in_data = dataloader.get_samples(i)[0]
                start = time.time()
                res = self.backend.inference(in_data, npu_version=npu_version)
                latencies.append(time.time() - start)
                del res
                remote_latencies.append(self.backend.get_perf()["duration"])

            total_batch = dataloader.get_total_batch(total)

            fps = round(total_batch / sum(latencies), 1)
            pure_fps = round(total_batch / sum(remote_latencies), 1)

        else:
            # use one data, to generate thread_nums batch-model to cal performance
            qps, latency = self.backend.benchmark(self.best_batch)
            pid = os.getpid()
            os.environ["PID"] = str(pid)
            latencies.append(latency)
            remote_latencies = latencies
            total_batch = -1
            fps = qps * self.best_batch

        print("pure_mode : ", pure_mode, "total batch : ", total_batch)
        print("latencies: ", latencies)
        print("remote_latencies: ", remote_latencies)

        avg_latency = np.mean(latencies)

        perf_dict = {}
        perf_dict["BATCH_SIZE"] = self.best_batch
        perf_dict["FPS"] = round(fps, 5)
        # perf_dict["PURE_FPS"] = round(pure_fps, 5)
        perf_dict["AVG_Latency"] = round(avg_latency, 5)
        latencies.sort()
        p99_latency = round(latencies[int(len(latencies) * 0.99)], 5)
        perf_dict["P99_Latency"] = np.mean(p99_latency)

        return perf_dict

    @check_ret("pre_optimize failed")
    def pre_optimize(self, pre_compile_config):
        logging.root.level = logging.WARNING
        log.info("Running Backend Pre Compilation...")
        self.configs = pre_compile_config
        self.model_name = self.configs["model"]
        # 做模型转换，把动态batch改成固定batch
        return True

    @check_ret("compile failed")
    def compile(self, pre_compile_config):
        logging.root.level = logging.WARNING
        log.info("Running Backend Compilation...")
        self.configs = pre_compile_config["model_info"]
        self.compile_tools = pre_compile_config.get("compile_tools", None)
        self.compile_args = pre_compile_config.get("compile_args", "")
        # 用命令来编译模型

        # stc --model clear_model.pb --input_format "NCHW" --input_shape "input_tensor:8,224,224,3" --output "resnet50_tf"

        # {'workload': {'model': 'open_resnet50-tf-fp32', 'test_perf': True, 'test_accuracy': True, 'test_numeric': True, 'clients': 3,
        #                 'iterations': 1, 'batch_sizes': [1, 4, 8, 16, 32, 64], 'data_percent': 100, 'compile_only': False},
        # 'model_info': {'model': 'open_resnet50-tf-fp32', 'model_path': 'model_zoo/open/regular/open_resnet50/resnet50-fp32',
        #                 'framework': 'Tensorflow', 'nnf_version': 'v2.4.0', 'model_format': 'saved_model', 'model_precision': 'FP32',
        #                 'inputs': 'input_tensor:0', 'outputs': 'softmax_tensor:0', 'input_shape': {'input_tensor:0': [1, 224, 224, 3]},
        #                 'input_type': 'FLOAT32', 'dataset_name': 'open_imagenet', 'max_batch_size': 64, 'layout': 'NHWC', 'best_batch': [8]}}
        self.update_compile_data(self.configs, pre_compile_config.get("npu_version", "npu-v1"))

        self.best_batch = self.configs["best_batch"]
        model_name = self.configs["model"]

        compile_info = {}
        compile_info["model"] = self.model_name
        compile_info["compile_precision"] = "fp16"
        compile_info["best_batch"] = self.configs["best_batch"]

        compile_info["framework"] = self.configs["framework"]
        compile_info["input_type"] = self.configs["input_type"]

        def gen_mix_cmd():
            input_name = self.configs["inputs"]
            input_shapes = []
            outputs = ""

            for name in input_name.split(","):
                shape = self.configs["input_shape"][name]
                new_shape = []
                for a in shape:
                    if type(a) == str:
                        raw_str = a
                        for name in split_expression(a):
                            if name not in self.configs:
                                log.error(
                                    f"[Fake_Dataset use expression is {raw_str}, but name: {name}. not in configs_names: {list(self.configs)}]"
                                )
                            a = a.replace(name, f'self.configs["{name}"]')
                        a = eval(a)
                    new_shape.append(a)

                input_shapes.append("[" + ",".join(str(val) for val in new_shape) + "]")

            input_shapes = ",".join(val for val in input_shapes)
            outputs = self.configs["outputs"]
            output_num = len(self.configs["outputs"].split(","))
            input_dtypes = self.configs["input_type"]
            output_dtypes = ",".join(self.configs["model_precision"] for _ in range(output_num))

            res_path = os.path.join(self.tmpdir, model_name)

            out_cmd = [
                "stc_ddk.stc_aic",
                "--model",
                self.configs["model_path"],
                "--input_names",
                input_name,
                "--input_shapes",
                input_shapes,
                "--input_dtypes",
                input_dtypes,
                "--output_names",
                outputs,
                "--output_dtypes",
                output_dtypes,
                "--outdir",
                res_path,
            ]
            if "ddk_config" in self.configs:
                out_cmd += ["--config", self.configs["ddk_config"]]
            if isOnnx(self.configs["trans_format"]) and not isOnnx(self.configs["model_format"]):
                out_cmd += ["--to_onnx"]

            return out_cmd, res_path

        def gen_mltc_to_mlir(npu_version):
            input_names = self.configs["inputs"]
            input_shapes = []
            outputs = ""

            for input_name, input_dtype in zip(input_names.split(","), self.configs["input_type"].split(",")):
                shape = self.configs["input_shape"][input_name]
                new_shape = []
                for a in shape:
                    if type(a) == str:
                        raw_str = a
                        for name in split_expression(a):
                            if name not in self.configs:
                                log.error(
                                    f"[Fake_Dataset use expression is {raw_str}, but name: {name}. not in configs_names: {list(self.configs)}]"
                                )
                            a = a.replace(name, f'self.configs["{name}"]')
                        a = eval(a)
                    new_shape.append(a)

                input_shapes.append(
                    input_name + ":" + "x".join(str(a) for a in new_shape) + "x" + convert_mltc_dtype(input_dtype)
                )

            input_shapes = ",".join(val for val in input_shapes)
            outputs = self.configs["outputs"]

            res_path = os.path.join(self.tmpdir, model_name)

            if isOnnx(self.configs["model_format"]):
                drops = input_shapes
                drops = drops.split(",")
                drops = ["x".join(val.split("x")[:-1]) for val in drops]
                drops = ",".join(val for val in drops)

                out_cmd = ["mltc-onnx", "-s", self.configs["model_path"], "-i", drops, "-o", res_path + ".mlir"]

            else:
                out_cmd = [
                    "mltc-tf",
                    "-s",
                    self.configs["model_path"],
                    "-i",
                    input_shapes,
                    "-t",
                    outputs,
                    "-o",
                    res_path + ".mlir",
                ]

            out_cmd.extend(["&&", "mltc-be", f"-arch={npu_version}", res_path + ".mlir", "-o", res_path + ".stcobj"])

            return out_cmd, res_path + ".stcobj"

        if self.compile_tools == "mltc":
            out_cmd, res_path = gen_mltc_to_mlir(self.npu_version)

            cmd_words = " ".join(str(val) for val in out_cmd) + " " + self.compile_args
            if os.path.exists(res_path):
                os.remove(res_path)
        else:
            out_cmd, res_path = gen_mix_cmd()
            cmd_words = " ".join(str(val) for val in out_cmd)
            if os.path.exists(res_path):
                shutil.rmtree(res_path)

        # os.system(cmd_words)
        try:
            print(cmd_words)
            process = subprocess.Popen(cmd_words, shell=True)
            pid = process.pid
            os.environ["PID"] = str(pid)
            process.wait()
        except:
            print(cmd_words)
            print("try to avoid Broken pipe exception")
        if process.returncode != 0:
            log.error(f"Failed shell_cmd: {out_cmd}")
            sys.exit(process.returncode)

        if not os.path.exists(res_path) or (self.compile_tools != "mltc" and self.check_aic(res_path)):
            run_cmd = " ".join(str(val) for val in out_cmd)
            log.error("model convert error. run_cmd is : {}".format(run_cmd))
            compile_info["compile_status"] = "failed"
        else:
            compile_info["compile_status"] = "success"

        return compile_info["compile_status"] == "success", compile_info

    def predict(self, feeds):
        stc_out = self.backend.inference(feeds, npu_version = self.npu_version)
        return stc_out

    @check_ret("model Load failed")
    def load_model(self, thread_num=4):
        res = self.backend.load(self.tmpdir + self.model_name, self.remote_path, thread_num)
        if not res:
            self.backend.unload()
        return res

    def unload_model(self):
        self.backend.unload()

    def switch_to_remote(self, ip, user, password):
        self.backend = load_stc_instance("remote")(ip, user, password)

    def switch_to_local(self, version="thread_ddk"):
        self.backend = load_stc_instance(version)()

    def check_aic(self, res_path):
        aic_fail_flag = False
        p = Path(res_path)
        json_file = p / "model.json"
        if json_file.exists():
            with open(str(json_file), "r") as f:
                model_info = json.load(f)
            if len(model_info["nodes"]) > 0:
                for node in model_info["nodes"]:
                    file = p / node["source"]
                    if not file.exists():
                        aic_fail_flag = True
                        break
            else:
                aic_fail_flag = True
        else:
            aic_fail_flag = True
        return aic_fail_flag
