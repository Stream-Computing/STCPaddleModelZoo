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
    common utils
"""

from email.utils import decode_rfc2231
import os
import types
import copy
import threading
from socket import socket, AF_INET, SOCK_STREAM
from subprocess import Popen, PIPE
import logging
from .minIO_remote import get_file, get_files


log = logging.getLogger("Hs_mlperf")


def isOnnx(name: str):
    return name.upper() == "onnx".upper()


def isTf(name: str):
    return name.upper() == "tensorflow".upper()

def isPaddle(name: str):
    return name.upper() == "paddlepaddle".upper()

def isPytorch(name: str):
    return name.upper() == "pytorch".upper()


def isPb(name: str):
    return name.upper() == "pb".upper()

def isKeras(name: str):
    return name.upper() == "h5".upper() or name.upper() == "keras".upper()

def isSavedModel(name: str):
    return name.upper() == "saved_model".upper()

def isRelay(name: str):
    return name.upper() == "relay".upper()

def isParams(name: str):
    return name.upper() == "params".upper()

def isPt(name: str):
    return name.upper() == "pt".upper()


def check_ret(return_str):
    def decorator(fn):
        def wrapping(*args,**kwargs):
            res = fn(*args, **kwargs)
            temp = res[0] if isinstance(res, tuple) else res
            if temp == False:
                log.error(return_str)
                assert temp == True, return_str
            return res
        return wrapping
    return decorator


@check_ret("remote setting error.")
def check_remote(args):
    if not args.ip or not args.user_name or not args.password:
        log.error(
            "ERROR: [if you want to use remote, must at input command set ip, user_name, password.]"
        )
        return False
    return True


def port_check(host, port):
    try:
        s = socket(AF_INET, SOCK_STREAM)
        s.connect((host, port))
        s.close()
        return True
    except:
        return False


def ip_check(host):
    try:
        check = Popen("ping {} -c4\n".format(host),
                      stdin=PIPE,
                      stdout=PIPE,
                      shell=True)
        data = check.stdout.read().decode("gbk")
        if "TTL" in data or "ttl" in data:
            return True
    except:
        return False


class MyThread(threading.Thread):

    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.res = self.func(*self.args)

    def get_res(self):
        try:
            return self.res
        except Exception:
            return "failed", None

@check_ret("workload json check failed")
def workload_json_check(data):
    for key in [
            "test_perf",
            "test_accuracy",
            "test_numeric",
            "remote",
            "batch_total",
    ]:
        if key not in data:
            data[key] = False

    for key in ["batch_random", "batch_fix"]:
        if key not in data:
            data[key] = 0
        if not isinstance(data[key], int):
            log.error(
                "Workload json file error. [{} must to be int, but input type is {}]"
                .format(key, type(data[key])))
            return False
    return True

@check_ret("modelzoo json check failed")
def modelzoo_json_check(data):

    # 1. "input_range": 如果dataset_name是null，采用fake_dataset运行，指明每个输入数据的数据范围。
    # 2. "best_batch": 最优性能下的batch。 list类型，所有输入的batch都要指定
    # 3. "best_thread_num"： 最优性能下的线程数。 int类型, 如果没设置，默认8
    # 4. "layout"： 'NCHW'， 'NHWC'. 模型输入的数据类型
    # 5. "trans_format"： 'pb', 'onnx'. 模型是否要进行格式转换，转换成别的格式，进行模型转换。 如果当前文件格式和trans_format格式相同，则不进行转换。

    for key in ["model", "model_path", "framework", "nnf_version", "layout", \
            "model_precision", "inputs", "outputs", "input_shape", "input_type"]:
        if key not in data:
            log.error(
                "ModelZoo json file error. [{} not in the file. file is {}]".
                format(key, data))
            return False
        if key == "model_path" and not os.path.exists(data["model_path"]):
            is_dir = (data["model_path"].find(".") == -1)
            if is_dir:
                get_files(data["model_path"], data["model_path"])
            if len(data["model_path"].split(",")) == 2:
                get_file(data["model_path"].split(",")[0], data["model_path"].split(",")[0])
                get_file(data["model_path"].split(",")[1], data["model_path"].split(",")[1])
            else:
                get_file(data["model_path"], data["model_path"])

        if key == "model_path":
            model_split = data["model_path"].split(",")
            for cell in model_split:
                if not os.path.exists(cell):
                    log.error(
                        "ModelZoo json file error. [model_path : {} do not exists]".
                        format(data["model_path"]))
                    return False
        if key == "framework" and data["framework"].upper() not in [
                "Pytorch".upper(),
                "onnx".upper(),
                "tensorflow".upper(),
                "paddlepaddle".upper()
        ]:
            log.error(
                "ModelZoo json file error. [framework : {} , do not support .just support {pytorch, onnx, tensorflow}]"
                .format(data["framework"]))
            return False
        if key == "nnf_version" and data["nnf_version"][0] != "v":
            log.error(
                "ModelZoo json file error. [nnf_version : {} , must start by 'v']"
                .format(data["nnf_version"]))
            return False
        if key == "outputs" and len(data["outputs"]) == 0:
            log.error("ModelZoo json file error. [outputs at least has 1 name]")
            return False

        if key == "input_shape":
            input_name = data["inputs"].split(",")
            for key, val in data["input_shape"].items():
                if key not in input_name:
                    log.error(
                        "ModelZoo json file error. [input_shape name : {} not in inputs: {}]"
                        .format(key, data["input_shape"]))
                    return False
            if len(data["input_shape"]) != len(input_name):
                log.error(
                    "ModelZoo json file error. [input_name not equal input_shape, {}----{}]"
                    .format(input_name, data["input_shape"].keys()))
                return False

        if key == "input_type":
            types = data["input_type"].split(",")
            input_name = data["inputs"].split(",")
            if len(types) != len(input_name):
                log.error(
                    "ModelZoo json file error. [input_name lens not equal input_type lens]"
                    .format(input_name, data["input_shape"].keys()))
                return False

        if (key == "dataset_name" and data["dataset_name"] == "open imagenet"
                and len(input_name=data["inputs"].split(",")) != 1):
            log.error(
                "ModelZoo json file error. [imagenet dataset must has one input node, but input is {}]"
                .format(data["inputs"]))
            return False

    # ddk_config
    if "ddk_config" in data:
        get_file(data["ddk_config"], data["ddk_config"])

    # model_format
    if "model_format" not in data:
        if os.path.isdir(data["model_path"]):
            data["model_format"] = "saved_model"
        elif len(data["model_path"].split(",")) == 2:
            data["model_format"] = "relay"
        elif data["model_path"].split(".")[-1].lower() == "h5":
            data["model_format"] = "keras"
        else:
            data["model_format"] = data["model_path"].split(".")[-1].lower()     

    # set trans_format if model is H5, then trans2onnx.
    keras_trans_format = "onnx" if data["model_format"] in ["keras", "pt"] else data["model_format"]
    data["trans_format"] = data.get("trans_format", keras_trans_format)

    # generate layout
    # because some onnx model transfrom from tf, and input format is nhwc, it's so comfuse. 
    # so this parameter must be set
    if "layout" not in data:
        data["layout"] = "NCHW" if data["model_format"] in ["relay", "onnx"] else "NHWC"

    if "input_range" in data and len(data["input_range"]) != len(data["inputs"].split(",")):
        log.error(
            "ModelZoo json file error. [input range lens not equal input_name lens, input_range \
                        : {}.   inputs : {}]".format(data["input_range"],
                                                     data["inputs"]))
        return False

    if "best_batch" in data and not isinstance(data["best_batch"], int):
        log.error("ModelZoo json file error. [best_batch type must be int]")
        return False
    
    no_quote = True
    for name, shape in data["input_shape"].items():
        for a in shape:
            if isinstance(a, str):
                no_quote = False

    if no_quote:
        log.error("ModelZoo json file error. [input_shape must has best_batch quote ]")
        return False

    if not data["dataset_name"]:
        log.info("Loading Dataset: Dataset does not exist, using fake data")
        data["dataset_name"] = "fake_dataset"

    if data["dataset_name"] != "fake_dataset" and "dataset_map" not in data:
        log.error("ModelZoo json file error. [if do not use fake_dataset must to set *dataset_map* .]")
        return False

    if not data.get("input_range", ""):
        temp = {}
        for name in data["inputs"].split(","):
            temp[name] = [0, 1]
        data["input_range"] = temp

    data["custom_function"] = data.get("custom_function", "")

    return True



class Reporter():
    def __init__(self, model_info, workload, engine_type, thread_num, remote):
        self.graph_compile_report = {}
        self.base_report = {
            "Model": workload["model"].upper(),
            "Backend": engine_type,
            "Dataset": model_info["dataset_name"].upper(),
            "Thread_num": thread_num,
            "Region": "Remote" if remote else "Local"
        }
        self.base_report["Performance"] = []
        self.accuracy_report = {"Data Percent": workload["data_percent"]}
        self.base_report["Output_name"] = model_info["outputs"]


    def set_compile_time(self, time):
        self.graph_compile_report["Compile Duration"] = round(time, 5)
        self.base_report["Graph Compile"] = self.graph_compile_report

    def set_accuracy_res(self, accuracy_results):
        self.accuracy_report.update(accuracy_results)
        self.base_report["Accuracy"] = self.accuracy_report

    def set_diff_res(self, diff_results):
        self.accuracy_report["Numeric"] = diff_results
        self.accuracy_report["Diff Dist"] = diff_results["image_names"]

    def set_perf_res(self, perf_results, batch_mode, fix_data):
        perf_results["batch_mode"] = [batch_mode, fix_data]
        self.base_report["Performance"].append(perf_results)

def refix(file):
    res = {}
    performance = []
    accuracy_list = []
    for key, val in file.items():
        if key == "Performance":
            performance = copy.deepcopy(val)
            for cell in performance:
                cell["batch_size"] = cell["BATCH_SIZE"]
                cell["batch_num"] = cell["batch_mode"][1]
                cell["batch_mode"] = cell["batch_mode"][0]
                cell.pop("BATCH_SIZE")
            continue
        if key == "Graph Compile":
            for w, n in val.items():
                res[w] = n
            continue
        if key == "Accuracy":
            for a_key, a_val in val.items():
                if a_key != "Numeric":
                    if a_key == "Diff Dist":
                        continue
                    res[a_key] = a_val
                else:
                    for numeric_key, numeric_val in a_val.items():
                        if numeric_key == "image_names":
                            continue
                        cell = {}
                        cell["output_name"] = numeric_key
                        cell.update(numeric_val)
                        accuracy_list.append(cell)
            continue
        if key in ["Output_name", "Diff Dist"]:
            continue

        res[key] = val

    # recursive to get all combine data
    if not accuracy_list:
        accuracy_list = [{"Data Percent": None if "Data Percent" not in res else res["Data Percent"]}]
    ans = []
    temp = [performance, accuracy_list]
    def recursion(data, index):
        if index > 1:
            ans.append(data)
            return
        for val in temp[index]:
            recursion(dict(list(data.items()) + list(val.items())), index + 1)

    recursion(res, 0)

    return ans


def split_expression(expression):
    table = {"(", ")", "+", "-", "*", "/", " "}
    res = []
    def check(s):
        for a in s:
            if a in table or a.isdigit():
                return False
        return True

    def recursion(word):
        if not word: return []
        if check(word): return [word,]

        for i, a in enumerate(word):
            if a in table:
                return recursion(word[:i]) + recursion(word[i+1:])
        return []

    return list(set(recursion(expression)))
