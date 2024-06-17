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
convert model json
1. from model_path to generation two json file: [model_zoo-xxx.json, workloads-xxx.json]
2. read model, and dependence the model input name, input shape, input dtype.
3. model pre convert rules
    base format        convert format
    relay               onnx
    h5                  pb 
"""

import os
import json
import subprocess
import warnings
import collections

import numpy as np
import torch
import onnx
import tensorflow as tf
import pandas as pd
from glob import glob

from toolutils.minIO_remote import get_file, get_files

warnings.filterwarnings("ignore")

# class Base
class BaseWorkLoads:
    def __init__(self,):
        pass

    def to_json(self,):
        pass


class BaseRead:
    def __init__(self):
        pass

    def get_shape(self, node):
        return None

    def get_from_path(self, model_file):
        return None

    def load_model(self, file):
        return None


class NormalWorks(BaseWorkLoads):
    def __init__(self, model_name):
        self.model = model_name
        self.test_perf = True
        self.test_accuracy = True
        self.test_numeric = True
        self.remote = False
        self.iterations = 1
        self.batch_sizes = [1]
        self.data_percent = 10
        self.compile_only = False

    def to_json(self):
        res = {}
        res["model"] = self.model
        res["test_perf"] = self.test_perf
        res["test_accuracy"] = self.test_accuracy
        res["test_numeric"] = self.test_numeric
        res["remote"] = self.remote
        # res["batch_random"] = 30
        # res["batch_fix"] = 800
        res["batch_total"] = False
        res["iterations"] = self.iterations
        res["batch_sizes"] = self.batch_sizes
        res["data_percent"] = self.data_percent
        res["compile_only"] = self.compile_only
        return res


class GetOnnx(BaseRead):
    def get_shape(self, node):
        table = {
            1: "float32",
            2: "uint8",
            3: "int8",
            4: "uint16",
            5: "int16",
            6: "int32",
            7: "int64",
            8: "string",
            9: "boolean",
            10: "float16",
            11: "float64",
            12: "uint32",
            14: "uint64",
            15: "complex128",
            16: "bfloat16",
        }

        shape = node.type.tensor_type.shape.dim
        shape = [val.dim_value if val.dim_value else -1 for val in shape]
        element_id = node.type.tensor_type.elem_type
        type_str = table[element_id].upper()

        return shape, type_str

    def get_from_path(self, model_file):
        in_names, out_names = [], []
        in_types, out_types = [], []
        in_shapes = {}
        model = self.load_model(model_file)
        intitial = set()
        for node in model.graph.initializer:
            intitial.add(node.name)
        for input in model.graph.input:
            if input.name in intitial:
                continue

            in_names.append(input.name)
            shape, type_str = self.get_shape(input)
            in_types.append(type_str)
            in_shapes[input.name] = shape

        for output in model.graph.output:
            shape, type_str = self.get_shape(output)
            # because some model, read output is two, but one is empty
            if shape:
                out_names.append(output.name)
                out_types.append(type_str)

        return in_names, out_names, in_types, in_shapes

    def load_model(self, file):
        return onnx.load(file)


class GetTf(BaseRead):
    def get_shape(self, node):
        shape = node.attr["shape"].shape.dim
        shape = [val.size for val in shape]
        trans = {"HALF":"FLOAT16", "FLOAT":"FLOAT32"}
        type_str = node.attr["dtype"].__str__().split("_")[-1][:-1]

        return shape, type_str if type_str not in trans else trans[type_str]

    def get_from_path(self, model_file):
        in_names, out_names = [], []
        in_types, out_types = [], []
        in_shapes, table, vis = {}, set(), set()
        graph = collections.defaultdict(list)

        def recursion(root):
            if not root or root in vis:
                return
            vis.add(root)
            for next in graph[root]:
                recursion(next)

        graph_def = self.load_model(model_file)
        for i, node in enumerate(graph_def.node):
            if node.input:
                for name in node.input:
                    real_name = node.name if node.name[-2:] != ":0" else node.name[:-2]
                    real_name = real_name[1:] if real_name[0] == "^" else real_name
                    b_name = name if name[-2:] != ":0" else name[:-2]
                    b_name = b_name[1:] if b_name[0] == "^" else b_name
                    graph[b_name].append(real_name)

            for name in node.input:
                real_name = name if name[-2:] != ":0" else name[:-2]
                real_name = real_name[1:] if real_name[0] == "^" else real_name
                table.add(real_name)
            if node.op == "Placeholder":
                in_names.append(node.name)
                shape, type_str = self.get_shape(node)
                in_types.append(type_str)
                in_shapes[node.name] = shape


        for start in in_names:
            recursion(start)

        for node in graph_def.node:
            if node.name not in table and node.name in vis:
                out_names.append(node.name)

        return in_names, out_names, in_types, in_shapes

    def load_model(self, file):
        if os.path.isdir(file):
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    meta_graph_def = tf.compat.v1.saved_model.loader.load(sess, ["serve"], file)
                    graph_def = sess.graph_def
        else:
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(str(file), "rb") as f:
                graph_def.ParseFromString(f.read())

        return graph_def


def relay2onnx(input_model, input_params, output_model):
    cmd = ["stc_ddk.relay2onnx", "-r", input_model, "-p", input_params, "-s", output_model]
    subprocess.call(cmd)


def h52pb(input_model, output_model):
    cmd = ["bash", "toolutils/convert.sh", input_model, output_model, "keras2pb"]
    subprocess.call(cmd)


def get_input_from_csv(df, model_path):

    cell = df[df["remote_model_path"] == model_path]
    outputs = cell["outputs"].values[0]
    if np.isnan(outputs):
        return None, None, None, None
    if outputs[0] == "[":
        outputs = ",".join(eval(outputs))
    in_names = [val.lstrip().rstrip() for val in cell["inputs"].values[0].split(",")]
    in_shapes = eval(cell["input_shape"].values[0])
    
    in_types = [val.lstrip().rstrip() for val in cell["input_type"].values[0].split(",")]
    out_names = [val.lstrip().rstrip() for val in outputs.split(",")]

    return in_names, out_names, in_types, in_shapes


class Process:
    def __init__(self, model_name, model_path):
        self.model_name = model_name
        self.model_path = model_path

        self.dataset_name = "fake_dataset"
        self.framework = ""
        self.nnf_version = ""
        self.model_format = ""
        self.trans_format = ""
        self.inputs = []
        self.outputs = []
        self.input_shape = {}
        self.input_type = []
        self.layout = ""
        self.shape_reader = None
        self.cvs_reader = get_input_from_csv
        self.res = {}
        self.res["model_path"] = model_path

    def get_info(self, csv_df=None):
        if self.shape_reader:
            self.inputs, self.outputs, self.input_type, self.input_shape = self.shape_reader(self.model_path)
            # 因为pb文件有一个bug，已知两种：
            # 1. IteratorV2 算子，没法读取输入信息。 [input_name都是空的。] 直接报错。
            # 2. placeholder 没有shape信息，只能通过csv读进去，或者认为指定。[input name 有shape没有] 尝试从csv读取。
            # 3. 部分模型的H W 轴也是动态的， 这种场景，需要去csv读一下，看看有没有正常的数据。
            check_neg = any([val == -1 for cell in self.input_shape.values() for val in cell])
            if self.inputs and check_neg and isinstance(csv_df, pd.DataFrame):
                _, _, _, csv_input_shape = self.cvs_reader(csv_df, self.res["model_path"])
                if csv_input_shape:
                    # check csv file correctily
                    for name in self.input_shape.keys():
                        if name not in list(csv_input_shape.keys()):
                            print(f"Error: model read input_name not same as csv_file: \
                                    input name : {name}, csv_name : {list(csv_input_shape.keys())}")
                            return False
                    self.input_shape = csv_input_shape
            if self.inputs and not self.input_shape[self.inputs[0]] and isinstance(csv_df, pd.DataFrame):
                self.inputs, self.outputs, self.input_type, self.input_shape = self.cvs_reader(csv_df, self.res["model_path"])

        else:
            self.inputs, self.outputs, self.input_type, self.input_shape = self.cvs_reader(csv_df, self.model_path)


        # some tf model config contain ":0", but real don't contain ":0"
        if self.model_path[-2:] == "pb":
            print("csv file incorrect. read csv get :0 name, and this model is tensorflow model.")
            self.inputs = [name[:-2] if name[-2:] == ":0" else name for name in self.inputs]
            self.outputs = [name[:-2] if name[-2:] == ":0" else name for name in self.outputs]
            self.input_shape = {(key[:-2] if key[-2:] == ":0" else key) : val for key, val in self.input_shape.items()}
        return len(self.inputs) != 0

    def to_json(self):
        self.res["model"] = self.model_name
        self.res["framework"] = self.framework
        self.res["nnf_version"] = self.nnf_version
        self.res["model_format"] = self.model_format
        self.res["trans_format"] = self.trans_format
        self.res["inputs"] = ",".join(self.inputs)
        self.res["outputs"] = ",".join(self.outputs)
        self.res["input_shape"] = self.input_shape
        self.res["input_type"] = ",".join(self.input_type)
        self.res["layout"] = self.layout
        self.res["dataset_name"] = self.dataset_name
        self.res["best_batch"] = []
        self.res["model_precision"] = "FLOAT32"
        self.res["best_thread_num"] = 4

        table = set()
        # TODO: batch size for a scalar input is meaningless.
        for val in self.res["inputs"].split(","):
            batch = self.res["input_shape"][val][0] if self.res["input_shape"][val] else None
            batch = 8 if batch == -1 else batch
            if batch is not None:
                table.add(batch)

        if len(table) != 1:
            self.res["best_batch"] = -1
            print("model dim[0] not same, please user specified input dims")
        else:
            self.res["best_batch"] = list(table)[0]
            for key, val in self.res["input_shape"].items():
                self.res["input_shape"][key] = ["best_batch"] + val[1:]

        return self.res


class Tf(Process):
    def __init__(self, suffix, model_name, model_path):
        super().__init__(model_name, model_path)
        self.framework = "Tensorflow"
        self.nnf_version = "v1.15.0"
        self.model_format = suffix
        self.trans_format = "pb"
        self.layout = "NHWC"
        self.shape_reader = GetTf().get_from_path


class Onnx(Process):
    def __init__(self, suffix, model_name, model_path):
        super().__init__(model_name, model_path)
        self.framework = "Onnx"
        self.nnf_version = "v1.8.1"
        self.model_format = suffix
        self.trans_format = "onnx"
        self.layout = "NCHW"
        self.shape_reader = GetOnnx().get_from_path


class Torch(Process):
    def __init__(self, suffix, model_name, model_path):
        super().__init__(model_name, model_path)
        self.framework = "Pytorch"
        self.nnf_version = "v1.12.1"
        self.model_format = suffix
        self.trans_format = "onnx"
        self.layout = "NCHW"


class Relay(Onnx):
    def __init__(self, suffix, model_name, model_path):
        super().__init__(suffix, model_name, model_path)

    def get_info(self, csv_df=None):
        relay_path, params_path = self.model_path.split(",")[0], self.model_path.split(",")[1]
        file_name = relay_path.split("/")[-1].split(".")[0]
        output_temp = f"relay/{file_name}.onnx"
        relay2onnx(relay_path, params_path, output_temp)
        self.model_path = output_temp
        return super().get_info(csv_df)


class Keras(Tf):
    def __init__(self, suffix, model_name, model_path):
        super().__init__(suffix, model_name, model_path)
        self.trans_format = "onnx"
        self.shape_reader = None


def get_process(model_name, model_path):
    nodes = model_path.split(",")[0]
    suffix = nodes.split("/")[-1].split(".")[-1]
    print(suffix, nodes)
    if os.path.isdir(model_path):
        onnx_file = glob(os.path.join(model_path, "*.onnx"))
        # onnx with data saved external.
        if onnx:
            return Onnx(suffix, model_name, onnx_file[0])
        # tf savedmodel.
        return Tf(suffix, model_name, model_path)

    framedict = {'onnx': Onnx, 'pb': Tf, 'relay': Relay, 'h5': Keras, 'pt': Torch}
    frame = framedict.get(suffix.lower())
    return None if not frame else frame(suffix, model_name, model_path)


if __name__ == "__main__":
    import argparse

    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="modelzoo_config.csv",
        help="https://streamcomputing.feishu.cn/sheets/shtcnX9jxQGweLKcbAF4n5IiU7d?sheet=Kx0iKt  format",
    )
    parser.add_argument("--suffix", default="tmp", help="output file suffix")
    parser.add_argument("--model_name", default=None, help="generate model_name")
    parser.add_argument("--model_path", default=None, help="miniO remote model_path")

    args = parser.parse_args()

    os.makedirs(f"model_zoo-{args.suffix}", exist_ok=True)
    os.makedirs(f"workloads-{args.suffix}", exist_ok=True)

    def cal_oneline(model_name, model_path, csv_df=None):
        print(model_name)
        if not os.path.splitext(model_path)[1]:
            # if os.path.isdir(model_path):
            get_files(model_path, model_path)
        if len(model_path.split(",")) == 2:
            get_file(model_path.split(",")[0], model_path.split(",")[0])
            get_file(model_path.split(",")[1], model_path.split(",")[1])
        else:
            get_file(model_path, model_path)

        workload = NormalWorks(model_name)
        process_node = get_process(model_name, model_path)

        if not process_node.get_info(csv_df):
            print(f"ERROR model: {cell['remote_model_path']}, get input failed")
            return
        model_dict = process_node.to_json()
        work_dict = workload.to_json()

        with open("model_zoo-{}/{}.json".format(args.suffix, model_name), "w") as f:
            json.dump(model_dict, f, indent=4)
        with open("workloads-{}/{}.json".format(args.suffix, model_name), "w") as f:
            json.dump(work_dict, f, indent=4)

    if args.model_name and args.model_path:
        cal_oneline(args.model_name, args.model_path)
    else:
        csv_df = pd.read_csv(args.csv)
        for i in range(len(csv_df)):
            cell = csv_df.iloc[i]

            model_name = cell["model"]
            model_path = cell["remote_model_path"]

            cal_oneline(model_name, model_path, csv_df)

