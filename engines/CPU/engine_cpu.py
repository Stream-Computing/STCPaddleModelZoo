import os
import json
import logging

import tensorflow as tf

tf.get_logger().setLevel("ERROR")
import torch
from tensorflow.keras.models import load_model
import onnxruntime
import paddle.inference as paddle_infer

import time
import numpy as np
from toolutils.common_utils import isTf, isPytorch, isPb, isKeras, isPaddle
from pathlib import Path

from engines import engine

log = logging.getLogger("Hs_mlperf")

pt_dtype_map = {
    "UINT8": torch.uint8,
    "INT8": torch.int8,
    "INT16": torch.int16,
    "INT32": torch.int32,
    "INT64": torch.int64,
    "FLOAT16": torch.float16,
    "FLOAT32": torch.float32,
    "FLOAT64": torch.float64,
    "SHORT": torch.int16,
    "INT": torch.int32,
    "LONG": torch.int64,
    "HALF": torch.float16,
    "FLOAT": torch.float32,
    "DOUBLE": torch.float64,
    "BOOL": torch.bool,
}

INPUT_TYPE = {
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "FLOAT64": np.float64,
    "SHORT": np.int16,
    "INT": np.int32,
    "LONG": np.int64,
    "HALF": np.float16,
    "FLOAT": np.float32,
    "DOUBLE": np.float64,
    "BOOL": np.bool_,
}


class EngineCPU(engine.Engine):
    def __init__(self):
        super(EngineCPU, self).__init__()
        self.hardware_type = "CPU"
        self.need_reload = False
        self.model_runtimes = []

    def compile(self, config, dataloader=None):
        self.configs = config["model_info"]
        self.workload = config["workload"]
        self.model_info = config["model_info"]
        self.configs["compile_status"] = "success"
        self.best_batch = self.configs["best_batch"]

        return True, self.configs

    def update_compile_data(self, info):
        self.best_batch = info["best_batch"]

    def get_interact_profile(self, config):
        model_profile = []
        file_path = "engines/CPU/" + self.hardware_type + ".json"
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                model_profile = json.load(f)
        else:
            log.info("File path: {} does not exist, please check".format(file_path))

        return model_profile

    def predict(self, feeds):
        if not self.model_runtimes:
            self.load_model()

        results = {}
        if isTf(self.framework) and isPb(self.model_format):

            def get_real_feeds(input):
                res = {}
                for key, val in input.items():
                    res[key + ":0"] = val
                return res

            def get_real_outputs(outputs):
                res = []
                for val in outputs:
                    res.append(val + ":0")
                return res

            def model_infer(model, feed, outputs):
                with model.as_default():
                    with tf.compat.v1.Session(graph=model) as sess:
                        sess.inter_op_num_threads = 32
                        sess.intra_op_num_threads = 32
                        res = sess.run(outputs, feed_dict=feed)  # 运行一次模型
                return res

            real_feeds = get_real_feeds(feeds)
            real_outputs = get_real_outputs(self.outputs)
            for model_runtime in self.model_runtimes:
                _results = model_infer(model_runtime, real_feeds, real_outputs)

            for key, val in zip(self.outputs, _results):
                results[key] = val

            assert len(results) != 0

        elif isTf(self.framework) and isKeras(self.model_format):

            input_tensors = []
            for key, val in feeds.items():
                input_tensors.append(val)

            for model_runtime in self.model_runtimes:
                res = model_runtime.predict(*input_tensors)
            if isinstance(res, dict):
                results = res
            elif isinstance(res, tuple):
                for i, key in enumerate(self.outputs):
                    results[key] = list(res)[i]
            else:
                results = {self.outputs[0]: res}

        elif isTf(self.framework):
            entry_rt = self.model_runtimes[0].signatures["serving_default"]
            all_sn_inputs = entry_rt.structured_input_signature

            def get_real_feeds(feeds, sn_inputs):
                sn_inputs = tf.nest.flatten(sn_inputs, True)
                real_feeds = {}
                itr = 0
                for _, val in feeds.items():
                    real_feeds[sn_inputs[itr].name] = tf.constant(val)
                    itr += 1
                return real_feeds

            real_feeds = get_real_feeds(feeds, all_sn_inputs)

            for model_runtime in self.model_runtimes:
                with tf.device("/CPU:0"):
                    _results = model_runtime.signatures["serving_default"](**real_feeds)

            for key, val in _results.items():
                results[key] = val.numpy()

            assert len(results) != 0

        elif isPytorch(self.framework):
            input_tensors = []
            i = 0
            for key, _ in feeds.items():
                input_tensors.append(torch.tensor(feeds[key], dtype=pt_dtype_map[self.input_type[i]]).to(self.device))
                i += 1
            with torch.no_grad():
                for model_runtime in self.model_runtimes:
                    res = model_runtime(*input_tensors)

            if isinstance(res, dict):
                for key, val in res.items():
                    results[key] = val.cpu().detach().numpy()
            elif isinstance(res, tuple):
                for i, key in enumerate(self.outputs):
                    results[key] = list(res)[i]
            else:
                results = {self.outputs[0]: res.cpu().numpy()}

        elif isPaddle(self.framework):
            predictor = self.model_runtimes[0]

            input_names = predictor.get_input_names()
            # print("input_names:",input_names)
            for i, name in enumerate(input_names):
                input_tensor = predictor.get_input_handle(name)
                input_tensor.reshape(feeds[name].shape)
                input_tensor.copy_from_cpu(feeds[name].copy())

            predictor.run()

            output_names = predictor.get_output_names()
            for i, name in enumerate(output_names):
                output_tensor = predictor.get_output_handle(name)
                output_data = output_tensor.copy_to_cpu()
                results[name] = output_data
            return results

        else:
            for model_runtime in self.model_runtimes:
                output_names = []
                for node in model_runtime.get_outputs():
                    output_names.append(node.name)

                res = model_runtime.run(output_names, feeds)

                for key, val in zip(output_names, res):
                    results[key] = val
        return results

    def benchmark(self, dataloader, percent=100):
        batch_sizes = self.workload["batch_sizes"]
        iterations = self.workload["iterations"]
        reports = []
        for batch_size in batch_sizes:
            times_range = []
            report = {}
            report["BS"] = batch_size
            test_data = self._get_fake_samples(
                batch_size,
                self.configs["input_shape"],
                self.configs["input_type"].split(","),
            )

            for _ in range(30):
                self.predict(test_data)

            for _ in range(iterations):
                start_time = time.time()
                self.predict(test_data)
                end_time = time.time()
                times_range.append(end_time - start_time)

            times_range.sort()
            tail_latency = round(times_range[int(len(times_range) * 0.99)] * 1000, 2)
            avg_latency = round(sum(times_range) / iterations * 1000, 2)
            qps = int(1000.0 * batch_size / avg_latency)

            log.info(
                "Batch size is {}, QPS: {}, Avg Latency:{}, Tail Latency:{}".format(
                    batch_size, qps, avg_latency, tail_latency
                )
            )

            report["QPS"] = qps
            report["AVG Latency"] = avg_latency
            report["P99 Latency"] = tail_latency
            reports.append(report)

        return reports

    def get_loaded_batch_size(self):
        return self.best_batch

    def load_model(self, thread_num=None):
        self.input_type = self.configs["input_type"].split(",")
        self.framework = self.configs["framework"]

        self.model_name = self.configs["model"]
        self.model_format = self.configs["model_format"]
        self.input_shapes = self.configs["input_shape"]
        self.outputs = self.configs["outputs"].split(",")
        if isTf(self.framework) and isPb(self.model_format):
            model = tf.Graph()
            with model.as_default():
                od_graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(self.configs["model_path"], "rb") as fid:  # 加载模型
                    od_graph_def.ParseFromString(fid.read())
                    tf.import_graph_def(od_graph_def, name="")
        elif isTf(self.framework) and isKeras(self.model_format):
            model = load_model(self.configs["model_path"])
        elif isTf(self.framework):
            with tf.device("/CPU:0"):
                model = tf.saved_model.load(self.configs["model_path"])
        elif isPytorch(self.framework):
            self.device = "cpu"
            model = torch.jit.load(self.configs["model_path"], torch.device("cpu"))
            model.eval()
        elif isPaddle(self.framework):
            self.device = "cpu"
            base_name = self.configs["model_path"]
            pd_model, pd_iparams = None, None
            paddle_dir = Path(base_name)
            for file in paddle_dir.iterdir():
                if file.suffix == ".pdmodel":
                    pd_model = str(file)
                elif file.suffix == ".pdiparams":
                    pd_iparams = str(file)

            config = paddle_infer.Config(pd_model, pd_iparams)
            config.disable_gpu()
            model = paddle_infer.create_predictor(config)
        else:
            sess_options = onnxruntime.SessionOptions()
            sess_options.log_severity_level = 4
            sess_options.intra_op_num_threads = 8
            model = onnxruntime.InferenceSession(
                self.configs["model_path"], providers=["CPUExecutionProvider"], sess_options=sess_options
            )

        self.model_runtimes.append(model)

    def unload_model(self):
        if self.model_runtimes:
            for cons in self.model_runtimes:
                del cons

    def _get_fake_samples(self, batch_size, shape, input_type):
        data = {}
        if input_type:
            i = 0
            for key, val in shape.items():
                val = [val[0] * batch_size] + val[1:]
                data[key] = np.random.random(size=val).astype(INPUT_TYPE[input_type[i]])
                i += 1
            return data
        else:
            raise ValueError("Please provide input type")
