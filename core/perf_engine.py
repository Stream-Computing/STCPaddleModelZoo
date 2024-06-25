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

"""
import argparse
import logging
import os
import importlib
import json

import subprocess
import time
import sys
from typing import Any, Dict
import pandas as pd

from toolutils.common_utils import check_remote, modelzoo_json_check, workload_json_check
from toolutils.common_utils import MyThread

from core.dispatch import load_workload, load_dataset, load_engine, get_accuracy_checker
from toolutils.build_pdf import build_pdf
from toolutils.venv_utils import activate_venv, deactivate_venv
from toolutils.stc_model_utils import model_convert
from toolutils.common_utils import Reporter, refix



log = logging.getLogger("Hs_mlperf")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def get_args():
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=[], help="The task going to be evaluted, refs to workloads/")
    parser.add_argument("--hardware_type", default="STC", help="The backend going to be evaluted, refs to engines/")
    parser.add_argument("--compile_tools", default="thread_ddk", help="The compile tools names : thread_ddk or mltc")
    parser.add_argument("--compile_only", action="store_true", help="Run compilation only")
    parser.add_argument("--use_cache", default=None, help="use model cache or cpu_res cache. eg: model cpu all")
    parser.add_argument("--iterations", type=int, help="Iterations we need to do Inference")
    parser.add_argument("--venv_path", default=None, help="Iterations we need to do Inference")
    parser.add_argument("--batch_sizes", type=int, nargs="+", help="Batch sizes we will test in performace mode")
    parser.add_argument("--config_json", default=None, help="Iterations we need to do Inference")
    parser.add_argument(
        "--data_percent",
        type=int,
        help="Data percent we will used in the whole data set when we will test in accuracy mode",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset_name we will used as input data set to do Inference. Default in model_zoo/",
    )
    parser.add_argument("--ip", default="", help="IP we will used in Remote model")
    parser.add_argument("--user_name", default="", help="user_name we will used as ssh user_name in Remote model")
    parser.add_argument("--password", default="", help="password we will used as ssh password in Remote model")
    parser.add_argument("--force_remote", action="store_true", help="force model use remote inference, ignore workloads set")
    parser.add_argument("--log_level", default=os.environ.get("STFS_LOG_LEVEL", "info"), help="The log level(eg. info)")
    parser.add_argument("--ci_check", action="store_true", help="check numeric accuracy, greater than 0.99")
    parser.add_argument("--mltc_compile_args", default="", help="add mltc compile args")

    args = parser.parse_args()

    if args.log_level == "info" or args.log_level == "INFO":
        log.setLevel(logging.INFO)
    elif args.log_level == "debug" or args.log_level == "DEBUG":
        log.setLevel(logging.DEBUG)
    elif args.log_level == "error" or args.log_level == "ERROR":
        log.setLevel(logging.ERROR)
    elif args.log_level == "warning" or args.log_level == "WARNING":
        log.setLevel(logging.WARNING)
    elif args.log_level == "critical" or args.log_level == "CRITICAL":
        log.setLevel(logging.CRITICAL)

    return args


class PerfEngine(object):
    def __init__(self) -> None:
        super().__init__()
        self.args = get_args()
        self.npu_version = "npu-v2"
        if self.args.compile_tools != "mltc" or "npu-v1" in self.args.mltc_compile_args:
            self.npu_version = "npu-v1"
        self.args.mltc_compile_args = self.args.mltc_compile_args.replace(f"-arch={self.npu_version}","")
        self.workloads = load_workload(self.args.tasks, "workloads" if self.npu_version != "npu-v2" else "workloads_v2")
        self.engine_type = self.args.hardware_type
        self.old_os_path = os.environ["PATH"]
        self.prev_sys_path = list(sys.path)
        self.real_prefix = sys.prefix

    def start_engine(self) -> None:
        """
        Byte MlPerf will create an virtual env for each backend to avoid dependance conflict
        """
        success, total = 0, len(self.workloads)
        if total == 0:
            return

        self.backend = load_engine(self.engine_type)

        for workload in self.workloads:
            t = MyThread(self.workload_perf, (workload,))
            t.start()
            t.join()
            print("thread already done ........ ")
            status, workload_report = t.get_res()

            if self.args.ci_check and status !="success":
                assert False
            if status == "success":
                success += 1
            if not workload_report or not isinstance(workload_report, dict):
                continue
            workload_report.update({"Time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())})
            self.save_report(workload_report)

        model_converge = (success / total) * 100 if total > 0 else 0

        results = {
            "Time_end": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "Backend": self.engine_type,
            "Model Coverage": model_converge,
        }


        output_dir = os.path.abspath("reports/" + self.engine_type)
        os.makedirs(output_dir, exist_ok=True)

        file = []
        if os.path.exists("reports/" + self.engine_type + "/Task_Status.json"):
            with open("reports/" + self.engine_type + "/Task_Status.json", "r") as f:
                file = json.load(f)

        file.append(results)

        with open("reports/" + self.engine_type + "/Task_Status.json", "w") as f:
            json.dump(file, f, indent=4)


    def workload_perf(self, workload: Dict[str, Any]):

        if self.args.force_remote:
            workload["remote"] = True

        log.info("Start to test model: {}.".format(workload["model"]))
        
        # updata workloads by shell input
        if self.args.config_json:
            if self.args.config_json[-5:] == ".json":
                with open(self.args.config_json, "r") as f:
                    self.args.config_json = json.load(f)
            else:
                self.args.config_json = json.loads(self.args.config_json)

        if self.args.config_json:
            for key, val in self.args.config_json.items():
                if key in workload:
                    workload[key] = val

        workload_json_check(workload)
        model_info = self.get_model_info(workload["model"], "model_zoo" if self.npu_version != "npu-v2" else "model_zoo_v2")
        
        if self.args.data_percent and isinstance(self.args.data_percent, int):
            workload['data_percent'] = self.args.data_percent

        # updata mdoelzoo by shell input
        if self.args.config_json:
            for key, val in self.args.config_json.items():
                if key in model_info:
                    model_info[key] = val

        modelzoo_json_check(model_info)

        if self.args.dataset_name:
            model_info["dataset_name"] = self.args.dataset_name
            workload["test_numeric"] = model_info["dataset_name"]=="fake_dataset" or workload["test_numeric"]

        reporter = Reporter(model_info, workload, self.engine_type, model_info["best_thread_num"], workload["remote"])

        model_convert(model_info)
        self.backend.pre_optimize(model_info)
        start = time.time()
        # 需要考虑batch不同问题，会造成保存的模型和应用的模型batch不一致。
        if self.args.compile_tools == "mltc":
            stcobj_chache_path = self.backend.tmpdir + self.backend.model_name + ".stcobj"
        else:
            stcobj_chache_path = self.backend.tmpdir + self.backend.model_name
        duration_bak = None
        if self.args.use_cache in ["model", "all"] and os.path.exists(stcobj_chache_path):
            log.info("get lazy_compile. file is : {}".format(stcobj_chache_path))
            self.backend.update_compile_data(model_info, self.npu_version)
            compile_info = {}
            compile_info["compile_status"] = "success"
            duration_bak = self.get_compile_time(model_info["model"])
        else:
            _, compile_info = self.backend.compile({"workload": workload, "model_info": model_info, 
                              "compile_tools": self.args.compile_tools,
                              "compile_args": self.args.mltc_compile_args,
                              "npu_version": self.npu_version})

        reporter.set_compile_time(time.time() - start if not duration_bak else duration_bak)

        # compile only mode will stop here
        if self.args.compile_only or workload["compile_only"]:
            return compile_info["compile_status"], reporter.base_report

        # if all model parallel compile can let venv confused
        if not self.args.ci_check:
            activate_venv("common", model_info, self.args.venv_path)

        # init dataset
        model_info['data_percent'] = workload['data_percent']
        ds = load_dataset(model_info)

        # compile only mode will stop here
        if self.args.compile_only or workload["compile_only"]:
            return compile_info["compile_status"], reporter.base_report

        # check remote functions
        if workload["remote"]:
            check_remote(self.args)
            self.backend.switch_to_remote(self.args.ip, self.args.user_name, self.args.password)
        else:
            self.backend.switch_to_local(self.args.compile_tools)
        self.backend.load_model(model_info["best_thread_num"])

        # test accuracy
        AccuracyChecker = get_accuracy_checker(model_info["dataset_name"])
        AccuracyChecker.update(self.backend, model_info)
        AccuracyChecker.set_dataloader(ds)

        if workload["test_accuracy"] or workload["test_numeric"]:
            log.info("Running Accuracy Checker...")

            if "accuracy_batch" in model_info:
                best_batch = model_info["accuracy_batch"]
            else:
                best_batch = self.backend.get_loaded_batch_size()

            ds.rebatch(best_batch)
            accuracy_results = AccuracyChecker.calculate_acc(workload["data_percent"])
            reporter.set_accuracy_res(accuracy_results)

        # test numeric
        if workload["test_numeric"] and self.engine_type.upper() != "CPU":
            log.info("Running Numeric Checker...")
            # check 相同名称 ，路径相同，名称相同，batch数量相同
            # 还需要专门人员使用，一般不支持该功能
            golden_filepath = AccuracyChecker.get_cpu_data_filepath()

            if not os.path.exists(golden_filepath) or self.args.use_cache not in ["cpu", "all"]:
                ds.rebatch(self.backend.get_loaded_batch_size())
                AccuracyChecker.updata_engine(load_engine("CPU"))
                cpu_accuracy = AccuracyChecker.calculate_acc(workload["data_percent"])
                print("-----------------CPU Accuracy--------------------------")
                print(cpu_accuracy)

            AccuracyChecker.updata_engine(self.backend)
            diff_results = AccuracyChecker.calculate_diff()
            reporter.set_diff_res(diff_results)


        # function to test qps and latency
        if workload["test_perf"] and self.engine_type.upper() != "CPU":
            log.info("Runing QPS Checker...")
            for batch_mode in ["batch_random", "batch_fix", "batch_total", "batch_normal"]:
                if self.args.compile_tools == "mltc":
                    continue
                if batch_mode == "batch_normal":
                    ds.rebatch(self.backend.get_loaded_batch_size())
                    fix_data = True
                else:
                    fix_data = workload.get(batch_mode, 0)
                    if not fix_data:
                        continue
                    ds.benchmark_resample(self.backend.get_loaded_batch_size(), batch_mode, fix_data)
                print("batch_mode : ", batch_mode)
                performance_reports = self.backend.benchmark(ds, workload["data_percent"], batch_mode == "batch_normal", npu_version=self.npu_version)
                reporter.set_perf_res(performance_reports, batch_mode, fix_data or True)


        self.backend.unload_model()

        print(reporter.base_report)

        with open(AccuracyChecker.get_output_dir() + "/result.json", "w") as f:
            json.dump(reporter.base_report, f, indent=4)

        build_pdf(AccuracyChecker.get_output_dir() + "/")
        log.info("Testing Finish. Report save to : [ {}/]".format(AccuracyChecker.get_output_dir()))
        
        if self.args.ci_check and "Accuracy" in reporter.base_report and "Numeric" in reporter.base_report["Accuracy"]:
            for out_names, cell in reporter.base_report['Accuracy']['Numeric'].items():
                if "cos_sim" in cell:
                    assert cell["cos_sim"] > 0.99, f"RESULT ACCURACY ERROR. [{out_names} : cos_sim {cell['cos_sim']} < 0.99]"
                if "Nan_percent" in cell:
                    assert cell["Nan_percent"] == 0, f"RESULT ACCURACY ERROR. [{out_names} : has nan output]"
                if "Inf_percent" in cell:
                    assert cell["Inf_percent"] == 0, f"RESULT ACCURACY ERROR. [{out_names} : has inf output]"

        return compile_info["compile_status"], reporter.base_report

    def get_model_info(self, model_name: str, zoo_path="model_zoo") -> Dict[str, Any]:
        with open(zoo_path +  "/" + model_name + ".json", "r") as f:
            model_info = json.load(f)
        return model_info

    def get_compile_time(self, Model_name):
        threshold_time = 5 # s
        if not os.path.exists("reports/" + self.engine_type + "/Overall.json"):
            return None

        with open("reports/" + self.engine_type + "/Overall.json", "r") as f:
            file = json.load(f)

        for node in reversed(file):
            if node["Model"].upper() == Model_name.upper() and node["Graph Compile"]["Compile Duration"] > threshold_time:
                return node["Graph Compile"]["Compile Duration"]
        
        return None

    def save_report(self, results):
        output_dir = os.path.abspath("reports/" + self.engine_type)
        os.makedirs(output_dir, exist_ok=True)

        file = []
        if os.path.exists("reports/" + self.engine_type + "/Overall.json"):
            with open("reports/" + self.engine_type + "/Overall.json", "r") as f:
                file = json.load(f)

        file.append(results)

        with open("reports/" + self.engine_type + "/Overall.json", "w") as f:
            json.dump(file, f, indent=4)

        # load resfile to csv
        base_file = pd.DataFrame()
        if os.path.exists("reports/" + self.engine_type + "/Overall.csv"):
            try:
                base_file = pd.read_csv("reports/" + self.engine_type + "/Overall.csv")
            except:
                log.info("Overall.csv format not correct. Create a new one.")
        for node in refix(results):
            base_file = base_file.append(pd.DataFrame().from_dict(node,orient="index").T)

        base_file.to_csv("reports/" + self.engine_type + "/Overall.csv", index=False)


if __name__ == "__main__":
    engine = PerfEngine()
    engine.start_engine()
