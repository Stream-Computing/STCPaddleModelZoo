import os
import json
from toolutils.common_utils import *
from toolutils.stc_model_utils import model_convert
from engines.STC.engine_stc import EngineSTC
import argparse
from tabulate import tabulate
import time, signal


def get_config(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def check_config(model_name):
    model_config_path = "model_zoo_paddle/{}.json".format(model_name)
    if not os.path.exists(model_config_path):
        print("model config {} not found !".format(model_config_path))
        exit(1)
    return model_config_path


def get_compile_time(case_name):
    compile_time_csv = os.path.join("./tests/daily", "./model_compile_time.csv")
    perf_dct = {}
    fr1 = open(compile_time_csv, "r")
    for line in fr1:
        model_name, compile_time, _ = line.strip().split(",")
        if model_name != 'model_name':
            perf_dct[model_name] = float(compile_time) if compile_time != "0.0" else 300
    fr1.close()
    return int(perf_dct.get(case_name, 300))


def set_timeout(callback):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                case_name = args[0]["model"]
                print(case_name)
                num = get_compile_time(case_name) * 3
                signal.alarm(num)
                print("start alarm signal.")
                r = func(*args, **kwargs)
                print("close alarm signal.")
                signal.alarm(0)
                return r
            except RuntimeError as e:
                callback()

        return to_do
    return wrap


def after_timeout():
    pid = os.environ['PID']
    if pid: 
        os.kill(int(pid),  signal.SIGKILL)
    print("engine_compile Time out!")

@set_timeout(after_timeout)
def engine_compile(model_info, compile_out_path):
    model_convert(model_info)
    t0 = time.time()
    os.environ['PID'] = '0'
    engine = EngineSTC(compile_out_path)
    engine.model_name = model_info["model"]
    engine.compile({"model_info": model_info})
    dt0 = time.time() - t0
    print(tabulate([{"compile_time": dt0}], headers="keys", tablefmt="fancy_grid"))
    print("************ engine build module {}:{} s ************".format(model_info["model"], dt0))

def check_result(model_name, compile_out_path):
    path = os.getcwd()
    if compile_out_path:
        filepath = os.path.join(compile_out_path,model_name)
    else:
        filepath = os.path.join(path,"engines/STC/mix_tmp",model_name)
    if os.path.exists(filepath):
        Files = os.listdir(filepath)
        for k in range(len(Files)):
            Files[k] = os.path.splitext(Files[k])[1]

        str_obj = ".stcobj"
        if str_obj in Files:
            print("************  aic_compile_success  ************")
        else:
            print("************  aic_compile_failed  ************")
            exit(1)
    else:
        print("************  aic_compile_failed  ************")
        exit(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        default="resnet50",
        required=True,
        help="Please input the model name",
    )
    parser.add_argument(
        "-o",
        "--compile_out_path",
        default="./engines/STC/mix_tmp/",
        help="[optional] to spec an extern compile_out_path",
    )
    args = parser.parse_args()
    model_name = args.model_name
    compile_out_path = args.compile_out_path
    print("************  start compile  ************")
    model_config_path = check_config(model_name)
    model_info = get_config(model_config_path)
    modelzoo_json_check(model_info)

    engine_compile(model_info, compile_out_path)
    check_result(model_name, compile_out_path)
    print("************  end compile  ************")
