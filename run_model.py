'''
功能：该脚本主要功能是运行模型推理
参数：
model_name：模型名字
thread_num：单次请求 stream
input_batchsize：模型的batch
npu_id：调用的npu 卡号
request_num：请求次数
out_dir：日志输出目录
custom：不同的客户项目
'''
import os
import tb
import json
from engines.STC.engine_stc import EngineSTC
from core.dispatch import load_workload, load_dataset, load_engine, get_accuracy_checker
from toolutils.stc_model_utils import model_convert
from toolutils.common_utils import *
from tb.utils.utils import is_valid, stcprof_list_to_dict
import argparse
import time, signal
import numpy as np


def get_config(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None


def check_config(model_name):
    model_config_path = "model_zoo/{}.json".format(model_name)
    workloads_config_path = "workloads_paddle/{}.json".format(model_name)
    if not os.path.exists(model_config_path) or not os.path.exists(model_config_path):
        print("model config not found !")
        exit(1)
    return model_config_path, workloads_config_path

def get_infer_time(case_name):
    compile_time_csv = os.path.join("./tests/daily", "./model_infer_time.csv")
    perf_dct = {}
    fr1 = open(compile_time_csv, "r")
    for line in fr1:
        model_name, infer_times = line.strip().split(",")
        if model_name != 'case_name':
            perf_dct[model_name] = float(infer_times) if infer_times != "0.0" else 600
    fr1.close()
    return int(perf_dct.get(case_name, 600))


def set_timeout(callback):
    def wrap(func):
        def handle(signum, frame):
            raise RuntimeError

        def to_do(*args, **kwargs):
            try:
                signal.signal(signal.SIGALRM, handle)
                case_name = args[1]["model"]
                num = get_infer_time(case_name) * 3
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


def print_result(results):
    print('\n******************************* Test Info ****************************************')
    print("model_name:          {}".format(results['Model']))
    print("dataset:             {}".format(model_info['dataset_name']))
    print('batch_size:          {}'.format(results['input_batchsize']))
    print('thread_num(stream):  {}'.format(results['Thread_num']))
    print('npu_num:             {}'.format(results['npu_num']))
    print('cluster_num:         {}'.format(results['cluster_num']))
    print('sample_num:          {}'.format(results['samples']))
    print('batch_count:         {}'.format(results['batch_count']))

    print('\n******************************* Time Info ********************************************')
    print("start_time_all:      {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results['total_start']))))
    print("end_time_all:        {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results['total_end']))))
    print("start_time_infer:    {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results['infer_start']))))
    print("end_time_infer:      {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results['infer_end']))))
    print("all_time:            {}".format(results['total_dt']))
    print('pure inference time: {}'.format(results['infer_dt']))
    print('data load time:      {}'.format(results['data_prepare_dt']))
    print('model load time:     {}'.format(results['model_load_dt']))
    print("misc time:           {}".format(results['misc_dt']))
        
    print('\n********************* performance by pure inference time ********************************')
    print('pure samples/sec:    {}'.format(results['samples/sec']))
    print('pure latency:        {}'.format(results['avg_latency']))

    print('\n********************* performance by total time  ********************************')
    print('samples/sec:         {}'.format(results['samples'] / results['total_dt'])) # 每秒跑多少 sample
    print('latency:             {}'.format(results['total_dt'] / results['batch_count'])) # 每个batch跑多少s
    
    
def tran_perf_result(perf_recs):
    recs = stcprof_recs_handler([perf_recs])
    print("Perf mean: {}".format(stcprof_list_to_dict(recs[0])))
    return recs


def stcprof_recs_handler(perf_recs):
    recs_array = np.array(perf_recs)
    recs_mean = np.mean(recs_array, axis=0)
    recs_std = np.std(recs_array, axis=0)
    recs_conf_inter = np.transpose(np.array([recs_mean - 1.96 * recs_std, recs_mean + 1.96 * recs_std]))
    return recs_mean.tolist(), recs_conf_inter.tolist()   


def prepare_dataset(model_info, input_batchsize, request_num=None):
    print("************ start load dataset ************")
    t0 = time.time()
    ds = load_dataset(model_info)
    ds.rebatch(input_batchsize)
    samples_count = ds.get_total_batch(1) * ds.get_batch_count()
    if request_num is not None:
        batch_count = request_num
    else:
        batch_count = ds.get_batch_count()
    batch_size = ds.get_total_batch(1)
    samples_count = batch_size * batch_count
    dt0 = time.time() - t0
    print("************ end load dataset ************")
    return ds, samples_count, batch_size, batch_count, dt0


def load_engine(model_info, thread_num, ds, out_dir, compile_dir):
    print("************ start load engine ************")
    t1 = time.time()
    engine = EngineSTC(compile_dir)
    engine.model_name = model_info["model"]
    engine.switch_to_local()
    engine.update_compile_data(model_info)
    res = engine.load_model(thread_num)
    if not res:
        print("load model failed")
        exit(1)
    AccuracyChecker = get_accuracy_checker(model_info["dataset_name"])
    AccuracyChecker.update(engine, model_info)
    AccuracyChecker.set_dataloader(ds)
    AccuracyChecker.output_dir = out_dir
    dt1 = time.time() - t1
    print("************ end load engine ************")
    print("************ load engine cost {} s ************".format(dt1))
    return AccuracyChecker, dt1,engine


def run_inference(AccuracyChecker,data_percent=100, request_num=None):
    print("************ start run inference ************")
    t2 = time.time()
    print("request_num:", request_num)
    print("data_percent:", data_percent)
    accuracy_results = {}
    if request_num is not None:
        accuracy_results.update(AccuracyChecker.calculate_acc(data_percent=data_percent, request_num=request_num))
    else:
        accuracy_results.update(AccuracyChecker.calculate_acc(data_percent=data_percent))
    dt2 = time.time() - t2
    print("************ end run inference ************")
    print("************ run inference cost {} s ************".format(dt2))
    return accuracy_results, dt2


def load_engine_cycle(model_info, thread_num, ds, out_dir, compile_dir):
    print("************ start load cycle engine ************")
    engine_start_time = time.time()
    engine = EngineSTC(compile_dir)
    engine.model_name = model_info["model"]
    engine.switch_to_local(version="local_ddk")
    engine.update_compile_data(model_info)
    res = engine.load_model(thread_num)
    if not res:
        print("load model failed")
        exit(1)
    AccuracyChecker = get_accuracy_checker(model_info["dataset_name"])
    AccuracyChecker.update(engine, model_info)
    AccuracyChecker.set_dataloader(ds)
    AccuracyChecker.output_dir = out_dir
    cycle_engine_time = time.time() - engine_start_time
    print("************ end load cycle engine ************")
    print("************ load cycle engine cost {} s ************".format(cycle_engine_time))
    return AccuracyChecker, engine


def run_cycle(AccuracyChecker_cycle, data_percent=100):
    print("************ start run cycle ************")
    t3 = time.time()
    print("request_num:", request_num)
    print("data_percent:", data_percent)
    
    cycle_results = {}
    
    tb.runtime.hal.stcProfStart()
    cycle_results.update(AccuracyChecker_cycle.calculate_acc(data_percent=data_percent))
    res_cycle = tb.runtime.hal.stcProfStop()
    tb.runtime.hal.stcProfClear(os.getpid())
    
    dt3 = time.time() - t3
    print("************ end run cycle ************")
    print("************ run cycle cost {} s ************".format(dt3))
    recs_perf = tran_perf_result([int(item) for item in res_cycle])
    print("total_cycles:",recs_perf[0][0])
    results["total_cycles"] = recs_perf[0][0]
    return recs_perf[0][0]


def test_numeric(AccuracyChecker):
    print("Running Numeric Checker...")
    diff_results = AccuracyChecker.calculate_diff()
    print("diff_results",diff_results)
    return diff_results

# @set_timeout(after_timeout)
def test_perf(backend, workload):
    perf_result = []
    print("Runing QPS Checker...")
    for batch_mode in ["batch_random", "batch_fix", "batch_total", "batch_normal"]:
        if batch_mode == "batch_normal":
            ds.rebatch(backend.get_loaded_batch_size())
            fix_data = True
        else:
            fix_data = workload.get(batch_mode, 0)
            if not fix_data:
                continue
            ds.benchmark_resample(backend.get_loaded_batch_size(), batch_mode, fix_data)
        print("batch_mode: ", batch_mode)
        os.environ['PID'] = '0'
        performance_reports = backend.benchmark(ds, workload["data_percent"], batch_mode == "batch_normal")
        performance_reports.update({'batch_mode':[batch_mode,fix_data]})
        print("performance_reports: {}".format(performance_reports))
        perf_result.append(performance_reports)
    return perf_result


if __name__ == "__main__":
    start_time_all = time.time()

    parser = argparse.ArgumentParser(description = '...')
    parser.add_argument('--model_name', '-m', type=str, help = 'model_name', required=True)
    parser.add_argument('--thread_num', '-t', type=int, help = 'thread_num', default=-1)
    parser.add_argument('--input_batchsize', '-b', type=int, help = 'input_batchsize', default=-1)
    parser.add_argument('--npu_id', '-n', nargs='*', type=int, help = 'npu_num', default=None)
    parser.add_argument('--out_dir', '-d', type=str, help = 'out_dir', default=None)
    parser.add_argument('--compile_dir', '-i', type=str, help = 'compile_dir', default="./engines/STC/mix_tmp/")
    parser.add_argument('--request_num', '-r', type=int, help = 'request_num', default=None)
    parser.add_argument("--data_percent", "-p", type=int, help="Please input data_percent", default=-1)
    parser.add_argument("--custom", "-c", type=str, help="custom", default=None)
    args = parser.parse_args()

    model_name = args.model_name
    thread_num = args.thread_num
    input_batchsize = args.input_batchsize
    npu_id = args.npu_id
    out_dir = args.out_dir
    compile_dir = args.compile_dir
    request_num = args.request_num
    data_percent = args.data_percent
    custom = args.custom

    if out_dir is None:
        out_dir = './inference_log/{}'.format(model_name)
        out_dir = os.path.realpath(out_dir)
    else:
        out_dir = os.path.join(out_dir,'inference_log/{}'.format(model_name))
    os.makedirs(out_dir, exist_ok=True)

    # set npu
    cluster_list = []
    if npu_id is None:
        npu_num = 1
        npu_id = 0
        cluster_list += list(range( npu_id * 4, npu_id * 4 + 4))
    else:
        npu_num = npu_id
        for n_i in npu_id:
            cluster_list += list(range( n_i * 4, n_i * 4 + 4))
    cluster_list_s = [str(x) for x in cluster_list]
    os.environ['STC_SET_DEVICES'] = ','.join(cluster_list_s)
    print('STC_SET_DEVICES={}'.format(os.environ['STC_SET_DEVICES']))

    #Initialization Parameter
    model_config_path, workloads_config_path = check_config(model_name)
    model_info = get_config(model_config_path)
    workload_info = get_config(workloads_config_path)
    modelzoo_json_check(model_info)
    model_convert(model_info)
    
    # limit load and run data_percent
    model_info["data_percent"] = workload_info["data_percent"]
    if data_percent != -1:
        model_info["data_percent"] = data_percent
        
    if thread_num == -1:
        thread_num = model_info['best_thread_num'] * npu_num
    if input_batchsize == -1:
        if "batch_fix" in workload_info:
            input_batchsize = workload_info["batch_fix"] * npu_num
        elif "accuracy_batch" in model_info:
            input_batchsize = model_info["accuracy_batch"] * npu_num
        else:
            input_batchsize = model_info["best_batch"] * npu_num
    
    ds, samples_count, batch_size, batch_count, dt0 = prepare_dataset(model_info, input_batchsize, request_num)

    # initialization model
    results = {'Model': model_name}
    results['Backend'] = "STC"
    results['Region'] = 'Local'
    results['Dataset'] = model_info['dataset_name']
    results['Thread_num'] = thread_num
    results['Output_name'] = model_info['outputs']
    print("model_info", model_info)
    AccuracyChecker, dt1, engine = load_engine(model_info, thread_num, ds, out_dir, compile_dir)
    if workload_info["test_accuracy"] or workload_info["test_numeric"]:
        accuracy_results, dt2 = run_inference(AccuracyChecker,model_info["data_percent"], request_num)
        accuracy_results.update({'Data Percent':model_info["data_percent"]})
        results.update({'Accuracy':accuracy_results})
    print("accuracy_results: ", accuracy_results)
    if custom == "paddle":
        if 'Top-1' in results.keys():
            results['Top-1_acc'] = accuracy_results['Top-1']
        if 'Top-5' in results.keys():
            results['Top-5_acc'] = accuracy_results['Top-5']
        if 'mAP50' in results.keys():
            results['mAP50'] = accuracy_results['mAP50']
        if 'accuracy' in results.keys():
            results['acc'] = accuracy_results['accuracy']
        if 'acc' in results.keys():
            results['acc'] = accuracy_results['acc']
        if 'precision' in results.keys():
            results['precision'] = accuracy_results['precision']
        if 'recall' in results.keys():
            results['recall'] = accuracy_results['recall']
        if 'hmean' in results.keys():
            results['hmean'] = accuracy_results['hmean']

    if workload_info["test_numeric"]:
        diff_results = test_numeric(AccuracyChecker)
        results.update({'Numeric':diff_results}) 
    
    if workload_info["test_perf"]:    
        perf_result = test_perf(engine,workload_info)
        results.update({'Performance':perf_result}) 
    engine.unload_model()

    if  results.get('Accuracy',{}).get('avg_latency',0) > 0:
        results['avg_latency'] = results['Accuracy']['avg_latency']
        results['samples'] = results['Accuracy']['samples']
        results['infer_start'] = results['Accuracy']['infer_start']
        results['infer_end'] = results['Accuracy']['infer_end']
        results['infer_dt'] = results['avg_latency'] * batch_count
        results['samples/sec'] = results['samples'] / results['infer_dt']
        results['misc_dt'] = dt2 - results['avg_latency'] * batch_count
    else:
        results['samples'] = samples_count
        results['samples/sec'] = -1
        results['infer_dt'] = -1
        results['misc_dt'] = -1
        results['avg_latency'] = -1
        results['infer_start'] = -1
        results['infer_end'] = -1
    
    results['data_prepare_dt'] = dt0
    results['model_load_dt'] = dt1
    results['inference_dt'] = dt2
    results['run_start_time'] = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time_all))
    
    end_time_all = time.time()
    all_time = end_time_all - start_time_all

    results['out_dir'] = out_dir
    results['total_start'] = start_time_all
    results['total_end'] = end_time_all
    results['total_dt'] = all_time
    results['batch_count']  = batch_count
    results['input_batchsize'] = input_batchsize
    
    results['npu_id'] = str(npu_id)
    results['npu_num'] = npu_num
    results['cluster_num'] = ','.join(cluster_list_s)
    
    # model_info["dataset_name"] = "fake_dataset"
    # ds, _, _, _, _ = prepare_dataset(model_info, input_batchsize, request_num)
    # AccuracyChecker_cycle, engine_cycle = load_engine_cycle(model_info, thread_num, ds, out_dir, compile_dir) 
    # run_cycle(AccuracyChecker_cycle,data_percent=model_info["data_percent"])
    # engine_cycle.unload_model()
    
    print_result(results)
    # tb.runtime.hal.stcUnRegisterObject()
    # with open(os.path.join(out_dir, model_name + ".json"), "w") as outfile:
    #     json.dump(results, outfile, sort_keys=False, indent=4, ensure_ascii=False)