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
import os,sys
file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(file_path)
sys.path.append(dir_path)
import tb
import json
from engines.STC.engine_stc import EngineSTC
from core.dispatch import load_workload, load_dataset, load_engine, get_accuracy_checker
from toolutils.stc_model_utils import model_convert
from toolutils.common_utils import *
import argparse
import time
from tabulate import tabulate


def get_config(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

def print_result(results, custom=None, debug = False):

    print('\n******************************* Test Info ****************************************')
    print("model_name:          {}".format(results['model_name']))
    print("dataset:             {}".format(model_info['dataset_name']))
    print('batch_size:          {}'.format(results['input_batchsize']))
    print('thread_num(stream):  {}'.format(results['thread_num']))
    print('npu_num:             {}'.format(results['npu_num']))
    print('cluster_num:         {}'.format(results['cluster_num']))
    print('sample_num:          {}'.format(results['samples']))
    print('batch_count:         {}'.format(results['batch_count']))

    print('\n******************************* Time Info ********************************************')
    print("start_time_all:      {}".format(time.strftime("%Y%m%d-%H-%M-%S", time.localtime(results['total_start']))))
    print("end_time_all:        {}".format(time.strftime("%Y%m%d-%H-%M-%S", time.localtime(results['total_end']))))
    print("start_time_infer:    {}".format(time.strftime("%Y%m%d-%H-%M-%S", time.localtime(results['infer_start']))))
    print("end_time_infer:      {}".format(time.strftime("%Y%m%d-%H-%M-%S", time.localtime(results['infer_end']))))
    print("all_time:            {}".format(results['total_dt']))
    print('pure inference time: {}'.format(results['infer_dt']))
    if debug:
        print('data load time:      {}'.format(results['data_prepare_dt']))
        print('model load time:     {}'.format(results['model_load_dt']))
        print("misc time:      {}".format(results['misc_dt']))

    if results['model_name'] in ['yolov5s_640_640','yolov5s_v5_0']:
        print('\n******************************* Metrics ({})********************************************'.format(results['model_name']))
        print("infer_mAP@mAP50:     {}".format(results['mAP50']))
        print("infer_data:          {}/predictions.json".format(results['out_dir']))
        print("latency_data:        {}/latency.txt".format(results['out_dir']))
    if results['model_name'] in ['resnet50-torchvision-v0_10_0']:
        print('\n******************************* Metrics ({})********************************************'.format(results['model_name']))
        print("infer_accuracy@Top-1:     {}".format(results['Top-1']))
        if 'Top-5' in results.keys():
            print("infer_accuracy@Top-5:     {}".format(results['Top-5']))
        print("infer_data:               {}/predictions.csv".format(results['out_dir']))
        print("latency_data:             {}/latency.txt".format(results['out_dir']))
    
        
    print('\n********************* performance by pure inference time ********************************')
    print('pure samples/sec:    {}'.format(results['samples/sec']))
    print('pure latency:        {}'.format(results['avg_latency']))

    print('\n********************* performance by total time  ********************************')
    print('samples/sec:         {}'.format(results['samples'] / results['total_dt'])) # 每秒跑多少 sample
    print('latency:             {}'.format(results['total_dt'] / results['batch_count'])) # 每个batch跑多少s
    
    

if __name__ == "__main__":

    start_time_all = time.time()

    parser = argparse.ArgumentParser(description = '...')
    parser.add_argument('--model_name', '-m', type=str, help = 'model_name', required=True)
    parser.add_argument('--thread_num', '-t', type=int, help = 'thread_num', default=-1)
    parser.add_argument('--input_batchsize', '-b', type=int, help = 'input_batchsize', default=-1)
    parser.add_argument('--npu_id', '-n', nargs='*', type=int, help = 'npu_num', default=None)
    parser.add_argument('--out_dir', '-d', type=str, help = 'out_dir', default=None)
    parser.add_argument('--request_num', '-r', type=int, help = 'request_num', default=None)
    parser.add_argument('--custom', '-c', type=str, help = 'custom', default='cty')
    parser.add_argument('--debug', '-g', action='store_true', help = 'show more info', default=False)
    args = parser.parse_args()

    model_name = args.model_name
    thread_num = args.thread_num
    input_batchsize = args.input_batchsize
    npu_id = args.npu_id
    out_dir = args.out_dir
    request_num = args.request_num
    custom = args.custom
    debug = args.debug

    if out_dir is None:
        out_dir = './log/{}_{}'.format(model_name, time.strftime("%Y%m%d%H%M%S", time.localtime(start_time_all)))
        out_dir = os.path.realpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 设置npu
    if npu_id is None:
        npu_num = 1
    else:
        npu_num = len(npu_id)
        cluster_list = []
        for n_i in npu_id:
            cluster_list += list(range( n_i * 4, n_i * 4 + 4))
        cluster_list = [str(x) for x in cluster_list]
        os.environ['STC_SET_DEVICES'] = ','.join(cluster_list)
        print('STC_SET_DEVICES={}'.format(os.environ['STC_SET_DEVICES']))

    # 初始化参数
    model_info = get_config('model_zoo/{}.json'.format(model_name))
    workload_info = get_config('workloads/{}.json'.format(model_name))
    modelzoo_json_check(model_info)
    model_convert(model_info)
    if thread_num == -1:
        thread_num = model_info['best_thread_num'] * npu_num
    if input_batchsize == -1:
        input_batchsize = workload_info['batch_fix'] * npu_num

    # 准备数据集
    print('start load dataset ....')
    t0 = time.time()
    model_info['data_percent'] = workload_info['data_percent']
    ds = load_dataset(model_info)
    ds.rebatch(input_batchsize)
    
    if request_num is not None:
        batch_count = request_num
    else:
        batch_count = ds.get_batch_count()
    batch_size = ds.get_total_batch(1)
    # samples_count = batch_size * batch_count
    dt0 = time.time() - t0
    print('end load dataset')

    # 初始化模型
    print('start load engine')
    t1 = time.time()
    engine = EngineSTC()
    engine.model_name = model_info['model']
    engine.switch_to_local()
    engine.update_compile_data(model_info)
    res = engine.load_model(thread_num)
    if not res:
        print('load model failed')
        exit(1)
    AccuracyChecker = get_accuracy_checker(model_info["dataset_name"])
    AccuracyChecker.update(engine, model_info)
    AccuracyChecker.set_dataloader(ds)
    AccuracyChecker.output_dir = out_dir
    dt1 = time.time() - t1
    print('end load engine')
    
    results = {'model_name': model_name}
    t2 = time.time()
    if request_num is not None:
        results.update(AccuracyChecker.calculate_acc(data_percent=100, request_num=request_num))
    else:
        results.update(AccuracyChecker.calculate_acc(data_percent=100))
    dt2 = time.time() - t2
    
    if 'avg_latency' in results:
        results['infer_dt'] = results['avg_latency'] * batch_count
        results['samples/sec'] = results['samples'] / results['infer_dt']
        results['misc_dt'] = dt2 - results['avg_latency'] * batch_count
    else:
        results['samples/sec'] = -1
        results['infer_dt'] = -1
        results['misc_dt'] = -1
        results['avg_latency'] = -1

    results['data_prepare_dt'] = dt0
    results['model_load_dt'] = dt1
    results['thread_num'] = thread_num
    results['npu'] = str(npu_id)
    
    show_result = {}
    show_result_1 = {}
    show_result['model_name'] = results['model_name']
    show_result['avg_latency'] = results['avg_latency']
    show_result['samples/sec'] = results['samples/sec']
    show_result['stream'] = results['thread_num']
    show_result['infer_dt'] = results['infer_dt']
    if custom == "cty":
        if 'Top-1' in results.keys():
            show_result['Top-1_acc'] = results['Top-1']
        if 'Top-5' in results.keys():
            show_result['Top-5_acc'] = results['Top-5']
        if 'mAP50' in results.keys():
            show_result['mAP50'] = results['mAP50']
        if 'accuracy' in results.keys():
            show_result['acc'] = results['accuracy']
        if 'acc' in results.keys():
            show_result['acc'] = results['acc']
        if 'precision' in results.keys():
            show_result['precision'] = results['precision']
        if 'recall' in results.keys():
            show_result['recall'] = results['recall']
        if 'hmean' in results.keys():
            show_result['hmean'] = results['hmean']
        if 'EER' in results.keys():
            show_result['eer'] = results['EER']
        if 'RTF' in results.keys():
            show_result['RTF'] = results['RTF']
        if 'race   acc@1===' in results.keys():
            show_result['race_acc'] = results['race   acc@1===']
        if 'gender acc@1===' in results.keys():
            show_result['gender_acc'] = results['gender acc@1===']
        if 'age    acc@1===' in results.keys():
            show_result['age_acc'] = results['age    acc@1===']
        if 'TPR-FPR(1E-6)' in results.keys():
            show_result_1['TPR-FPR(1E-6)'] = results['TPR-FPR(1E-6)']
        if 'TPR-FPR(1E-5)' in results.keys():
            show_result_1['TPR-FPR(1E-5)'] = results['TPR-FPR(1E-5)']
        if 'TPR-FPR(1E-4)' in results.keys():
            show_result_1['TPR-FPR(1E-4)'] = results['TPR-FPR(1E-4)']
        if 'TPR-FPR(1E-3)' in results.keys():
            show_result_1['TPR-FPR(1E-3)'] = results['TPR-FPR(1E-3)']
        if 'TPR-FPR(1E-2)' in results.keys():
            show_result_1['TPR-FPR(1E-2)'] = results['TPR-FPR(1E-2)']
        if 'TPR-FPR(1E-1)' in results.keys():
            show_result_1['TPR-FPR(1E-1)'] = results['TPR-FPR(1E-1)']
        if 'AVERAGE' in results.keys():
            show_result['AVERAGE'] = results['AVERAGE']
        if 'F1' in results.keys():
            show_result['F1'] = results['F1']
        if 'PSNR' in results.keys():
            show_result['PSNR'] = results['PSNR']
        if 'EM' in results.keys():
            show_result['EM'] = results['EM']
        if 'Easy_Val_AP' in results.keys():
            show_result['Easy_Val_AP'] = results['Easy_Val_AP']
        if 'Medium_Val_AP' in results.keys():
            show_result['Medium_Val_AP'] = results['Medium_Val_AP']
        if 'Hard_Val_AP' in results.keys():
            show_result['Hard_Val_AP'] = results['Hard_Val_AP']
        if 'AP' in results.keys():
            show_result['AP'] = results['AP']
            
        if 'Rank-1' in results.keys():
            show_result_1['Rank-1'] = round(results['Rank-1'],2)
        if 'Rank-5' in results.keys():
            show_result_1['Rank-5'] = round(results['Rank-5'],2)
        if 'Rank-10' in results.keys():
            show_result_1['Rank-10'] = round(results['Rank-10'],2)
        if 'mAP' in results.keys():
            show_result_1['mAP'] = round(results['mAP'],2)
        if 'mINP' in results.keys():
            show_result_1['mINP'] = round(results['mINP'],2)
        if 'metric' in results.keys():
            show_result_1['metric'] = round(results['metric'],2)

        if 'MOTA' in results.keys():
            show_result['MOTA'] = results['MOTA']
        if 'IDF1' in results.keys():
            show_result['IDF1'] = results['IDF1']
        if 'IDs' in results.keys():
            show_result['IDs'] = results['IDs']
        if 'MT' in results.keys():
            show_result['MT'] = results['MT']
        if 'ML' in results.keys():
            show_result['ML'] = results['ML']
            
    print(tabulate([show_result], headers="keys",tablefmt='fancy_grid'))
    if len(show_result_1) > 0:
        print(tabulate([show_result_1], headers="keys",tablefmt='fancy_grid'))
    end_time_all = time.time()
    all_time = end_time_all - start_time_all

    results['model_name'] = model_name
    results['dataset_name'] = model_info['dataset_name']
    results['out_dir'] = out_dir
    results['total_start'] = start_time_all
    results['total_end']   = end_time_all
    results['total_dt']    = all_time
    results['batch_count']  = batch_count
    results['input_batchsize']   = input_batchsize
    results['thread_num'] = thread_num
    results['npu_num'] = npu_num
    results['cluster_num'] = npu_num * 4

    print_result(results, debug=debug)
    tb.runtime.hal.stcUnRegisterObject()

