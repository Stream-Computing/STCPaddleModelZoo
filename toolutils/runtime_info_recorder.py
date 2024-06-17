import os
import subprocess
import re
from datetime import datetime
import time
import argparse
import logging
import json
import asyncio
import signal

DATE_FMT = "%Y-%m-%d %H:%M:%S"
FORMAT = "%(asctime)s,%(message)s"

def handle_sigint(sig, frame):
    print('Exit recording......')
    exit(0)

# 注册信号处理函数
signal.signal(signal.SIGINT, handle_sigint)

# 将stc-smi命令取得的结果保存进列表对象里返回
def get_stc_smi_info():
    info_list = []
    block = []
    # 执行stc-smi命令
    status,res = subprocess.getstatusoutput("stc-smi")
    # 执行结果不为0，命令执行错误，报错，返回错误信息
    if status != 0:
        return info_list

    # 将执行结果根据换行符切分保存到列表
    results = res.split("\n")
    # 获取Driver Version
    driverVersion = results[1].split(':')[2].replace('|','').strip()
    # 获取并保存两处等号分隔符所在的行号，便于分别处理
    for i in range(len(results)):
        if re.match("^\+=+\+$",results[i]) is not None:
            block.append(i)

    # 通过两处分隔符所在行号算出NPU个数
    npu_num = int((block[1] - block[0] - 5) / 3)
    # 上半部分起始index
    start_index_p1 = block[0] + 1
    # 上半部分结束index
    end_index_p1 = start_index_p1 + npu_num * 3
    # 遍历上半部分内容
    for i in range(start_index_p1, end_index_p1, 3):
        npu_info = {}

        # 取得第一行内容
        line_1 = results[i].split("|")
        npu_info["no"] = line_1[1][:5].strip()
        npu_info["name"] = line_1[1][5:17].strip()
        npu_info["frequency"] = line_1[1][17:].strip()
        npu_info["bus_id"] = line_1[2].strip()
        npu_info["npu_util"] = line_1[3].strip()
        npu_info["driver_version"] = driverVersion
        # 取得第二行内容
        line_2 = results[i + 1].split("|")
        npu_info['fan'] = line_2[1][:5].strip()
        npu_info['temp'] = line_2[1][5:17].strip()
        npu_info['power_curr'] = line_2[1][17:25].strip()
        column = line_2[1].split("/")
        npu_info['power_total'] = column[-1].strip()
        npu_info["cluster_count"] = line_2[2].strip()
        memory_list = line_2[3].split("/")
        npu_info["memory_used"] = memory_list[0].strip()
        npu_info["memory_total"] = memory_list[1].strip()
        # CLUSTER信息占位
        npu_info["clusters"] = []

        info_list.append(npu_info)

    # 通过第二处分隔符所在行号算出板卡进程个数
    proc_num = int((len(results) - block[1] - 1) / 2)
    # 下半部分起始index
    start_index_p2 = block[1] + 1
    # 下半部分结束index
    end_index_p2 = start_index_p2 + proc_num * 2
    # 收集CLUSTER相关信息，装入对应字典的clusters里
    for i in range(start_index_p2, end_index_p2, 2):
        # 获取边栏中间的内容
        data = results[i].split("|")[1]
        # 碰到无进程的行跳过
        if data.strip() == "There is no process.":
            continue
        dict_cluster = {}
        dict_cluster["cluster"] = data[6:14].strip()
        dict_cluster["pid"] = data[14:28].strip()
        dict_cluster["process_name"] = data[30:67].strip()
        dict_cluster["memory_used"] = data[68:].strip()

        # npu编号即为列表index
        npu = data[:6].strip()
        index = int(npu)
        # 将当前字典加进对应字典的clusters列表里
        info_list[index]["clusters"].append(dict_cluster)

    return info_list

def get_cpu_mem_util(name):
    
    # 执行命令获取cpu和内存使用率
    status,res = subprocess.getstatusoutput("docker stats --no-stream | grep " + name + " | awk '{print $3,$7}'")
    if status != 0 or not res:
        return 0, 0
    
    result = res.split(' ')
    cpu_util = float(result[0][:-1])
    mem_util = float(result[1][:-1])

    return cpu_util, mem_util

# 收集硬件信息的多线程任务
async def getData(name):
    try:
        # 获取smi信息
        smi_info = get_stc_smi_info()
        if not smi_info:
            print('Error: No stc-smi result!')
            exit(2)
        cpu_util, mem_util = get_cpu_mem_util(name)

        table = []
        # 遍历板卡
        for npu in smi_info:
            # 板卡编号
            no = int(npu.get('no'))
            # npu利用率
            npu_util = float(npu.get('npu_util')[:-1])
            # 温度
            npu_temp = float(npu.get('temp')[:-1])
            # 当前功耗
            npu_power_curr = float(npu.get('power_curr')[:-1])
            # 当前内存使用 注：由于单位会从M变为G，所以需要乘以1024
            npu_memory_used = float(npu.get('memory_used')[:-1]) if npu.get('memory_used')[-1] == 'M' else float(npu.get('memory_used')[:-1])*1024

            # 输出日志
            logging.info(f'{no},{npu_util},{npu_temp},{npu_power_curr},{npu_memory_used},{cpu_util},{mem_util}')
            # 打印控制台
            table.append([datetime.now().strftime(DATE_FMT),no,npu_util,npu_temp,npu_power_curr,npu_memory_used,cpu_util,mem_util])

        # 控制台打印最新信息（擦除旧信息）
        os.system(r'clear')
        # 各列对齐
        title = ['timestamp','no','npu_util','npu_temp','npu_power','npu_memory_used','cpu_util','memory_util']
        print("%-20s %3s %10s %10s %10s %16s %10s %13s" % (title[0], title[1], title[2], title[3], title[4], title[5], title[6], title[7]))
        for row in table:
            print("%-20s %3s %10s %10s %10s %16s %10s %13s" % (row[0], row[1], f"{row[2]} %", f"{row[3]} C", f"{row[4]} W", f"{row[5]} M", f"{row[6]} %", f"{row[7]} %"))
    except KeyboardInterrupt:
        pass

async def wait(interval):
    await asyncio.sleep(5)

# 启动多线程任务记录硬件信息
async def recording(interval, name):
    while(True):
        # 注：记录硬件信息本身需要花费一定时间。单线程的话总时间可能大于指定时间（比如5秒变成5.5秒之类）
        # 为了解决这个问题，启用多线程，sleep方法不用等待前置任务结束再实行，可以实现更精准的时间间隔
        task1 = asyncio.create_task(wait(interval))
        task2 = asyncio.create_task(getData(name))
        await asyncio.gather(task1, task2)

def main():

    # 设定文件入参格式
    parser = argparse.ArgumentParser(prog='device_info_recorder.py', description='Recording the device information')
    parser.add_argument('-o', '--out_dir', type=str, default='./tmp_out', help='Output file directory without filename. Default: ./tmp_out')
    parser.add_argument('-i', '--interval', type=int, default=5, help='Interval time of recording. Unit:Second. Default: 5')
    parser.add_argument('-n', '--name', type=str, default='cty_test_stc', help='Container Name to get the runtime info from. Default: cty_test_stc')
    args = parser.parse_args()

    # 读取参数
    out_dir = args.out_dir
    interval = args.interval
    container_name = args.name
    # 如果文件夹不存在，则创建
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    

    # 当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # 输出文件名
    filename = f'capture_{timestamp}.csv'
    out_file = f'{out_dir}/{filename}'

    # 设定日志输出格式
    LEVEL = logging.INFO
    logging.basicConfig(level=LEVEL, format=FORMAT, datefmt=DATE_FMT, filename=out_file)
    
    # 输出Title
    column = ['no','npu_util','npu_temp','npu_power','npu_memory_used','cpu_util','memory_util']
    logging.info(','.join(column))

    # 启动收集信息的线程
    try:
        asyncio.run(recording(interval, container_name))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()