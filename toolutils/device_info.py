import os
import subprocess
import argparse
import json

def write_basic_info(out_dir):

    output_filename = f'{out_dir}/device_info.json'

    try:
        # CPU型号、CPU主频
        status,res = subprocess.getstatusoutput("cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq")
        if status == 0:
            cpu_type = res.strip()
            cpu_frequency = res.split(' ')[-1]
        else:
            cpu_type =  ""
            cpu_frequency = ""

        # CPU个数
        status,res = subprocess.getstatusoutput('''cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l''')
        if status == 0:
            cpu_num = res
        else:
            cpu_num =  ""

        # CPU物理核心数
        status,res = subprocess.getstatusoutput(''' cat /proc/cpuinfo| grep "cpu cores"| uniq | awk '{print $NF}' ''')
        if status == 0:
            cpu_core_num = res
        else:
            cpu_core_num =  ""

        # CPU逻辑核心数
        status,res = subprocess.getstatusoutput(''' cat /proc/cpuinfo| grep "processor"| wc -l ''')
        if status == 0:
            cpu_logic_num = res
        else:
            cpu_logic_num =  ""

        # 单个内存条容量
        status,res = subprocess.getstatusoutput(''' dmidecode | grep -A16 "Memory Device$" |grep Size|grep -v No|uniq -c|awk -F ':' '{print $2}' ''')
        if status == 0:
            mem_size = res.strip()
        else:
            mem_size =  ""

        # 单个内存条频率
        status,res = subprocess.getstatusoutput(''' dmidecode | grep  "Max Speed"|uniq -c|awk -F ':' '{print $2}' ''')
        if status == 0:
            men_frequency = res.strip()
        else:
            men_frequency =  ""

        # 总内存条总数量
        status,res = subprocess.getstatusoutput(''' dmidecode | grep -A16 "Memory Device$" |grep Size|grep -v No|wc -l ''')
        if status == 0:
            mem_num = res.strip()
        else:
            mem_num =  ""

        # 内存总容量
        status,res = subprocess.getstatusoutput(''' cat /proc/meminfo | grep MemTotal | awk -F ':' '{print $2}' ''')
        if status == 0:
            total_mem_size = res.strip()
        else:
            total_mem_size =  ""

        info = {
            "CPU型号": cpu_type,
            "CPU主频": cpu_frequency,
            "CPU个数": cpu_num,
            "CPU物理核心数": cpu_core_num,
            "CPU逻辑核心数": cpu_logic_num,
            "单个内存条容量": mem_size,
            "单个内存条频率": men_frequency,
            "总内存条总数量": mem_num,
            "内存总容量": total_mem_size
        }

        output_info = json.dumps(info,ensure_ascii=False,indent=2)
        f = open(output_filename, "w")
        f.write(output_info)
        f.close()

    except Exception as e:
        f = open(output_filename, "w")
        f.write('无法获取当前设备信息。')
        f.close()

def main():

    # 设定文件入参格式
    parser = argparse.ArgumentParser(prog='device_info.py', description='Recording the device information')
    parser.add_argument('-o', '--out_dir', type=str, default='./tmp_out', help='Output file directory without filename. Default: ./tmp_out')
    args = parser.parse_args()

    # 读取参数
    out_dir = args.out_dir
    # 如果文件夹不存在，则创建
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)    

    # 记录当前设备静态信息
    write_basic_info(out_dir)

if __name__ == "__main__":
    main()