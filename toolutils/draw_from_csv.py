import os
import subprocess
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

def main():

    # 设定文件入参格式
    parser = argparse.ArgumentParser(prog='draw_from_csv.py', description='draw the picture from the csv file')
    parser.add_argument('-i', '--input', type=str, default='', help='Input csv file path. Default: the latest csv file under the tmp_out')
    parser.add_argument('-w', '--width', type=int, default=30, help='Width of the output picture. Unit:100px Default: 30')
    args = parser.parse_args()

    # 读取参数
    path = args.input
    width = args.width

    if path and not os.path.exists(path):
        # 如果input参数存在但是路径非法，则报错
        print('ERROR: Input path does not exist.')
        exit(2)
    elif not path:
        # 如果参数不存在则自动获取./tmp_out下最新csv文件
        # curdir = os.path.abspath(os.path.dirname(__file__))
        # status,res = subprocess.getstatusoutput(' ls -lt '+ curdir +'''/tmp_out | grep csv | awk 'NR==1{print $NF}' ''')
        status,res = subprocess.getstatusoutput(''' ls -lt ./tmp_out | grep csv | awk 'NR==1{print $NF}' ''')
        # 拼装input文件路径
        file_path = f"./tmp_out/{res}"
        # 获取文件名和扩展名
        name, ext = os.path.splitext(res)
        # 更换为新的扩展名
        new_ext = '.jpg'
        new_filename = name + new_ext
        # 输出图片文件路径
        out_file = os.path.join('./tmp_out', new_filename)
    elif os.path.isdir(path):
        # 如果参数存在且为目录，则自动获取指定目录下最新csv文件
        status,res = subprocess.getstatusoutput(''' ls -lt ''' + path + ''' | grep csv | awk 'NR==1{print $NF}' ''')
        # 拼装input文件路径
        file_path = f"{path}/{res}"
        # 获取文件名和扩展名
        name, ext = os.path.splitext(res)
        # 更换为新的扩展名
        new_ext = '.jpg'
        new_filename = name + new_ext
        # 输出图片文件路径
        out_file = os.path.join(path, new_filename)
    else:
        # 参数存在且不为目录，即为文件，则指定该文件为读取对象
        file_path = path
        # 获取文件名和扩展名
        dirname, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)
        # 更换为新的扩展名
        new_ext = '.jpg'
        new_filename = name + new_ext
        # 输出图片文件路径
        out_file = os.path.join(dirname, new_filename)

    # curdir = os.path.abspath(os.path.dirname(__file__))
    # file_path = f'{curdir}/tmp_out/capture_20230321141609.csv'

    # # 输出文件名
    # out_file = f'{curdir}/tmp_out/{os.path.splitext(os.path.basename(path))[0]}.jpg'

    try:
        # 用pandas库读取CSV文件
        csv_result = pd.read_csv(file_path)
        # 取得NPU数量
        npu_list = set(csv_result.loc[:,'no'])
        npu_cnt = len(npu_list)

        # 指定画布大小及排版
        fig, axes = plt.subplots(2, 3, figsize=(width,12))
        column = ['no','npu_util','npu_temp','npu_power','npu_memory_used','cpu_util','memory_util']
        for i in range(1, len(column)):
            if i < 5:
                for no in npu_list:
                    # 获取第一列数据，转化为字符串列表
                    x = csv_result.iloc[lambda x: x.index %npu_cnt == no, 0].to_list()
                    # 将字符串列表转化为数字列表
                    x = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in x]
                    # 获取第四列数据，转化为字符串列表
                    y = csv_result.iloc[lambda x: x.index %npu_cnt == no, i + 1].to_list()

                    if i < 4:
                        ax = axes[0][i-1]
                    else:
                        ax = axes[1][0]
                    # 作图
                    ax.plot(x, y, label=f'npu_{no}')

                # 设定各子画布的标题
                if i == 1:
                    ax.set_title(f"{column[i]}(%)")
                elif i == 2:
                    ax.set_title(f"{column[i]}(C)")
                elif i == 3:
                    ax.set_title(f"{column[i]}(W)")
                elif i == 4:
                    ax.set_title(f"{column[i]}(M)")
                # 显示曲线的标签
                ax.legend()
            else:
                # 获取第一列数据，转化为字符串列表
                x = csv_result.iloc[lambda x: x.index %npu_cnt == 0, 0].to_list()
                # 将字符串列表转化为数字列表
                x = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in x]
                # 获取第四列数据，转化为字符串列表
                y = csv_result.iloc[lambda x: x.index %npu_cnt == 0, i + 1].to_list()

                ax = axes[1][i - 4]
                # 作图
                ax.plot(x, y, label=f'npu_{no}')
                # 设定各子画布的标题
                ax.set_title(f"{column[i]}(%)")

            # 显示网格
            ax.grid(True)
            # 设定横坐标时间间隔1min
            ax.set_xticks(pd.date_range(x[0],x[-1],freq='1min'))
            # 设定横坐标文字显示偏转角度
            ax.set_xticklabels(ax.get_xticks(),rotation=45)
            # 设定横坐标显示范围
            ax.set_xlim(x[0],x[-1])
            # 设定横坐标显示格式：只显示时分
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        # 生成PNG图片文件
        plt.savefig(out_file)
        # 关闭对象，清除已有缓存
        plt.close()
    except Exception as e:
        print(f"Failed to create the picture. Please check the detail of the input file: {file_path}")
   
if __name__ == "__main__":
    main()