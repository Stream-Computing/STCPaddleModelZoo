# STCPaddleModelZoo使用说明
## 概述
STCPaddleModelZoo使用说明是希姆计算提供的一个机器学习基准测试工具，用于评估机器学习模型在不同硬件和软件系统上的性能和精度。STCPaddleModelZoo使用说明具有以下特点：
- 模型的运行环境更贴近客户真实业务场景。
- 降低部署复杂度，用户只需添加模型配置信息、模型任务执行配置信息和数据集信息即可执行模型推理任务。
- 尽量做到模型和框架分离，方便模型从框架中单独抽离。
- 功能模块简单，易维护，易定位，遇到问题时能快速找到问题模块。
## 前提条件
- 已安装Python 3.7。
- 已安装TensorTurbo，且版本不低于TensorTurbo 1.10。安装TensorTurbo的详细步骤，请参见异构环境安装指南。
- 已安装STC_DDK。安装的详细步骤，请参见STC_DDK使用指南。
- 已获取STCPaddleModelZoo代码仓的下载权限。
## 源码安装STCPaddleModelZoo
1. 下载STCPaddleModelZoo的源码。
$ git clone --recursive git@github.com:Stream-Computing/STCPaddleModelZoo.git
$ cd STCPaddleModelZoo
2. 安装相关依赖。
$ pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
3. 安装run_engine，方便运行多线程任务。
$  cd ci
$  ./run_engine_install.sh 
## 使用步骤
下文以ResNet50模型为例，演示了使用STCPaddleModelZoo的过程。
1. 在MinIO上的solution目录下上传原始模型文件和数据集。如果没有数据集，也可不上传数据集，使用随机生成的数据。
2. 在STCPaddleModelZoo/model_zoo目录下，上传包含模型配置信息的resnet50.json文件，文件格式参考如下：
```
{
    "model_path": "model_zoo/models/ResNet50_infer",
    "model": "resnet50",
    "framework": "PaddlePaddle",
    "nnf_version": "v1.8.1",
    "model_format": "paddlepaddle",
    "trans_format": "onnx",
    "inputs": "inputs",
    "outputs": "softmax_tensosave_infer_model/scale_0.tmp_1r_fp16",
    "input_shape": {
        "inputs": [
            "best_batch",
            3,
            224,
            224
        ]
    },
    "input_type": "FLOAT16",
    "layout": "NCHW",
    "dataset_name": "open_imagenet",
    "best_batch": 64,
    "model_precision": "FLOAT32",
    "best_thread_num": 4,
    "dataset_map": {
        "image": "input_tensor"
    }
}
```
| 参数	  | 说明|是否必选|数据类型|示例|	
|---------|-------------------|-----------|------------------|------------------------|
| model_path | 模型文件路径| 是 | str | model_zoo/models/ResNet50_infer|
| model      | 模型名 | 是 | str | resnet50 |
| framework  | 训练模型使用的机器学习框架 | 是 | str | PaddlePaddle |
| nnf_version | 训练模型使用的机器学习框架版本 | 是 | str | v1.8.1 |
| model_format | 原始模型的格式 | 否 | str | paddlepaddle |
| trans_format | 原始模型转换后的格式 | 否 | str | onnx |
| inputs | 模型输入名 | 是 | str | inputs |
| outputs | 模型输出名 | 是 | str | softmax_tensosave_infer_model/scale_0.tmp_1r_fp16 |
| input_shape | 模型输入的shape信息 | 是 | dict{str:list[int]} | "inputs": [ "best_batch", 3, 224, 224 ] |
| input_type | 模型输入数据类型 | 是 | str | FLOAT16 |
| layout | 模型输入的数据排布格式 | 否 | str | NCHW |
| dataset_name | 数据集名称 | 否 | str | open_imagenet |
| best_batch | 模型达到最优性能时使用的batch | 是 | list[int] | 64 |
| model_precision | 模型精度 | 否 | str | FLOAT32 |
| best_thread_num | 最优性能下的线程数 | 否 | int | 4 |
| dataset_map | 数据集对应关系图 | 否 | dict{数据集名称：模型输入名称} | { "image": "input_tensor" } |
| custom_function | 用户自定义随机数生成规则的代码路径 | 否 | str | custom_function/test_func.py |
| ddk_config | ddk配置文件路径 | 否 | str | model_zoo/models/2D_Unet/config.json |
| input_range | 如果dataset_name是null，采用fake_dataset运行模型，需要指明每个输入数据的数据范围 | 否 | dict{str:list[min_val, max_val]} | {"input_ids.1": [1,100], "attention_mask.1": [1,100], "token_type_ids.1": [1,100]} |
| best_thread_num | 最优性能下的线程数 | 否 | int | 8 |
3. 数据集名称对应关系表
| 数据集名称	  | 输入名|数据集输出数量|
|-------------------|-----------|------------------|
| open_cail2019 | batch_token_ids、batch_segment_ids|  1 |
| open_cifar |  	image、text | 2 |
| open_criteo_kaggle | new_categorical_placeholder、new_numeric_placeholder | 1 |
| open_imagenet | image | 1 |
| open_squad | input_ids、input_mask、segment_ids | 2 |
说明：如果模型不需要3个输入，可以根据实际情况输入。
4. 在STCPaddleModelZoo/workloads目录下，上传包含模型任务执行相关信息的resnet50.json文件，文件格式参考如下：
{
    "model": "resnet50",
    "test_perf": true,
    "test_accuracy": true,
    "test_numeric": true,
    "remote": false,
    "batch_random": 130,
    "batch_fix": 6400,
    "data_percent": 100,
    "compile_only": false
}           
| 参数	  | 说明|是否必选|数据类型|示例|	    
|---------|-------------------|-----------|------------------|----------|

| model      | 模型名，需要跟modelzoo配置文件中的模型名相同 | 是 | str | resnet50 |
| test_perf  | 是否测试性能 | 是 | bool | true |
| test_accuracy | 是否测试模型的精度 | 是 | bool | true |
| test_numeric | 是否测试数值精度 | 是 | bool | true |
| remote | 是否使用远程机器进行模型推理 | 是 | bool | false |
| batch_random | 进行模型性能测试时，采用随机的batch打包方式时的最大的batch数 | 否 | int | 130 |
| batch_fix | 进行模型性能测试时，采用固定的batch打包方式时使用的固定的batch数量 | 否 | int | 6400 |
| data_percent | 执行模型推理任务使用的数据集占比，最低不低于1个batch | 是 | int | 10 |
| compile_only | 是否只进行模型编译 | 是 | bool | true |

5. 在STCPaddleModelZoo工程目录下执行以下命令，获取推理结果。
$ python3 run_model.py -m ${model_name} 
6. 执行完成后，可以在屏显看到推理结果。
注：其中表格中samples/sec 字段代表图片处理速度、sample_num字段代表图片的数量、avg_latency字段代表平均时延(单位s)、Top-1_acc字段代表该模型精度结果。
