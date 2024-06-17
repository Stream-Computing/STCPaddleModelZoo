# PaddlePaddle Streamcomputing Model Zoo
### 🤝 百度飞桨 x 希姆计算AI模型库
* 兼容性适配：目前希姆计算与百度飞桨深度学习框架已完成I级兼容性适配认证，支持当下主流模型应用场景，覆盖了计算机视觉、智能语音、自然语言处理、推荐等领域，支持当下主流模型数量10+;
* 一键启动：通过兼容飞桨推理接口，用户通过指定run_model()接口一键启动推理模型，并部署在希姆NPU上执行;
* 性能评估：run_model()接口默认开启性能评估功能，用户可通过指定参数对模型进行性能评估;
* 支持拓展：用户可自行准备飞桨预训练inference模型，通过希姆NPU实现加速推理;
* 其他特性：有关run_model()接口的是详细使用方法可参考[Paddle-STCNNE](Paddle-STCNNE.md);
        
### 📦 模型信息
| Models	                    | Evaluate Datasets|Input shape	| Acc(paddle)|Acc(STC NPU)|	Latency(s)(STC NPU) | Inference Model 
|-------------------------------|-------------------|-----------|------------------|------------------------|--------------------------|--------------|
|ResNet34	                    |ImageNet1k	     |1x3x224x224   |0.7457	                |0.7578	            |0.07 | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|ResNet50	                |ImageNet1k	     |1x3x224x224   |0.7566	                |0.7549	            |0.959    | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|ResNet101        	                |ImageNet1k	     |1x3x224x224   |0.7756               |0.7749	            |0.1051 | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|DenseNet121	                        |ImageNet1k	     |1x3x224x224   |0.7566	                |0.7753	            |0.0845   | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|VGG16	                    |ImageNet1k	     |1x3x224x224   | 0.72	                |0.7226	            |0.4364 | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|EfficientNetB0	                        |ImageNet1k	     |1x3x224x224   |0.7738	                |0.8046	            |0.1258   | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|EfficientNetB1        	            |ImageNet1k	     |1x3x224x224   |0.7915	                |0.8203	            |0.166     | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|EfficientNetB5	                        |ImageNet1k	     |1x3x224x224   | 0.7215	                |0.8203	            |0.0395     | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|ShuffleNetV2_x1_0	                        |ImageNet1k	     |1x3x224x224   |0.688                |0.6972	            |0.04    | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 )  
|SlowFast	                |ImageNet1k	     |1x3x224x224   |74.35	                |74.99	            |0.03      | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 )   
|NCF	                    |ImageNet1k	     |1x3x224x224   |HR@10：0.58NDCG@10：0.33|HR@10：0.60NDCG@10：0.33	            |0.01      | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|DeepFM	                    |ImageNet1k	     |1x3x224x224   |0.78                |0.7745	            |0.044     | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 ) 
|DIEN	                |ImageNet1k	     |1x3x224x224   |0.7133	                |0.71332	            |2.968     | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 )   
|MMOE	                |ImageNet1k	     |1x3x224x224   |0.99                |0.9922            |0.0021    | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 )   
|DLRM	                |ImageNet1k	     |1x3x224x224   |0.79               |0.7869	            |0.01      | [inference_model](https://pan.baidu.com/s/1nT_1K7kdjs6Ydq2Ctdyxzw?pwd=x033 )   
### 🎈 推理预测
#### 以图像分类为例简要介绍模型使用方法，其他模型场景详细用法请参考飞桨官方模型库：

##### 1.模型准备： 通过链接下载希姆飞桨ImageNet1K图像分类模型，例如 MobileNetV3.pdmodel 、 MobileNetV3.pdiparams，文件夹放到model_zoo目录下
##### 2.数据准备： 输入图像应符合NCHW Format , Shape 为 [1,3,224,224]，也可以通过链接下载数据集，将数据集文件夹中的文件下载后放到datasets目录下
##### 3.模型编译：
```bash
python3 compile_model.py -m  ${MODEL_NAME}
```
##### 3.执行推理：
```bash
python3 run_model.py  -m  ${MODEL_NAME}
```
模型文件和数据集自动从model_zoo/${MODEL_NAME}.json中加载。
##### 4.获取最终推理结果，如图像类别、OCR检测结果等，可参考飞桨模型库相关代码     