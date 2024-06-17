#! /bin/python
import fire
import tensorturbo
import tb
import os
import tvm
from tb.relay.graph_schedule import relay

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def model_convert(model, input_format, input_shape, output):
    inputs = []
    input_tensor = {}
    for node in input_shape.split(";"):
        for i in reversed(range(len(node))):
            if node[i] == ":":
                break
        name, sp = node[:i], node[i+1: ]
        dims = []
        inputs.append(name)
        for a in sp.split(","):
            dims.append(int(a))
        input_tensor[name] = dims
    print(input_tensor)
    print(inputs)
    out_path = output + '.stcobj'
    suffix = model.split(".")[-1]
    # 导入模型
    if suffix == "pb":
        print(model, inputs)
        resnet50_v2 = tensorturbo.model.from_tensorflow(model)
        resnet50_v2.set_input_shape(input_tensor)
    elif suffix == "onnx":
        resnet50_v2 = tensorturbo.model.from_onnx(model, input_tensor)
    # 获取、修改模型信息
    print(resnet50_v2.get_input_info())
    print(resnet50_v2.get_output_info())

    if suffix =="onnx":
        desired_layouts = {"nn.conv2d": ["NHWC", "HWIO"], "nn.max_pool2d": ["NHWC", "HWIO"], "nn.global_avg_pool2d": ["NHWC", "HWIO"]}
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            resnet50_v2.relay = seq(resnet50_v2.relay)

    resnet50_v2.compile(
        output_file=out_path,
        schedule=None,
        required_pass=[]
    )

if __name__ == '__main__':
    fire.Fire(model_convert)

