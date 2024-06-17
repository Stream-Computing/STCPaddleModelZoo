import argparse
import onnx
import numpy as np
import os

def compare_onnx_models(model1, model2, atol=1e-3, rtol=1e-3):
    if model1.graph != model2.graph:
        return False

    model1_weights = {tensor.name: np.array(tensor.raw_data) for tensor in model1.graph.initializer}
    model2_weights = {tensor.name: np.array(tensor.raw_data) for tensor in model2.graph.initializer}

    if len(model1_weights) != len(model2_weights):
        return False

    for name, weight1 in model1_weights.items():
        if name not in model2_weights:
            return False

        weight2 = model2_weights[name]
        if not np.allclose(weight1, weight2, atol=atol, rtol=rtol):
            return False

    return True

def main():
    parser = argparse.ArgumentParser(description="比较两个ONNX模型是否相同")
    parser.add_argument("--model_path1", "-m1", required=True, help="第一个模型的路径")
    parser.add_argument("--model_path2", "-m2", required=True, help="第二个模型的路径")

    args = parser.parse_args()

    if not os.path.exists(args.model_path1):
        print(f"模型1的路径不存在: {args.model_path1}")
        exit(1)

    if not os.path.exists(args.model_path2):
        print(f"模型1的路径不存在: {args.model_path1}")
        exit(1)

    try:
        model1 = onnx.load(args.model_path1)
    except Exception as e:
        print(f"加载模型1失败: {args.model_path1}")
        print(f"错误信息: {str(e)}")
        exit(1)

    try:
        model2 = onnx.load(args.model_path2)
    except Exception as e:
        print(f"加载模型2失败: {args.model_path2}")
        print(f"错误信息: {str(e)}")
        exit(1)

    result = compare_onnx_models(model1, model2)

    if result is False:
        print("模型不相等")
        exit(1)
    else:
        print("模型相等")

if name == "main":
    main()

