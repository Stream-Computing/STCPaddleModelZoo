import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.python.tools import freeze_graph
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import logging
import argparse
import os


def frozen_graph(h5_file_path, pb_name):
    workdir = os.path.dirname(pb_name)
    pb_name = pb_name.split("/")[-1]
    model = tf.keras.models.load_model(
        h5_file_path,
        custom_objects={
            "backend": backend,
        },
    )
    model.summary()

    full_model = tf.function(lambda input_1: model(input_1))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir=workdir, name=pb_name, as_text=False)
    print("model has been saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VC model h5->freezedpb script")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    frozen_graph(args.model_path, args.output_path)
