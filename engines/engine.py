
"""
abstract engine class
"""

# pylint: disable=unused-argument,missing-docstring


class Engine(object):
    def __init__(self):
        self.hardware_type = 'UnKnown'
        self.need_reload = False
        self.need_quant = False
        self.tmpdir = ""
        self.model_name = ""
        self.npu_version = "npu-v1"

    def version(self):
        raise NotImplementedError("Engine:version")

    def pre_optimize(self, configs):
        return True

    def compile(self, configs, dataloader=None):
        raise NotImplementedError("Engine:compile")

    # reserve for future
    def tuning(self, configs):
        return True

    # reserve for future
    def segment(self, configs):
        return True

    def get_interact_profile(self, config):
        raise NotImplementedError("Engine:get_interact_profile")


    # updata compile data for a model for lazy_compile model
    def update_compile_data(self, info, npu_version="npu-v1"):
        return True
    
    # align data by input data and model batch_size
    def align_batch(self, inputs):
        return True

    # 运行编译的model,返回data对应的模型output
    def predict(self, data):
        raise NotImplementedError("Engine:predict")

    # 获取后端最佳batch_size
    def get_best_batch_size(self):
        raise NotImplementedError("Engine:get_loaded_batch_size")

    # 获取当前加载的模型的batch size
    def get_loaded_batch_size(self):
        raise NotImplementedError("Engine:get_loaded_batch_size")

    # 临时：对编译产物性能测试
    def benchmark(self, dataloader, percent=100):
        raise NotImplementedError("Engine:benchmark")

    def switch_to_local(self, version=None):
        pass