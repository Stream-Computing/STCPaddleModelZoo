# Copyright 2022 Stream Computing Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""

"""
import logging
import numpy as np
from datasets import test_accuracy
from tqdm import tqdm
from toolutils.common_utils import isTf, isOnnx, isPytorch
import collections
import pickle
import time
import os
import pandas as pd

log = logging.getLogger("TestAccuracy")

class AccuracyChecker(test_accuracy.AccuracyChecker):
    def calculate_acc(self, data_percent, request_num = None):
        log.info("Start to calculate accuracy...")
        good, total, top5 = 0, 0, 0
        
        total_batch_count = self.dataloader.get_batch_count()
        num = (
            int((data_percent / 100) * total_batch_count)
            if data_percent
            else self.dataloader.get_batch_count()
        )
        if request_num is not None:
            num = request_num
            
        diffs = collections.defaultdict(list)
        total_batchs = []
        latencies = []
        total_results = []
        total_labels = []
        print('inference begin')
        t_infer_start = time.time()
        for i in tqdm(range(num)):
            test_data, labels = self.dataloader.get_samples(i % total_batch_count)
            total_batchs.append(test_data[list(test_data.keys())[0]].shape[0]) 
            
            t0 = time.time()
            results = self.compiled_model.predict(test_data)
            latencies.append(time.time() - t0)

            total_results.append(results)
            total_labels.append(labels)
        
        print('inference end')
        t_infer_end = time.time()

        ## 计算并保存延迟信息
        if isinstance(request_num, int):
            latencies = latencies[:request_num]
        print('avg latency: {}'.format(np.mean(latencies)))
        latency_data_path = '{}/latency_data.txt'.format(self.output_dir)
        with open(latency_data_path, "w", encoding='utf8') as f:
            f.write('\n'.join([str(x) for x in latencies]))

        ## 计算metrics 并保存结果(超过一个完整数据集会被截断)
        total_results = total_results[:total_batch_count]
        total_labels = total_labels[:total_batch_count]
        out_path = '{}/predictions.csv'.format(self.output_dir)
        out_list = []
        # import pdb; pdb.set_trace()
        for i in range(num):
            results = total_results[i]
            labels = total_labels[i]

            if "resnet50-tf-fp16" in self.configs["model"]:
                if 'classes' in results:
                    del results['classes']
            results = self._post_processing(
                results, self.configs['framework'])
        
            for j in range(len(results)):
                tmp_ret = {}
                tmp_ret['id'] = i % total_batch_count + j
                tmp_ret['class_top1'] = np.argmax(results[j])
                tmp_ret['target'] = labels[j]
                if np.argmax(results[j]) == labels[j]:
                    tmp_ret['infer_result'] = 'pass'
                    good += 1
                else:
                    tmp_ret['infer_result'] = 'failure'
                if self.dataloader.get_top5:
                    top_ind = np.argsort(-results[j])[:5]
                    for ind in top_ind:
                        if ind == labels[j]:
                            top5 += 1
                            break
                total += 1
                out_list.append(tmp_ret)
                
            diffs[0].extend(results)
        pd.DataFrame(data=out_list).to_csv(out_path, encoding='utf-8-sig', index=False)

        print('good ===> {}'.format(good))
        print('total ===> {}'.format(total))
            
        accuracy = round((good / total), 5)
        top5_acc = round((top5 / total), 5)
        log.info('Batch size is {}, Accuracy: {}'.format(
            self.dataloader.cur_bs, accuracy))

        for key, val in diffs.items():
            diffs[key] = np.array(val)

        # with open(self.get_vendor_data_filepath(), 'wb') as f:
        #     pickle.dump(diffs, f, protocol=4)
        metric_dict = {"Top-1": accuracy, "avg_latency": np.mean(latencies),
                "infer_start":t_infer_start,"infer_end":t_infer_end,
                "samples": np.sum(total_batchs)}
        if self.dataloader.get_top5:
            metric_dict.update({"Top-5": top5_acc})
        return metric_dict
    
    def _post_processing(self, inputs, framework):
        if isinstance(inputs, list):
            inputs = list(inputs[0])
        elif isinstance(inputs, dict):
            key = list(inputs.keys())[0]
            inputs = list(inputs[key])

        if isPytorch(framework) or isOnnx(framework):
            inputs = np.array([np.insert(inputs[i], 0, 0) for i in range(len(inputs))])
        return inputs
