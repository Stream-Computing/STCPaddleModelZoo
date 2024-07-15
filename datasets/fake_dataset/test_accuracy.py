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
import collections
import pickle

log = logging.getLogger("TestAccuracy")

class AccuracyChecker(test_accuracy.AccuracyChecker):
    def calculate_acc(self, data_percent):
        log.info("Start to calculate accuracy...")
        num = int((data_percent / 100) *
                self.dataloader.get_batch_count()) if data_percent else self.dataloader.get_batch_count()
        num = 0 if data_percent == -1 else num
        num = num or 1

        diffs = collections.defaultdict(list)
        for i in tqdm(range(num)):
            test_data, _ = self.dataloader.get_samples(i)

            results = self.compiled_model.predict(test_data)
            if isinstance(results, dict):
                list_key = list(results.keys())
                list_key.sort()
                for key in list_key:
                    diffs[key].extend(results[key].flatten())
            elif isinstance(results, list):
                for i, out in enumerate(results):
                    diffs[i].extend(out.flatten())
            else:
                diffs[0].extend(results)

        for key, val in diffs.items():
            diffs[key] = np.array(val)
        # import pdb; pdb.set_trace()
        log.info('Batch size is {}, Accuracy: {}'.format(self.dataloader.cur_bs, 0.0))
        with open(self.get_vendor_data_filepath(), 'wb') as f:
            pickle.dump(diffs, f, protocol=4)

        return {"Fake Dataset Accuracy": 0}
