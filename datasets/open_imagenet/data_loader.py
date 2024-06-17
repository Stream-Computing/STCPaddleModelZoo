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
implementation of imagenet dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
from os.path import split
import re
import time

import cv2
import numpy as np
import random
from tqdm import tqdm
from toolutils.common_utils import isTf, isOnnx, isPytorch
from PIL import Image

from datasets import data_loader
import tensorflow as tf
from pathos.multiprocessing import ProcessingPool as Pool

log = logging.getLogger("Imagenet")

INPUT_TYPE = {
    "UINT8": np.uint8,
    "FLOAT16": np.float16,
    "FLOAT32": np.float32,
    "LONG": np.long,
    "INT32": np.int32,
    "INT64": np.int64,
    "BOOL": np.bool
}

def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

interpolation =  Image.BILINEAR
class DataLoader(data_loader.Dataset):
    def __init__(self, config):
        super(DataLoader, self).__init__(config)
        log.info("Initial...")

        self.config = config
        self.cur_bs = 1

        if len(config['inputs'].split(",")) != 1:
            log.error("ImageNet dataset must keep one input, but this model config \
                has {} inputs, inputs name : {}".format(len(config['inputs'].split(","), config['inputs'])))
            return None

        image_format = self.get_image_format(self.config)

        self.get_top5 = self.config.get("get_top5", False)

        global interpolation
        if self.config.get('interpolation_bicubic', False):
            interpolation = Image.BICUBIC

        if image_format == "NCHW":
            pre_process = pre_process_imagenet_pytorch
            self.image_size = self.config["input_shape"][config["inputs"]][2:4]
        else:
            pre_process = pre_process_vgg
            self.image_size = self.config["input_shape"][config["inputs"]][1:3]

        if self.config.get("inception_tf_preprocess", False):
            pre_process = pre_process_inception_tf

        process_num = self.config.get("preprocess_num", 1)

        cache_dir = os.getcwd() + \
            "/datasets/{}".format(self.config['dataset_name'])
        self.input_name = self.config['inputs']
        self.input_type = self.config["input_type"]
        self.image_list = []
        self.label_list = []
        self.count = None
        self.use_cache = 0
        self.cache_dir = os.path.join(
            cache_dir, "preprocessed", self.config['dataset_name'], image_format + "-" + str(self.image_size[0]) + "-" + str(self.image_size[1]))
        self.data_path = "datasets/{}/ILSVRC2012_img_val".format(
            self.config['dataset_name'])
        self.pre_process = pre_process
        self.items = 0
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0
        os.makedirs(self.cache_dir, exist_ok=True)

        image_list = 'datasets/{}/val_map.txt'.format(
            self.config['dataset_name'])

        total_count = iter_count(image_list)
        counter = max(int(self.config['data_percent']) * total_count // 100, 1)

        start = time.time()
        with open(image_list, 'r') as f:
            s = f.readlines()[:counter]
            part_size = (len(s) + process_num - 1) // process_num
            split_part = [s[i*part_size : min(counter, (i+1)*part_size)] for i in range(process_num)]

            def func(in_files):
                image_list = []
                label_list = []
                for s in tqdm(in_files):
                    image_name, label = re.split(r"\s+", s.strip())
                    src = os.path.join(self.data_path, image_name)
                    # if not os.path.exists(src):
                    #     # if the image does not exists ignore it
                    #     not_found += 1
                    #     continue
                    os.makedirs(os.path.dirname(os.path.join(
                        self.cache_dir, image_name)), exist_ok=True)
                    dst = os.path.join(self.cache_dir, image_name)
                    if not os.path.exists(dst + ".npy"):
                        img_org = cv2.imread(src)
                        processed = self.pre_process(
                            img_org, need_transpose=self.need_transpose, dims=self.image_size)
                        np.save(dst, processed)

                    image_list.append(image_name)
                    # label_list.append(int(label)+1)
                    label_list.append(int(label))

                return image_list, label_list

            p = Pool(process_num)
            res = p.map(func, split_part)
        
        for node in res:
            self.image_list.extend(node[0])
            self.label_list.extend(node[1])

        self.items = len(self.image_list)

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        # if not_found > 0:
        #     log.info("reduced image list, %d images not found", not_found)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), self.use_cache, time_taken))

        self.label_list = np.array(self.label_list)
        print(self.items, self.cur_bs)
        self.batch_num = int(self.items / self.cur_bs)

        self.shuffle_index = [i for i in range(self.items)]
        random.seed(7)
        random.shuffle(self.shuffle_index)
        self.rebatch(self.cur_bs, skip=False)

    def name(self):
        return self.config['dataset_name']

    def preprocess(self):
        log.info("Preprocessing...")

        self.rebatch(self.cur_bs, skip=False)


    def rebatch(self, new_bs, skip=True, low=None, high=None, base_batch=None):
        log.info("Rebatching batch size to: {} ...".format(new_bs))
        new_bs = new_bs if isinstance(new_bs, int) else new_bs[0]
        if self.cur_bs == new_bs and skip and not low and not high:
            return

        if not (low and high):
            self.cur_bs = new_bs
            low, high = new_bs, new_bs
        else:
            self.cur_bs = -1

        step = 0
        self.sample_lens = []
        random.seed(2014)
        while step < self.items:
            cnt = random.randint(low, high)
            step += cnt
            self.sample_lens.append(cnt if step <= self.items else self.items - step + cnt)
            if step > self.items and self.cur_bs != -1:
                break
            
        self.batch_num = len(self.sample_lens)
        self.batched_data = []
        self.labels = []
        pre_index = 0
        for val in tqdm(self.sample_lens):
            split_data, labels = [], []
            for j in range(pre_index, pre_index + val):
                output, label = self.get_item(self.shuffle_index[j])
                split_data.append(output)
                labels.append(label)

            self.labels.append(labels)
            self.batched_data.append({self.input_name: np.array(split_data).astype(INPUT_TYPE[self.input_type])})
            pre_index += val


    def get_samples(self, sample_id):
        if sample_id >= len(self.batched_data) or sample_id < 0:
            raise ValueError("Your Input ID: {} is out of range: {}".format(
                sample_id, len(self.batched_data)))
        return self.batched_data[sample_id], self.labels[sample_id]


    def get_item(self, nr):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        return img, self.label_list[nr]

#
# pre-processing
#
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img


def pre_process_vgg(img, dims=None, need_transpose=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output_height, output_width = dims
    cv2_interpol = cv2.INTER_AREA
    img = resize_with_aspectratio(
        img, output_height, output_width, inter_pol=cv2_interpol)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')

    # normalize image
    means = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    img -= means

    # transpose if needed
    if need_transpose:
        img = img.transpose([2, 0, 1])
    return img

def pre_process_inception_tf(img, dims=None, need_transpose=False, central_fraction=0.875):
    height, width = dims
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf.compat.v1.disable_eager_execution()
    image = tf.constant(img)
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=central_fraction)
    image = tf.expand_dims(image, 0)
    image = tf.compat.v1.image.resize_bilinear(image, [height, width], align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape([height, width, 3])
    sess = tf.compat.v1.Session()
    image = sess.run(image)
    
    # transpose if needed
    if need_transpose:
        image = image.transpose([2, 0, 1])
    return image

def pre_process_imagenet_pytorch(img, dims=None, need_transpose=False):
    import torchvision.transforms.functional as F
    global interpolation
    output_height, output_width = dims

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = F.resize(img, 256, interpolation)
    img = F.center_crop(img, (output_height, output_width))
    img = F.to_tensor(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[
                      0.229, 0.224, 0.225], inplace=False)
    if not need_transpose:
        img = img.permute(1, 2, 0)  # NHWC
    img = np.asarray(img, dtype='float32')
    return img


def maybe_resize(img, dims):
    img = np.array(img, dtype=np.float32)
    if len(img.shape) < 3 or img.shape[2] != 3:
        # some images might be grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dims != None:
        im_height, im_width, _ = dims
        img = cv2.resize(img, (im_width, im_height),
                         interpolation=cv2.INTER_LINEAR)
    return img
