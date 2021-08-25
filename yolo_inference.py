#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
import sys

import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
vid_formats = ['.mov', '.avi', '.mp4']


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class LoadImages:  # for inference
    def __init__(self, path, img_size=416):
        path = str(Path(path))  # os-agnostic
        files = []
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, '*.*')))
        elif os.path.isfile(path):
            files = [path]

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        nI, nV = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nF = nI + nV  # number of files
        self.video_flag = [False] * nI + [True] * nV
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nF > 0, 'No images or videos found in ' + path

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nF:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nF:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nF, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nF, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nF  # number of files


if __name__ == '__main__':
    import ctypes
    import os
    import shutil
    import random
    import sys
    import time
    import cv2
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt
    import torch
    import torchvision

    CONF_THRESH = 0.5
    IOU_THRESHOLD = 0.4

    # load custom plugins
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov5s.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    if os.path.exists('output/'):
        shutil.rmtree('output/')
    os.makedirs('output/')
    # a YoLov5TRT instance
    # yolov5_wrapper = YoLov5TRT(engine_file_path)

    # Create a Context on this device,
    ctx = cuda.Device(0).make_context()
    stream = cuda.Stream()
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(TRT_LOGGER)

    # Deserialize the engine from file
    with open(engine_file_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        # We always use batch size 1.
        input_shape = (1, 3*416*352)
        # (3, 416, 352)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

        # Allocate device memory for inputs.
        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        for binding in range(3):
            context.set_binding_shape(binding, input_shape)
        assert context.all_binding_shapes_specified

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)

        def inference(features):
            print("\nRunning Inference...")
            eval_start_time = time.time()

            # Copy inputs
            # cuda.memcpy_htod_async(d_inputs[0], features["input_ids"], stream)
            # cuda.memcpy_htod_async(d_inputs[1], features["segment_ids"], stream)
            # cuda.memcpy_htod_async(d_inputs[2], features["input_mask"], stream)
            cuda.memcpy_htod_async(d_inputs[2], features, stream)
            # Run inference
            context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()

            eval_time_elapsed = time.time() - eval_start_time

            print("------------------------")
            print("Running inference in {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
            print("------------------------")
            print(h_output.shape)
            # (1, 256, 12, 1, 1)

            # ner /work/ner_convert_weight_output_path/bert_base_512.engine
            batch_h_output = np.squeeze(h_output, axis=(3, 4))
            # if h_output.shape[2] == 768:
            #     # [ 8 10  2  2 11 11 11 11 11 11 11 11 11 11 11 11  5 11 11 11]
            #     flat_batch_h_output = batch_h_output.reshape(-1, 768)
            #     flat_batch_rst = np.dot(flat_batch_h_output, project_logits_w.reshape(768, 12)) + project_logits_b
            #     batch_rst = flat_batch_rst.reshape(-1, 512, 12)
            # else:
                # 2
                # [10 10 10 10 11 11 11  3  3 11  3  3  3 11 11 11 10  5  5  5]
            batch_rst = batch_h_output

            batch_rst = np.tanh(batch_rst)
            class_idx = batch_rst.argmax(axis=2)
            predict_label = np.vectorize({}.get)(class_idx)
            print(predict_label[0][0:30])
            print(class_idx[0][0:30])


        EXIT_CMDS = ["exit", "quit", "q"]

        question_text = input("Question (to exit, type one of {:}): ".format(EXIT_CMDS))
        while question_text.strip() not in EXIT_CMDS:
            # features = question_features(question_text)
            features = []
            image_path = "/samples"
            image_list = LoadImages(image_path)
            for a_image in image_list:
                import pdb
                pdb.set_trace()
                print(a_image[2].shape)
                print(a_image[1].shape)
                # cv2.imshow(a_image[0], a_image[1])
                # cv2.imshow(a_image[0], a_image[2])
                inference(a_image[2])

# ~/tensorrtx/yolov5/build/
# 'build/'
# 'output/'
# "build/libmyplugins.so"
# "samples/"
