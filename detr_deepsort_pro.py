import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_DETR
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

# count
from collections import Counter
from collections import deque
import math
from PIL import Image, ImageDraw, ImageFont

# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_DETR()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        # self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            if not os.path.exists(self.args.save_path+self.args.output_name):
                os.mkdir(self.args.save_path+self.args.output_name)
                
            self.save_video_path = os.path.join(self.args.save_path, self.args.output_name + "/results.avi")
            self.save_results_path = os.path.join(self.args.save_path, self.args.output_name + "/results.txt")

            # create video writer
            #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def get_size_with_pil(self,label,size=25):
        font = ImageFont.truetype("./configs/simkai.ttf", size, encoding="utf-8")  # simhei.ttf
        return font.getsize(label)

    def compute_color_for_labels(self,class_id,class_total=80):
        offset = (class_id + 0) * 123457 % class_total;
        red = self.get_color(2, offset, class_total);
        green = self.get_color(1, offset, class_total);
        blue = self.get_color(0, offset, class_total);
        return (int(red*256),int(green*256),int(blue*256))

    #为了支持中文，用pil
    def put_text_to_cv2_img_with_pil(self,cv2_img,label,pt,color):
        pil_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
        pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
        draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
        font = ImageFont.truetype("./configs/simkai.ttf", 25, encoding="utf-8") #simhei.ttf
        font = ImageFont.truetype("./configs/Corporate-Logo-Bold-ver2.ttf", 25, encoding="utf-8") #Corporate-Logo-Bold-ver2.ttf
        draw.text(pt, label, color,font=font)
        return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式

    def tlbr_midpoint(self,box):
        minX, minY, maxX, maxY = box
        midpoint = (int((minX + maxX) / 2), int((minY + maxY) / 2))  # minus y coordinates to get proper xy format
        return midpoint

    def ccw(self,A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self,A, B, C, D): return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def vector_angle(self,midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))

    def get_color(self,c, x, max):
        colors = np.array([
        [1,0,1],
        [0,0,1],
        [0,1,1],
        [0,1,0],
        [1,1,0],
        [1,0,0]
        ]);
        ratio = (x / max) * 5;
        i = math.floor(ratio);
        j = math.ceil(ratio);
        ratio -= i;
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c];
        return r;

    def run(self):
        results = []
        idx_frame = 0
        paths = {}
        track_cls = 0
        last_track_id = -1
        total_track = 0
        angle = -1
        total_counter = 0
        up_count = 0
        down_count = 0
        class_counter = Counter()   # store counts of each detected class
        already_counted = deque(maxlen=100)   # temporary memory for storing counted IDs
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im, conf_threshold=0.9)

            # select person class
            mask = cls_ids == CLASSES.index('person')
            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            # bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]
          
            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # 1.视频中间画线
            #line = [(0, int(0.48 * ori_im.shape[0])), (int(ori_im.shape[1]), int(0.48 * ori_im.shape[0]))]
            line = [(0, int( ori_im.shape[0])), (int(ori_im.shape[1]), int(0.48 * ori_im.shape[0]))]
            cv2.line(ori_im, (line[0]), line[1], (255, 255, 255), 4)

            # 2. 统计人数
            for track in outputs:
                bbox = track[:4]
                track_id = track[-1]
                midpoint = self.tlbr_midpoint(bbox)
                origin_midpoint = (midpoint[0], ori_im.shape[0] - midpoint[1])  # get midpoint respective to botton-left

                if track_id not in paths:
                    paths[track_id] = deque(maxlen=2)
                    total_track = track_id
                paths[track_id].append(midpoint)
                previous_midpoint = paths[track_id][0]
                origin_previous_midpoint = (previous_midpoint[0], ori_im.shape[0] - previous_midpoint[1])

                if self.intersect(midpoint, previous_midpoint, line[0], line[1]) and track_id not in already_counted:
                    class_counter[track_cls] += 1
                    total_counter += 1
                    last_track_id = track_id;
                    # draw red line
                    cv2.line(ori_im, line[0], line[1], (0, 0, 255), 10)

                    already_counted.append(track_id)  # Set already counted for ID to true.

                    angle = self.vector_angle(origin_midpoint, origin_previous_midpoint)

                    if angle > 0:
                        up_count += 1
                    if angle < 0:
                        down_count += 1

                if len(paths) > 50:
                    del paths[list(paths)[0]]

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, identities))

            # 4. 绘制统计信息
            label = "人数: {}".format(str(total_track))
            t_size = self.get_size_with_pil(label, 25)
            x1 = 20
            y1 = 30
            color = self.compute_color_for_labels(2)
            #cv2.rectangle(ori_im, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_im = self.put_text_to_cv2_img_with_pil(ori_im, label, (x1 + 5, y1 - t_size[1] - 2), (255, 255, 255))

            label = "{}人白線を通過  ({} ↑, {} ↓)".format(str(total_counter), str(up_count), str(down_count))
            t_size = self.get_size_with_pil(label, 25)
            x1 = 20
            y1 = 60
            color = self.compute_color_for_labels(2)
            #cv2.rectangle(ori_im, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
            ori_im = self.put_text_to_cv2_img_with_pil(ori_im, label, (x1 + 5, y1 - t_size[1] - 2), (255, 255, 255))

            if last_track_id >= 0:
                label = "最新: {}番{}白線を通過".format(str(last_track_id), str("↑") if angle >= 0 else str('↓'))
                t_size = self.get_size_with_pil(label, 25)
                x1 = 20
                y1 = 90
                color = self.compute_color_for_labels(2)
                #cv2.rectangle(ori_im, (x1 - 1, y1), (x1 + t_size[0] + 10, y1 - t_size[1]), color, 2)
                ori_im = self.put_text_to_cv2_img_with_pil(ori_im, label, (x1 + 5, y1 - t_size[1] - 2), (255, 0, 0))

            end = time.time()

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--output_name", type=str, default="results")
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
