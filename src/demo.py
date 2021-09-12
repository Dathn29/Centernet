from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import torch
import os
import cv2
import time
from lib.utils.debugger import Debugger
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
def show_results( debugger, image, results,save_dir):
    debugger.add_img(image, img_id='ctdet')
    for bbox in results[1]:
      if bbox[4] > 0.3:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='ctdet')
    for i, v in debugger.imgs.items():
        cv2.imwrite(save_dir+"/detect.jpg",v)
def save_video( debugger, image, results,save_dir):
    debugger.add_img(image, img_id='ctdet')
    for bbox in results[1]:
      if bbox[4] > 0.3:
        debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='ctdet')
    return debugger
def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    print("Video will be saved to {}/detect.mp4".format(opt.save))
    t1 = time_sync()
    cam = cv2.VideoCapture(opt.demo)
    detector.pause = False
    fps = cam.get(cv2.CAP_PROP_FPS)
    w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer= cv2.VideoWriter(opt.save+"/detect.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    count =1
    while True:
        frame_count = cam.get(7)
        print("({}/{}) frame".format(count,int(frame_count)))
        _, image = cam.read()
        ret,debugger = detector.run(image)
        debugger = save_video(debugger,image,ret['results'],save_dir=opt.save)
        
        for i, v in debugger.imgs.items():
          vid_writer.write(v)
        count+=1
        if count == frame_count:
          t2 = time_sync()
          print("Total {} seconds".format(t2-t1))
          break
    cam.release()
    vid_writer.release()
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    for (image_name) in image_names:
      ret,debugger = detector.run(image_name)
      image = cv2.imread(image_name)
      show_results(debugger,image,ret['results'],save_dir=opt.save)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  demo(opt)

