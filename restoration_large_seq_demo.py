import argparse
import os
import sys
import threading
from glob import glob

import mmcv
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from mmedit.apis import init_model
from mmedit.core import tensor2img
from mmedit.utils import modify_args
from mmedit.datasets.pipelines import Compose


class PreRead:
    def __init__(self, iterator, bufsize=1, num=float("inf")):
        self._iterator = iterator
        self._bufsize = bufsize
        self._cache = []
        self._num = num

    def __iter__(self):
        self._readthread = threading.Thread(target=self._read, daemon=True)
        self._ready_event = threading.Event()
        self._read_event = threading.Event()
        self._iterator = iter(self._iterator)
        self._readthread.start()
        return self

    def __next__(self):
        if len(self._cache) == 0:
            if self._readthread.is_alive():
                self._ready_event.clear()
                self._ready_event.wait()
            else:
                raise StopIteration
        ret = self._cache.pop(0)
        self._read_event.set()
        return ret

    def _read(self):
        count = 0
        while True:
            if count >= self._num:
                break
            if len(self._cache) < self._bufsize:
                try:
                    self._cache.append(next(self._iterator))
                    count += 1
                    self._ready_event.set()
                except StopIteration:
                    break
            else:
                self._read_event.wait()
                self._read_event.clear()


def parse_args(arg=None):
    modify_args()
    parser = argparse.ArgumentParser(description="Restoration demo with optimised memory")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument("input_dir", help="directory containing input PNG images")
    parser.add_argument(
        "output_dir",
        help="directory of the output images "
             'or use "-" to directly output to stdout (rawvideo in rgb24 pixel format)',
    )
    parser.add_argument("--filename-tmpl", default="{:08d}.png", help="template of the file names")
    parser.add_argument(
        "--max-seq-len", type=int, default=1, help="maximum sequence length if recurrent framework is used"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device id")
    args = parser.parse_args(arg)
    return args


def main(arg=None, pipe=None):
    """Demo for image restoration models."""

    args = parse_args(arg)

    if not os.path.isdir(args.input_dir):
        print("input_dir must be a directory containing image files! ", file=sys.stderr)
        raise SystemExit(1)
    if args.output_dir != "-" and os.path.exists(args.output_dir) and not os.path.isdir(args.output_dir):
        print('output_dir only accept path or "-" (stdout rawvideo in rgb24) ! ', file=sys.stderr)
        raise SystemExit(1)

    if args.output_dir == "-" and pipe is None:
        sys.stdout = sys.stderr
        pipe = sys.__stdout__.buffer

    model = init_model(args.config, args.checkpoint, device=torch.device("cuda", args.device))
    device = next(model.parameters()).device

    # build the data pipeline
    if model.cfg.get("demo_pipeline", None):
        test_pipeline = model.cfg.demo_pipeline
    elif model.cfg.get("test_pipeline", None):
        test_pipeline = model.cfg.test_pipeline
    else:
        test_pipeline = model.cfg.val_pipeline

    img_files = sorted(glob(os.path.join(args.input_dir, '*.png')))
    tmp_pipeline = []
    for pipeline in test_pipeline:
        if pipeline["type"] not in ["GenerateSegmentIndices", "LoadImageFromFileList"]:
            tmp_pipeline.append(pipeline)
    test_pipeline = tmp_pipeline
    test_pipeline = Compose(test_pipeline)

    # 获取输入图像尺寸并计算填充
    first_frame = mmcv.imread(img_files[0])
    input_img_size = first_frame.shape[:2]#(411, 618)
    pad = (0, (4 - input_img_size[0] % 4) % 4, 0, (4 - input_img_size[1] % 4) % 4)
    print("input image size:", input_img_size, file=sys.stderr)
    if any(pad):
        print("pad:", pad, file=sys.stderr)

    frames = []
    frame_num = len(img_files)
    printed_output_size = False

    pre_read_iterator = PreRead(img_files, bufsize=args.max_seq_len)
    selected = ['_DSC8679.png', '_DSC8687.png', '_DSC8695.png', '_DSC8703.png', '_DSC8711.png', '_DSC8719.png',
                '_DSC8727.png', '_DSC8735.png', '_DSC8744.png', '_DSC8752.png', '_DSC8760.png', '_DSC8768.png',
                '_DSC8776.png', '_DSC8784.png', '_DSC8792.png', '_DSC8800.png', '_DSC8808.png', '_DSC8816.png',
                '_DSC8824.png', '_DSC8832.png', '_DSC8840.png', '_DSC8848.png', '_DSC8856.png', '_DSC8864.png',
                '_DSC8872.png']
    for i, img_file in tqdm(enumerate(pre_read_iterator), unit="frame", total=frame_num, maxinterval=1.0):
        frame = mmcv.imread(img_file)
        if frame is None or frame.ndim != 3:
            print(f"Error reading image {img_file}, skipping this file.", file=sys.stderr)
            continue
        if (i + 1) % args.max_seq_len and i + 1 != frame_num:
            frames.append(np.flip(frame, axis=2))
            continue

        frames.append(np.flip(frame, axis=2))
        data = dict(lq=frames, lq_path=None, key=args.input_dir)
        data_chunk = test_pipeline(data)["lq"].unsqueeze(0)
        if len(frames) == 1:
            data_chunk = data_chunk.unsqueeze(0)
        data_chunk = torch.nn.functional.pad(data_chunk, pad, "constant", 0)
        frames = []
        res = model(lq=data_chunk.to(device), test_mode=True)["output"].cpu()#[1, 1, 3, 1652, 2476]
        ratio = res.shape[-1] / data_chunk.shape[-1]# 4

        output_frame = tensor2img(res[:, :, :, :int(input_img_size[0] * ratio), :int(input_img_size[1] * ratio)])
        if not printed_output_size:
            print("output image size:", PIL.Image.fromarray(output_frame).size, file=sys.stderr)
            printed_output_size = True
        if os.path.basename(img_file) in selected:
            mmcv.imwrite(output_frame, os.path.join(args.output_dir, os.path.basename(img_file)))  # _DSC8679.png

if __name__ == "__main__":
    main()
