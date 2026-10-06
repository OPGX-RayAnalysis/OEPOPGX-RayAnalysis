"""Crop + CLAHE must keep a YOLO box on the object. No dataset, GPU, or weights."""

import cv2
import numpy as np

from opg.geometry import iou
from opg.preprocess import crop_black_border, preprocess, shift_yolo_box


def test_box_stays_on_object_after_crop(tmp_path):
    # 300x200, 20 px black border, grey interior, one white rectangle.
    img = np.zeros((200, 300), dtype=np.uint8)
    img[20:180, 20:280] = 80
    x1, y1, x2, y2 = 80, 60, 140, 120
    img[y1:y2, x1:x2] = 255

    path = tmp_path / "synthetic.png"
    assert cv2.imwrite(str(path), img)

    orig_h, orig_w = img.shape
    cx = (x1 + x2) / 2 / orig_w
    cy = (y1 + y2) / 2 / orig_h
    bw = (x2 - x1) / orig_w
    bh = (y2 - y1) / orig_h

    result = preprocess(path)
    cleaned, offset = result["image"], result["offset"]
    cropped, crop_offset = crop_black_border(img)
    assert offset == crop_offset == (20, 20)
    assert cleaned.shape == cropped.shape == (160, 260)

    shifted = shift_yolo_box(cx, cy, bw, bh, orig_w, orig_h, offset, cleaned.shape[1], cleaned.shape[0])
    assert shifted is not None
    ncx, ncy, nw, nh = shifted
    box = (
        (ncx - nw / 2) * cleaned.shape[1],
        (ncy - nh / 2) * cleaned.shape[0],
        (ncx + nw / 2) * cleaned.shape[1],
        (ncy + nh / 2) * cleaned.shape[0],
    )
    # White rectangle in the cropped image is (60, 40)-(120, 100).
    assert iou(box, (60, 40, 120, 100)) > 0.99
    assert cleaned[40:100, 60:120].min() >= 200
