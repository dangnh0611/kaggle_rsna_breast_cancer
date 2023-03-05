# TODO

- Rule-based filter. Typical cases:
    + small box downside with larger confident (small different, e.g 0.99 vs 0.991 confident)
    + overlapped TP. Reason for increase NMS thres --> keep smaller bbox instead of larger one
- Rules:
    + If large IOU --> keep smaller box
    + If up/down boxes --> keep upper box (also verify box size/area)
- Pseudo labeling --> retrain (add more val samples)
    + Add small boxes to train set to prevent FN.
    + Add double/triple boxes images to val set.

### YOLOX family
nano + 416: 1.0, 96.638
nano + 640: 1.0, 95.326
nano + 768: 1.0, 95.3
nano + 1024: 1.0, 93.4

tiny + 416: 1.0, 97.083
tiny + 416 + ms=5 : 1.0, 96.254
tiny + 416 + mosaic_ms=(0.1, 1.9): 1.0, 96.682
tiny + 416 + mosaic_ms=(0.5, 1.5): 1.0, 96.121

tiny + 640: 1.0, 96.403
tiny + 768: 1.0, 95.100
tiny + 1024: 1.0, 93.956

s + 416: 1.0, 96.082
s + 640: 1.0, 96.231


## V2
nano + 416: 95.974 @ 94.2            
nano + 416 + new_version: 96.264 @ 94.211  ->>>>>>

nano + 416 + AREA: 94.095 @  91.600

nano + 640: 95.856 @ 88.4

nano + 768: 96.217 @ 82.085

nano + 1024: 94.918 @ 89.4


tiny + 416 + old: 92.547 @ 97.083 

tiny + 416: 94.231 @ 90.2

tiny + 640: 94.950 @ 89.845

tiny + 640 + AREA: 92.778 @ 96.153  # train on v1 or v2?

tiny + 768 + AREA: 96.208 @ 68.029   !down to much

tiny + 1024 + AREA: 93.690 @ 73.698  !down to much


s + 416: 95.028 @ 0.858
s + 640: 96.104 @ 70.8
s + 768: 96.785 @ 78.7

Notes:
    - Larger image size --> overfit: tiny + >768 + AREA
    - Larger model size --> overfit: s + ? overfit on image size

