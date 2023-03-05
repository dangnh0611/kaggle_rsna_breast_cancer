
python -m yolox.tools.train -f exps/example/custom/yolox_s_bre_768.py -d 1 -b 32 --fp16 -o -c pretrains/yolox_s.pth


python -m yolox.tools.train -f exps/example/custom/yolox_s_bre_640.py -d 1 -b 32 --fp16 -o -c pretrains/yolox_s.pth

python -m yolox.tools.train -f exps/example/custom/yolox_s_bre_416.py -d 1 -b 32 --fp16 -o -c pretrains/yolox_s.pth
