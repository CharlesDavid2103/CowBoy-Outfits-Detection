
'''
train
python ./train.py  --data ./data/cowboy_data_config.yaml  --cfg ./models/cowboy_yolov5l6-transformer.yaml  --hyp ./data/hyps/cowboy_hyp_config.yaml   --project ./Cow_Boy_Outfits_Detection --name yolov5l6-transformer  --img-size 640 --batch-size 2 --epochs 10 --workers 4 --weights yolov5l6.pt
'''


'''
val
python ./val.py --weights ./Cow_Boy_Outfits_Detection/yolov5l6-transformer4/weights/best.pt --data ./data/cowboy_data_config.yaml --project ./Cow_Boy_Outfits_Detection --name test_960_0.45_0.001  --task test --augment --save-json --exist-ok  --imgsz 960 --batch-size 2 --iou-thres 0.45 --conf 0.001
'''

'''
python ./detect.py --weights ./Cow_Boy_Outfits_Detection/yolov5l6-transformer4/weights/best.pt --source ./data/images/my_test
'''

'''
train
python ./train.py  --resume ./Cow_Boy_Outfits_Detection/yolov5l6-transformer5/weights/last.pt  
'''
