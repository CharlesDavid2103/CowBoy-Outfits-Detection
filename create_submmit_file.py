'''
项目总结
1.预处理数据 preprocessing_data.py
    按类别比例 选择评估图片
    按yolov5的数据结构复制图片到相应的文件夹下
    创建训练、评估时需要的标签文件(转换box坐标)
    复制测试集图片
2.创建项目使用yolov5的配置文件
    --data ./data/cowboy_data_config.yaml           数据配置
    --cfg ./models/cowboy_yolov5l6-transformer.yaml 模型配置
    --hyp ./data/hyps/cowboy_hyp_config.yaml        超参数初始配置
3.调用yolov5 train开始训练
    python ./train.py  --data ./data/cowboy_data_config.yaml  --cfg ./models/cowboy_yolov5l6-transformer.yaml  --hyp ./data/hyps/cowboy_hyp_config.yaml   --project ./Cow_Boy_Outfits_Detection --name yolov5l6-transformer  --img-size 640 --batch-size 2 --epochs 10 --workers 8 --weights yolov5l6.pt
4.查看训练效果
    python ./detect.py --weights ./Cow_Boy_Outfits_Detection/yolov5l6-transformer4/weights/best.pt --source ./data/images/my_test
5.对测试集进行预测
    python ./val.py --weights ./Cow_Boy_Outfits_Detection/yolov5l6-transformer4/weights/best.pt --data ./data/cowboy_data_config.yaml --project ./Cow_Boy_Outfits_Detection --name test_960_0.45_0.001  --task test --augment --save-json --exist-ok  --imgsz 960 --batch-size 2 --iou-thres 0.45 --conf 0.001

6.根据最好的预测结果生成提交文件
    create_submmit_file.py
'''


import json
import zipfile
import os
import pandas as pd


categories = [87, 1034, 131, 318, 588]
trans = {f'{each}': f'{idx}' for (idx, each) in enumerate(categories)}

orginal_image_path = './images/'  # 原始图片位置

test_img_path = './data/images/test/'
os.makedirs(test_img_path, exist_ok=True)


os.makedirs(test_img_path, exist_ok=True)
test_data_file_path = './exc_data/test.csv'
test_data = pd.read_csv(test_data_file_path)


#打开最后的结果
submission = json.load(open('./Cow_Boy_Outfits_Detection/test_960_0.45_0.001/best_predictions.json', 'r'))
trans_imid = {f"{i.split('.')[0]}": j for i, j in zip(test_data['file_name'], test_data['id'])}
trans_cid = {k: v for (v, k) in trans.items()}
for each in submission:
    each['image_id'] = trans_imid[f"{each['image_id']}"]
    each['category_id'] = trans_cid[f"{each['category_id']}"]

with open('./Cow_Boy_Outfits_Detection/answer.json', 'w') as f:
    json.dump(submission, f)

# Get answer.zip file
zf = zipfile.ZipFile('./Cow_Boy_Outfits_Detection/answer.zip', 'w')
zf.write('./Cow_Boy_Outfits_Detection/answer.json', 'answer.json')
zf.close()
