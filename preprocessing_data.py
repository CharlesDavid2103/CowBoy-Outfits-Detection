import os
import shutil
import json
import yaml
import random
import pandas as pd

data = json.load(open('exc_data/train.json', 'r'))
ann = data['annotations']
random.seed(34)
random.shuffle(ann)

'''
'categories' 类别
'id': 87,   'name': 'belt'          腰带
'id': 1034, 'name': 'sunglasses'    太阳镜
'id': 131,  'name': 'boot'          靴子
'id': 318,  'name': 'cowboy_hat'    牛仔帽
'id': 588,  'name': 'jacket'        夹克
'''
categories = [87, 1034, 131, 318, 588]
categories_nums = {}

# 查看训练集中各类别数量

for i in categories:
    for j in ann:
        if j['category_id'] == i:
            name = 'category_' + str(i)
            categories_nums[name] = categories_nums.get(name, 0) + 1
total_num = sum(categories_nums.values())
print('total:', total_num)
print('categories_nums', categories_nums)

# 按比例选取图片
total_id = set(each['image_id'] for each in ann)
val_id = set()
temp_count = [0, 0, 0, 0, 0]
for each in ann:
    if (each['category_id'] == categories[0]) and (temp_count[0] < 2):
        val_id.add(each['image_id'])
        temp_count[0] += 1
    elif (each['category_id'] == categories[1]) and (temp_count[1] < 20):
        val_id.add(each['image_id'])
        temp_count[1] += 1
    elif (each['category_id'] == categories[2]) and (temp_count[2] < 4):
        val_id.add(each['image_id'])
        temp_count[2] += 1
    elif (each['category_id'] == categories[3]) and (temp_count[3] < 7):
        val_id.add(each['image_id'])
        temp_count[3] += 1
    elif (each['category_id'] == categories[4]) and (temp_count[4] < 17):
        val_id.add(each['image_id'])
        temp_count[4] += 1

val_ann = []
for imid in val_id:
    for each_ann in ann:
        if each_ann['image_id'] == imid:
            val_ann.append(each_ann)

print(len(val_id), len(val_ann))

print('val set:')
for kind in categories:
    num = 0
    for i in val_ann:
        if i['category_id'] == kind:
            num += 1
    print(f'id: {kind} counts: {num}')

train_id = total_id - val_id
train_ann = []
for each_ann in ann:
    for tid in train_id:
        if each_ann['image_id'] == tid:
            train_ann.append(each_ann)
            break
print(len(train_id), len(train_ann))

# 创建文件夹 生成符合yolo的数据结构
train_img_path = './data/images/train/'
val_img_path = './data/images/val/'

os.makedirs(train_img_path, exist_ok=True)
os.makedirs(val_img_path, exist_ok=True)
orginal_image_path = './images/'  # 原始图片位置

train_img = []
# Move train images
for j in data['images']:
    for i in train_id:
        if j['id'] == i:
            file_name = orginal_image_path + j['file_name']
            shutil.copy(file_name, train_img_path)
            train_img.append(j)

val_img = []
# Move val images
for j in data['images']:
    for i in val_id:
        if j['id'] == i:
            file_name = orginal_image_path + j['file_name']
            shutil.copy(file_name, val_img_path)
            val_img.append(j)

print(len(val_img), len(train_img))

# 生成训练集测试集label文件
train_label_file_path = './data/labels/train/'
val_label_file_path = './data/labels/val/'
os.makedirs(train_label_file_path, exist_ok=True)
os.makedirs(val_label_file_path, exist_ok=True)

train_info = [(each['id'], each['file_name'].split('.')[0], each['width'], each['height']) for each in train_img]
val_info = [(each['id'], each['file_name'].split('.')[0], each['width'], each['height']) for each in val_img]
trans = {f'{each}': f'{idx}' for (idx, each) in enumerate(categories)}  # Mapping the category_ids

# 创建训练时需要的标签文件
for (imid, fn, w, h) in train_info:
    with open(train_label_file_path + fn + '.txt', 'w') as t_f:
        for t_ann in train_ann:
            if t_ann['image_id'] == imid:
                # 转换 X_min,Y_min,w,h 到 X_center/width,Y_center/height,w/width,h/height
                bbox = [str((t_ann['bbox'][0] + (t_ann['bbox'][2] / 2) - 1) / float(w)) + ' ',
                        str((t_ann['bbox'][1] + (t_ann['bbox'][3] / 2) - 1) / float(h)) + ' ',
                        str(t_ann['bbox'][2] / float(w)) + ' ',
                        str(t_ann['bbox'][3] / float(h))]
                t_f.write(trans[str(t_ann['category_id'])] + ' ' + str(bbox[0] + bbox[1] + bbox[2] + bbox[3]))
                t_f.write('\n')

# 创建评估时需要的标签文件
for (imid, fn, w, h) in val_info:
    with open(val_label_file_path + fn + '.txt', 'w') as v_f:
        for v_ann in val_ann:
            if v_ann['image_id'] == imid:
                # convert X_min,Y_min,w,h to X_center/width,Ycenter/height,w/width,h/height
                bbox = [str((v_ann['bbox'][0] + (v_ann['bbox'][2] / 2) - 1) / float(w)) + ' ',
                        str((v_ann['bbox'][1] + (v_ann['bbox'][3] / 2) - 1) / float(h)) + ' ',
                        str(v_ann['bbox'][2] / float(w)) + ' ',
                        str(v_ann['bbox'][3] / float(h))]
                v_f.write(trans[str(v_ann['category_id'])] + ' ' + str(bbox[0] + bbox[1] + bbox[2] + bbox[3]))
                v_f.write('\n')

#复制测试集图片
test_img_path = './data/images/test/'
os.makedirs(test_img_path, exist_ok=True)

os.makedirs(test_img_path, exist_ok=True)
test_data_file_path = './exc_data/test.csv'
test_data = pd.read_csv(test_data_file_path)
for each in test_data['file_name']:
    file_name = orginal_image_path + each
    shutil.copy(file_name, test_img_path)