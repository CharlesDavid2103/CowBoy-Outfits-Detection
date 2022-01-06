import yaml

# 生成yolov5数据yaml文件
data_yaml = dict(
    path='./data',  # dataset root dir
    train='images/train',  # train images (relative to 'path')
    val='images/val',  # val images (relative to 'path')
    test='images/test',  # test images (relative to 'path') 推理的时候才用得到
    nc=5,
    names=['belt', 'sunglasses', 'boot', 'cowboy_hat', 'jacket'],
    download='None'
)
with open('./data/cowboy_data_config.yaml', 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

# 生成yolov5模型参数初始化yaml文件
hyp_yaml = dict(
    lr0=0.01,  # initial learning rate (SGD=1E-2, Adam=1E-3)
    lrf=0.16,  # final OneCycleLR learning rate (lr0 * lrf)
    momentum=0.937,  # SGD momentum/Adam beta1
    weight_decay=0.0005,  # optimizer weight decay 5e-4
    warmup_epochs=5.0,  # warmup epochs (fractions ok)
    warmup_momentum=0.8,  # warmup initial momentum
    warmup_bias_lr=0.1,  # warmup initial bias lr
    box=0.05,  # box loss gain
    cls=0.3,  # cls loss gain
    cls_pw=1.0,  # cls BCELoss positive_weight
    obj=0.7,  # obj loss gain (scale with pixels)
    obj_pw=1.0,  # obj BCELoss positive_weight
    iou_t=0.20,  # IoU training threshold
    anchor_t=4.0,  # anchor-multiple threshold
    fl_gamma=0.0,  # focal loss gamma (efficientDet default gamma=1.5)
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    degrees=0.0,  # image rotation (+/- deg)
    translate=0.1,  # image translation (+/- fraction)
    scale=0.25,  # image scale (+/- gain)
    shear=0.0,  # image shear (+/- deg)
    perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
    flipud=0.0,  # image flip up-down (probability)
    fliplr=0.5,  # image flip left-right (probability)
    mosaic=1.0,  # image mosaic (probability)
    mixup=0.0,  # image mixup (probability)
    copy_paste=0.0  # segment copy-paste (probability)
)
with open('./data/hyps/cowboy_hyp_config.yaml', 'w') as f:
    yaml.dump(hyp_yaml, f, default_flow_style=False)

# 生成模型yaml文件
model_yaml = dict(
    nc=5,  # number of classes
    depth_multiple=1.0,  # model depth multiple
    width_multiple=1.0,  # layer channel multiple
    anchors=3,  # 这里把默认的anchors配置改成了3以启用autoanchor, 获取针对自己训练时的img_size的更优质的anchor size

    # YOLOv5 backbone
    backbone=
    # [from, number, module, args]
    [[-1, 1, 'Focus', [64, 3]],  # 0-P1/2
     [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
     [-1, 3, 'C3', [128]],
     [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
     [-1, 9, 'C3', [256]],
     [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
     [-1, 9, 'C3TR', [512]],  # <-------- C3TR() Transformer module
     [-1, 1, 'Conv', [768, 3, 2]],  # 7-P5/32
     [-1, 3, 'C3', [768]],
     [-1, 1, 'Conv', [1024, 3, 2]],  # 9-P6/64
     [-1, 1, 'SPP', [1024, [3, 5, 7]]],
     [-1, 3, 'C3TR', [1024, 'False']],  # 11  <-------- C3TR() Transformer module
     ],

    # YOLOv5 head
    head=
    [[-1, 1, 'Conv', [768, 1, 1]],
     [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
     [[-1, 8], 1, 'Concat', [1]],  # cat backbone P5
     [-1, 3, 'C3', [768, 'False']],  # 15

     [-1, 1, 'Conv', [512, 1, 1]],
     [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
     [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
     [-1, 3, 'C3', [512, 'False']],  # 19

     [-1, 1, 'Conv', [256, 1, 1]],
     [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
     [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
     [-1, 3, 'C3', [256, 'False']],  # 23 (P3/8-small)

     [-1, 1, 'Conv', [256, 3, 2]],
     [[-1, 20], 1, 'Concat', [1]],  # cat head P4
     [-1, 3, 'C3', [512, 'False']],  # 26 (P4/16-medium)

     [-1, 1, 'Conv', [512, 3, 2]],
     [[-1, 16], 1, 'Concat', [1]],  # cat head P5
     [-1, 3, 'C3', [768, 'False']],  # 29 (P5/32-large)

     [-1, 1, 'Conv', [768, 3, 2]],
     [[-1, 12], 1, 'Concat', [1]],  # cat head P6
     [-1, 3, 'C3', [1024, 'False']],  # 32 (P6/64-xlarge)

     [[23, 26, 29, 32], 1, 'Detect', ['nc', 'anchors']],  # Detect(P3, P4, P5, P6)
     ],
)
with open('./models/cowboy_yolov5l6-transformer.yaml', 'w') as f:
    yaml.dump(model_yaml, f, default_flow_style=True)
