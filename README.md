# underwater-industrial-application-yolov5m6
This project wins the intelligent algorithm contest finalist award and stands out from over 2000teams in China Underwater Robot Professional Contest, entering the final of China Underwater Robot Professional Contest and ranking 13 out of 31 teams in finals.

## [和鲸社区Kesci 水下光学目标检测产业应用赛项 ]( https://www.heywhale.com/home/competition/60cd6142b027c60017cdc98f/content/8 )
# 环境:
## mmdetection
	+ 操作系统：Ubuntu 18.04.2
	+ GPU：1块2080Ti
	+ Python：Python 3.7.7
	+ NVIDIA依赖：
	    - NVCC: Cuda compilation tools, release 10.1, V10.1.243
	    - CuDNN 7.6.5
	+ 深度学习框架：
	    - PyTorch: 1.8.1
	    - TorchVision: 0.9.1
	    - OpenCV
	    - MMCV
	    - MMDetection(注意data clean 的版本不同)
## yolov5 

	训练环境:
		+ 操作系统：Ubuntu 18.04.2
		+ GPU：1块2080Ti
		+ Python：Python 3.7.7
	测试环境:
		 NVIDIA Jetson AGX Xavier
	
	
	# pip install -r requirements.txt
	
	# base ----------------------------------------
	matplotlib>=3.2.2
	numpy>=1.18.5
	opencv-python>=4.1.2
	Pillow
	PyYAML>=5.3.1
	scipy>=1.4.1
	torch>=1.7.0
	torchvision>=0.8.1
	tqdm>=4.41.0

	# logging -------------------------------------
	tensorboard>=2.4.1
	# wandb

	# plotting ------------------------------------
	seaborn>=0.11.0
	pandas

	# export --------------------------------------
	# coremltools>=4.1
	# onnx>=1.9.0
	# scikit-learn==0.19.2  # for coreml quantization
	# tensorflow==2.4.1  # for TFLite export

	# extras --------------------------------------
	# Cython  # for pycocotools https://github.com/cocodataset/cocoapi/issues/172
	# pycocotools>=2.0  # COCO mAP
	# albumentations>=1.0.3
	thop  # FLOPs computation

	

# 第一大步：@数据清理
### 文件说明：data_clean_Code用于数据清理
 	data_clean_Code/yangtiming-underwater-master ->为湛江赛拿第20名方案
  	data_clean_Code/underwater-detection-master  ->为triks团队湛江赛方案
	
## 使用说明
	
### 1. （这一步用我的yangtiming-underwater-master替代原有的cascade_rcnn_x101_64x4d_fpn_dcn_e15 ）【原因精度更高A榜0.562】
   	模型采用 cascade_rcnn_x101_64x4d_fpn_dcn_e15  
    + Backbone:
        + ResNeXt101-64x4d
    + Neck:
        + FPN
    + DCN
    + Global context(GC)
    + MS [(4096, 600), (4096, 1000)]
    + RandomRotate90°
    + 15epochs + step:[11, 13]  
    + A榜：0.55040585 
        + 注：不是所有数据
  

### 2. 基于1训练好的模型对训练数据进行清洗(tools/data_process/data_clean.py)
    + 1. 如果某张图片上所有预测框的confidence没有一个是大于0.9， 那么去掉该图片(即看不清的图片)
    + 2. 修正错误标注
        + 1. 先过滤掉confidence<0.1的predict boxes, 然后同GT boxes求iou
        + 2. 如果predict box同GT的最大iou大于0.6，但类别不一致， 那么就修正该gt box的类别
    trainall.json （与bbox1）修后的到   trainall-revised.json

### 3. 基于2修正后的数据进行训练->（基于2修正后的到 trainall-revised.json 修正采用cascade_rcnn_r50_rfp_sac后的到-> bbox3
   	模型采用cascade_rcnn_r50_rfp_sac
    + Backbone:
	+ ResNet50
    + Neck:
	RFP-SAC
    + GC + MS + RandomRotate90°
    + cascade_iou调整为：（0.55， 0.65， 0.75）
    + A榜： 0.56339531
	+ 注：所有数据
	
### 4. 基于3训练好的模型进一步清洗数据.  
    ->  trainall-revised-v3.json（与bbox3） 	进一步清洗数据 -> trainall-revised-v4.json)
    + 模型同3： 
    + A榜：0.56945031
        + 注：所有数据





##### 在验证集上面测试精度：1. 执行完mAP0.5:0.95=0.547 4. 执行完mAP0.5:0.95 = 0.560
        


# 第二大步：@数据清理完毕后，改用yolov5 （code/yolov5_V5_chuli_focal_loss_attention）
	使用背景介绍：
	使用模型为yolov5m6系列，迭代tricks的时候，采取用--img 640 进行迭代


### 最优模型：
	最终在yolov5m6上面的精度为：mAP0.5:0.95= 0.5374，agx速度0.2s每张
	最好模型：
	1.yolov5m6
	2.数据清洗
	2.attention模块：senet
	3.hsv_h,hsv_s,hsv_v=0
	4.focal_loss



### 使用的tricks如下：（按照时间顺序）

	1.按照第一大步的数据清洗：由原来的mAP0.5:0.95= 0.465->0.4880
	2.yolov5当中的hsv_h,hsv_s,hsv_v均设为0，mAP0.5:0.95= 0.4880 ->0.4940
	3.在loss.py当中加入focal_loss损失函数(157行，172行)，mAP0.5:0.95= 0.4940 ->0.4977
	4.更改原有yolov5的c3层改为senet(attention模块)，mAP0.5:0.95= 0.4977 -> 0.50069
	

以上按照
	
	python train.py  --weights weights/yolov5m6.pt --cfg models/hub/yolov5m6-senet.yaml --data data/underwater.yaml  --hyp data/hyps/hyp.scratch-p6.yaml --epochs 100 --batch-size 25 --img 640

最终要提交的时候，按照
	
	python train.py  --weights weights/yolov5m6.pt --cfg models/hub/yolov5m6-senet.yaml --data data/underwater.yaml  --hyp data/hyps/hyp.scratch-p6.yaml --epochs 250 --batch-size 4 --img 1280 --multi-scale

【注意：multi-scale大小可以在train.py文件夹下面更改】



测试 

	python3 val_tm_txt_csv.py --data  /data/underwater.yaml   --weights weights/best_05374.pt --img 1280 --save-txt --save-conf --half

【--half能提升速度（fp16），精度比fp32更高】

################

若要测试mAP，可以用
https://github.com/rafaelpadilla/review_object_detection_metrics.git

## 以下为比赛文档说明

	若有权限问题，则执行前 chmod +x main_test.sh

	1. 将测试集的图片放在本文件夹当中名字为test
	2.最终结果将放在answer当中(执行后自动生成)
	3.code文件夹为全部代码，以及包含训练权重
	4.执行main_test.sh开始运行



	(*)Q：何时开始计时？(注意：在执行main_test.sh之前命令窗口拉长，否则计时无法看到进度条)
	当执行 python3 ./val_tm_txt_csv.py 时，看见如下界面表示计时开始
	##                 Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95:   0%|          | 0/xxx [00:00<?, ?it/s]


	(*)Q：何时结束计时？
	当执行 python3 ./val_tm_txt_csv.py 完毕后即可停止计时

# reference
* [yolov5 ]( https://github.com/ultralytics/yolov5 )
* [yangtiming/underwater-mmdetection ]( https://github.com/yangtiming/underwater-mmdetection )
* team-tricks	




