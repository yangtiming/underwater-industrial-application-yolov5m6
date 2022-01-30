cd /root/URPC
rm -rf answer
rm -rf code/yolov5_V5_chuli_focal_loss_attention/runs/
rm -rf code/yolov5_V5_chuli_focal_loss_attention/results/
rm -rf code/yolov5_V5_chuli_focal_loss_attention/user_data/
mkdir answer
mkdir code/yolov5_V5_chuli_focal_loss_attention/results/

cd code/yolov5_V5_chuli_focal_loss_attention/tm_tools/
python3 generate_test_json.py
python3 COCO2YOLO.py  -j ./val.json  -o ../user_data/tmp_data/labels/val/
rm -rf val.json
cd ..
mkdir user_data/tmp_data/images
mkdir user_data/tmp_data/images/val
cp -r ../../test/ ./user_data/tmp_data/images/val
python3 val_tm_txt_csv.py --data  /data/underwater.yaml   --weights weights/best_0534.pt --img 1280 --save-txt --save-conf --half






