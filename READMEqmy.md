
## 可视化卡车推理结果
```shell
1.1验证集可视化
python visualize_truck_val.py --display-conf 0.1
1.2训练集可视化
python visualize_truck_val.py --split train --display-conf 0.1

```

## 对比实验
```shell
1.PCDet
训练
python train_pcdnet.py --weights "/media/ddc/新加卷/hys/qmy/PZ_8/weight/PCDNet.pt" --data "/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml" --device 0 --batch 8 --imgsz 512 --epochs 200
saff作为侧边输出输入decoder
python train_pim_saff_pan.py \
  --data "/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml" \
  --device 0,1 \
  --batch 32 \
  --epochs 200 \
  --workers 8 \
  --name pim_saff_pan
python train_pim_saff_pan.py \
  --data "/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml" \
  --device 2 \
  --batch 16 \
  --epochs 200 \
  --workers 8 \
  --name pim_saff_pan_depth关mosaic
  --cache
验证
python eval_pcdnet.py \
  --weights "/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/pcdnet_s0_aolp_dolp/weights/best.pt" \
  --pcdnet-root "/media/ddc/新加卷/hys/compare/AAAI24-PCDNet" \
  --data "/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml" \
  --device 0 \
  --batch 8 \
  --imgsz 512 \
  --workers 0 \
  --name pcdnet


特征图可视化
python visualize_pim_saff.py \
  --weights "/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/91512/weights/best.pt" \
  --image "/media/ddc/新加卷/hys/qmy/pz/images/2000-01-01-10-53-57_75.png" \
  --device 0 \
  --imgsz 512 \
  --conf 0.5 \
  --output "/media/ddc/新加卷/hys/qmy/PZ_8/runs/feature_vis/one_image"
```

