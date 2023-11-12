## Train
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python projects/IDOL/train_net.py --config-file projects/IDOL/configs/r50_train.yaml --num-gpus 8```
```

## Test
First, generate a json file containing the tracking results for each video frame in the validation set.

```
CUDA_VISIBLE_DEVICES=0 python projects/IDOL/train_net.py --config-file projects/IDOL/configs/coco_pretrain/r50_eval.yaml --num-gpus 1 --eval-only
```
Second, merge json to generate test.json and put it under {TrackEval/data/pse/idol/score}.
```
python truncode.py
```
Third, verify the results of the json file on unknown classes and known classes respectively.
```
python scripts/run_tao_ow.py --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL online-ghost
python scripts/run_tao_ow.py --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL idol --SUBSET known
```

## Model Zoo
We provide the result file, which can be downloaded by the following link.


|         Method          | OWTA(Unknown) |  OWTA(Known)  | Checkpoint |
|-------------------------|---------|---------|------------|
|         SimOWT          |  50.4   |  62.9   |  [Checkpoint](https://drive.google.com/file/d/1x7Xh_EKVkt7aAz3ioWpzUN-FxryNWKQk/view?usp=drive_link)|
