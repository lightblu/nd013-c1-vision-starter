# Project Writeup

## Project overview: 

*(This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self-driving car systems?)*

### Introduction: For what do we need object detection?

Object detection is a critical component of self-driving car systems because it allows the car to detect and track objects in its environment, such as pedestrians, other vehicles, and obstacles. 
This information is necessary for the car to make real-time decisions about its driving behavior, such as when to accelerate, brake, or change lanes. Without reliable object detection, the car would be unable to operate safely and effectively in complex, dynamic traffic environments.

Object detection is even important for simpler control assist systems in non-self-driving cars that are already out on the streets today. 
For example, it is a critical component of advanced driver assistance systems (ADAS) such as collision warning systems, automatic emergency braking, and lane departure warning systems. 
These systems use sensors such as cameras, radar, and lidar to detect and track objects in the car's environment, which enables the car to take action to avoid or mitigate collisions. 
Object detection is also important for parking assist systems, which use sensors to detect objects and provide guidance to the driver during parking maneuvers. 
Overall, object detection is an essential technology for enhancing the safety and convenience of modern cars, both self-driving and non-self-driving.

Object detection is currently used in cars for a wide range of use cases, including:
1. Collision avoidance: Object detection is used to detect obstacles in the car's path and alert the driver or activate automatic braking to avoid collisions.
2. Blind spot monitoring: Object detection is used to monitor the car's blind spots and alert the driver if there is a vehicle in the adjacent lane.
3. Pedestrian detection: Object detection is used to detect pedestrians in the car's path and alert the driver or activate automatic braking to avoid collisions.
4. Lane departure warning: Object detection is used to monitor the car's position in its lane and alert the driver if the car begins to drift out of the lane.
5. Traffic sign recognition: Object detection is used to recognize traffic signs such as stop signs, speed limit signs, and no-entry signs, and provide alerts to the driver.
6. Parking assistance: Object detection is used to detect obstacles around the car during parking maneuvers and provide guidance to the driver.
7. Adaptive cruise control: Object detection is used to maintain a safe distance between the car and the vehicle in front by adjusting the car's speed.
8. Road sign detection: Object detection is used to detect relevant road signs and display that information like allowed speed to the user, or take decisions like giving right on its own. 


### Goal of this project

## Set up

*(This section should contain a brief description of the steps to follow to run the code for this repository.)*


### Basic setup

- Fork and then clone repository from https://github.com/lightblu/nd013-c1-vision-starter

- Beyond this not much is needed as due to lack of a proper GPU, the provided VM workspaces is used, which has most of the data.

- For step two download pretrained model (262M uncompressed in the end!)

    cd/home/workspace/experiments/pretrained_model/
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
    tar -xvzf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz
    rm -rf ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

### VM recurring setup

Due to those VMs crashing or being recreated after shutting down, the reguluar routine included:
  
  - Configure git:
  
    git config --global user.email "..."
    git config --global user.name "..."
    git config --global credential.helper 'store --file ~/.git_credentials'
  
  - Install style checker for pep8 conformity
  
    pip install pycodestyle seaborn
  
  - Install chromium
  
    sudo apt-get update && sudo apt-get install chromium-browser sudo chromium-browser --no-sandbox 


### Launch jupyter notebook


As this Firefox version often crashes with jupyter notebook the suggested workaround is to use Chrome. 
Use the adapted launch_jupyter.sh to have it start directly with chromium-browser:
  
   jupyter notebook --port 3002 --ip=0.0.0.0 --allow-root --browser="chromium-browser %s --no-sandbox"

For step one download data (not necessary because of prepared worksapce)

    python edit_config.py --train_dir ./data/train --eval_dir ./data/val --batch_size 2 --checkpoint ./experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map ./experiments/label_map.pbtxt --target_dir experiments/reference

## Dataset

### Dataset Analysis: 

*(This section should contain a quantitative and qualitative description of the dataset. It should include images, charts, and other visualizations.)*


### Cross-validation: 

*(This section should detail the cross-validation strategy and justify your approach.)*

## Training

    python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline.config

and to see results (go to python -m tensorboard.main --logdir experiments/reference/

    python -m tensorboard.main --logdir experiments/reference/
    
Evaluation:

    python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline.config --checkpoint_dir=experiments/reference/

### Reference experiment

*(This section should detail the results of the reference experiment. It should include training metrics, Tensorboard charts, and a detailed explanation of the algorithm's performance.)*

### Improve on the reference

*(This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.)*


#### experiment1 - try with other config... and generate from original dir

   93  python edit_config.py --train_dir /home/workspace/nd013-c1-vision-starter/data/train/ --eval_dir /home/workspace/data/nd013-c1-vision-starter/val/ --batch_size 2 --checkpoint /home/workspace/nd013-c1-vision-starter/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/nd013-c1-vision-starter/experiments/label_map.pbtxt
   96  mv pipeline_new.config nd013-c1-vision-starter/experiments/experiment1/pipeline.cfg
   
python experiments/model_main_tf2.py --model_dir=experiments/experiment1/ --pipeline_config_path=experiments/experiment1/pipeline.config



-----


s) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    
INFO:tensorflow:Finished eval step 100
I0211 19:33:15.387050 139716821260032 model_lib_v2.py:939] Finished eval step 100
INFO:tensorflow:Performing evaluation on 198 images.
I0211 19:33:36.469685 139716821260032 coco_evaluation.py:293] Performing evaluation on 198 images.
creating index...
index created!
INFO:tensorflow:Loading and preparing annotation results...
I0211 19:33:36.476302 139716821260032 coco_tools.py:116] Loading and preparing annotation results...
INFO:tensorflow:DONE (t=0.02s)
I0211 19:33:36.494174 139716821260032 coco_tools.py:138] DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=31.17s).
Accumulating evaluation results...
DONE (t=0.32s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.097
INFO:tensorflow:Eval metrics at step 10000
I0211 19:34:08.024227 139716821260032 model_lib_v2.py:988] Eval metrics at step 10000
INFO:tensorflow:    + DetectionBoxes_Precision/mAP: 0.000033
I0211 19:34:08.032566 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP: 0.000033
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.50IOU: 0.000133
I0211 19:34:08.034201 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.50IOU: 0.000133
INFO:tensorflow:    + DetectionBoxes_Precision/mAP@.75IOU: 0.000007
I0211 19:34:08.035868 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP@.75IOU: 0.000007
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (small): 0.000000
I0211 19:34:08.037473 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (small): 0.000000
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (medium): 0.000000
I0211 19:34:08.038891 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (medium): 0.000000
INFO:tensorflow:    + DetectionBoxes_Precision/mAP (large): 0.000541
I0211 19:34:08.040392 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Precision/mAP (large): 0.000541
INFO:tensorflow:    + DetectionBoxes_Recall/AR@1: 0.000000
I0211 19:34:08.041853 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@1: 0.000000
INFO:tensorflow:    + DetectionBoxes_Recall/AR@10: 0.000050
I0211 19:34:08.043377 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@10: 0.000050
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100: 0.004821
I0211 19:34:08.044970 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100: 0.004821
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (small): 0.000000
I0211 19:34:08.046303 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (small): 0.000000
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (medium): 0.000000
I0211 19:34:08.047621 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (medium): 0.000000
INFO:tensorflow:    + DetectionBoxes_Recall/AR@100 (large): 0.096600
I0211 19:34:08.049200 139716821260032 model_lib_v2.py:991]  + DetectionBoxes_Recall/AR@100 (large): 0.096600
INFO:tensorflow:    + Loss/localization_loss: 1.095490
I0211 19:34:08.050574 139716821260032 model_lib_v2.py:991]  + Loss/localization_loss: 1.095490
INFO:tensorflow:    + Loss/classification_loss: 0.815495
I0211 19:34:08.052059 139716821260032 model_lib_v2.py:991]  + Loss/classification_loss: 0.815495
INFO:tensorflow:    + Loss/regularization_loss: 107.768509
I0211 19:34:08.053437 139716821260032 model_lib_v2.py:991]  + Loss/regularization_loss: 107.768509
INFO:tensorflow:    + Loss/total_loss: 109.679489
I0211 19:34:08.054764 139716821260032 model_lib_v2.py:991]  + Loss/total_loss: 109.679489
INFO:tensorflow:Waiting for new checkpoint at experiments/reference/
I0211 19:37:11.735399 139716821260032 checkpoint_utils.py:125] Waiting for new checkpoint at experiments/reference/
INFO:tensorflow:Timed-out waiting for a checkpoint.
I0211 19:38:10.927647 139716821260032 checkpoint_utils.py:188] Timed-out waiting for a checkpoint.


experiment1 catastrophically the same 

experiment2 fiddling, still bad paths?? 


python experiments/model_main_tf2.py --model_dir=experiments/experiment2/ --pipeline_config_path=experiments/experiment2/pipeline.config
python -m tensorboard.main --logdir experiments/experiment2/
 
 