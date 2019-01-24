# LPR Documentation

## Introduction 

This is the project for the LPR detection and processing for the HK use case

## Project structure 

```
project
│   README.md   
└───api
│   │   cron.py
│   │   errors.log
│   │   ftp_server.py
│   │   info.log
│   │   logging.json
│   │   main.py
│   │   mqtt_client.py
│   │
└───mrcnn
│   │   model-content
│   │   
└───imgs
│   │   folder for images downloaded
│   │   ...
└───weights
│   │   mask_rcnn_plc_0535.h5
│   │   this_is_the_best_lp_weight_1203.h5
│   detect.py
```


API folder contains the mqtt_client, ftp_client, logging and starting script  
IMG folder contains images, that will be downloaded from server  
MRCNN folder contains the MASK-RCNN code  
WEIGHTS folder contains the model weights for car plate and numbers  
DETECT.py is inference script  
ERRORS.log all errors will be logged here  
INFO.log all info will be logged here  


## Get Started

The LPR requires a lot of libraries to run. Best way to use conda env to install all
libraries on it.

### Step 1

Activate environment 

`` `source activate your-enrironment` `` (our example: `` `source activate tensorflow` ``)

### Step 2

Go to the project folder: ~/your_path/../LP_detect_dep/api

Run the script:  `` `python main.py cron` ``

Remove images from remote server:  `` `python main.py remove` ``

Restart the script: Stop it and run again  `` `python main.py cron` ``


## Issues

To stop the scrip just use Ctrl+C

To check why the script is crashed, please check the logs





