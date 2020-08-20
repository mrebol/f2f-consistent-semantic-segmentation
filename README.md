# Frame-To-Frame Consistent Semantic Segmentation

This code runs the method described in the paper [Frame-To-Frame Consistent Semantic Segmentation](https://arxiv.org/abs/2008.00948):

    @InProceedings{Rebol_2020_ACVRW,
      author = {Rebol, Manuel and Knöbelreiter, Patrick},
      title = {Frame-To-Frame Consistent Semantic Segmentation},
      booktitle = {Joint Austrian Computer Vision And Robotics Workshop (ACVRW)},
      month = {April},
      year = {2020}
    } 
    
![ESPNet vs our model](https://github.com/mrebol/f2fcss/blob/master/esp_vs_our_model.gif)
*ESPNet vs Our Model ESPNet_L1b*

## Dependencies
+ CUDA 11.0 for execution on GPU (optional)
+ Python 3.7
+ PyTorch 1.3.1 
+ pip packages in `requirements.txt`


## Dataset
We provide a data loader for the Cityscapes dataset in the `data` directory. 
Additionally, we included frames of the Cityscapes Demo Video in the repository to support quick first experiments.
Images placed inside the `data/cityscapes_video/leftImg8bit/val` directory do require corresponding ground truth files, whereas images in the `data/cityscapes_video/leftImg8bit/test` directory don't. 

## Configuration
The default configuration file is stored in `config/eval.yml`. 
If the GPU option is enabled, CUDA 11.0 needs to be installed additionally.

## Evaluation

The required python packages need to be installed using the provided `requirements.txt`:  

    pip install -r requirements.txt
    
To run evaulation with the default config file located at `config/eval.yml` enter:

    python eval.py --config config/eval.yml 

## Results
The results consist of the predicted semantic segmentation images and the Tensorboard logs. Both are stored in the `output` folder. The Tensorboard logs can be examined after installing [Tensorboard](https://www.tensorflow.org/tensorboard) with the command:

    tensorboard --logdir output 


