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
+ CUDA 11.0
+ Python 3.7
+ PyTorch 1.3.1 
+ pip packages in `requirements.txt`


## Dataset


## Config


## Evaluation

The required python packages need to be installed using the provided `requirements.txt`:  

    pip install -r requirements.txt
    
To run evaulation with the default config file located in `config/eval.yml` enter:

    python eval.py  

## Results
The results which consists of predicted semantic segmentation images and Tensorboard logging 
are stored in the `output` folder. The Tensorboard logs can be examined after installing [Tensorboard](https://www.tensorflow.org/tensorboard) with the command:

    tensorboard --logdir output 


