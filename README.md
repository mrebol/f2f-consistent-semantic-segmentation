# Frame-To-Frame Consistent Semantic Segmentation

This code implements the method introduced in the publication [Frame-To-Frame Consistent Semantic Segmentation](https://arxiv.org/abs/2008.00948):

    @InProceedings{Rebol_2020_ACVRW,
      author = {Rebol, Manuel and Knöbelreiter, Patrick},
      title = {Frame-To-Frame Consistent Semantic Segmentation},
      booktitle = {Joint Austrian Computer Vision And Robotics Workshop (ACVRW)},
      month = {April},
      year = {2020}
    } 
    
![ESPNet vs our model](https://github.com/mrebol/f2fcss/blob/master/resources/esp_vs_our_model.gif)
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
It loads the pretrained ESPNet_L1b model and inputs the dataset provided.
If the GPU parameter in the config is enabled, CUDA 11.0 needs to be installed additionally.


## Evaluation

The required python packages need to be installed using the provided `requirements.txt`:  

    pip install -r requirements.txt
    
To run the evaluation with the default config file located at `config/eval.yml` enter:

    python eval.py --config config/eval.yml 

## Results
The predicted semantic segmentation images are saved in the `output/<timestamp>/images/` folder.
Depending on the input config, the folder contains the semantic color maps and/or the semantic label ids.


Additionally, we generate Tensorboard logs at `output/<timestamp>/tensorboard/`. These logs can be 
examined after installing Tensorboard
    
    pip install tensorboard 
with the command:

    tensorboard --logdir output 

Tensorboard visualizes the statistics at [http://localhost:6006/](http://localhost:6006/) by default.

