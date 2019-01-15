# Voice Converter CycleGAN

https://github.com/leimao/Voice_Converter_CycleGAN



## Usage


### Dataset download
Download and unzip [VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211) dataset.

### Dataset preprocessing
you can use the dataset without any preprocessing.

### Training
you can train the model by running train_mydata.ipynb cells.
When you train the model, you can change the epoch num by changing the value of "num_epochs" variavble. this variable is on line 5 in 6th cell.

And if you want to continue training, you have to rewrite the number of "num_epochs" and "pre_epoch" variable. "pre_epoch" variable is on line 6 in 6th cell.
(for example, you have already trained 4000 epoch and want to continue training more 4000 epoch, then you rewrite pre_epoch from 0 to 4000 and eoch from 4000 to 8000.)

And you can change the MFCC numbers by changing the "num_mcep" variable. This variable is on line 13 in 6th cell.
If the OOM Error is occured. you have to change the frame size to more smaller number. This "n_frame" variable is on line 15 in 6th cell.

### Evaluation
you can test the trained model by running conv_test.ipynb file.
you have to edit the 2th cell variables.


## Extra

### How to do multi-gpu in keras

To make the mode parallel on multiple GPU in keras you have to include the following line: 

```
model = multi_gpu_model(model, gpus=G)
```
