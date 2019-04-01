# Superresolution using an efficient sub-pixel convolutional neural network

This trains a super-resolution network on images of WSI tissue samples using crops of high resolution and low resolution WSI camera samples. A snapshot of the model after every epoch is saved with filename model_epoch_<epoch_number>.pth


```
usage: main.py [-h] [--upscale_factor UPSCALE_FACTOR] [--full_size FULL_SZIE] [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED] [--train]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
  --full_size           size of high resolution images
  --batchSize           training batch size
  --testBatchSize       testing batch size
  --nEpochs             number of epochs to train for
  --lr                  Learning Rate. Default=0.01
  --cuda                use cuda
  --threads             number of threads for data loader to use Default=4
  --seed                random seed to use. Default=123
  --train               run training
```

## Example Usage:

### Data Structure
`%root%\dataset\%subset%\(train or test)\(lowres, int1, int2, highres)`
`Example: pt_enhancer\dataset\kidney\train\highres\`

### Train
`python main.py --upscale_factor 8 --full_size 1024 --batchSize 5 --nEpochs 1000 --lr 0.001 --threads 6 --cuda --train`

### Test
`python main.py --upscale_factor 8 --full_size 1024 --testBatchSize 1 --nEpochs 1000 --threads 4 --cuda`

### Super Resolve
`python super_resolve.py --upscale_factor 8 --full_size 1024 --model model_epoch_1000.pth --output_dir rrcnn_results/ --threads 6 --cuda`

## Sample
Kidney Tissue<br>
![orig](https://github.com/Jes-Lynch/pt_enhancer/blob/master/example/example.png)<br>
