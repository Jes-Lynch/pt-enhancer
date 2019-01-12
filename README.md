# Superresolution using an efficient sub-pixel convolutional neural network

This trains a super-resolution network on images of WSI tissue samples using crops of high resolution and low resolution WSI camera samples. A snapshot of the model after every epoch is saved with filename model_epoch_<epoch_number>.pth


```
usage: main.py [-h] --upscale_factor UPSCALE_FACTOR [--batchSize BATCHSIZE]
               [--testBatchSize TESTBATCHSIZE] [--nEpochs NEPOCHS] [--lr LR]
               [--cuda] [--threads THREADS] [--seed SEED] [--train]

PyTorch Super Res Example

optional arguments:
  -h, --help            show this help message and exit
  --upscale_factor      super resolution upscale factor
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
`%root%\dataset\%subset%\images\(highres or lowres)`
`Example: pt_enhancer\dataset\kidney\images\highres\`

### Train
`--upscale_factor 4 --batchSize 10 --nEpochs 1000 --lr 0.0001 --threads 5 --cuda --train`

### Test
`--upscale_factor 4 --testBatchSize 1 --nEpochs 1000 --threads 5 --cuda`

### Super Resolve
`python super_resolve.py --input_dir dataset/<type>/images/lowres --ref_dir dataset/<type>/images/highres/ --model model_epoch_100.pth --output_dir results --cuda`

## Sample
Kidney Tissue<br>
![orig](https://github.com/Jes-Lynch/pt_enhancer/blob/master/example/example.png)<br>
