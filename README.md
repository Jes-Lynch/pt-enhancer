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

### Train

`python main.py --upscale_factor 3 --batchSize 10 --testBatchSize 100 --nEpochs 100 --lr 0.001 --cuda --train`

### Test

`python main.py --upscale_factor 3 --batchSize 10 --testBatchSize 100 --nEpochs 100 --lr 0.001 --cuda`

### Super Resolve
`python super_resolve.py --input_dir dataset/<type>/images/lowres --ref_dir dataset/<type>/images/highres/ --model model_epoch_100.pth --output_dir results --cuda`
