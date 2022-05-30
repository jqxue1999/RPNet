# Adversarial-Attack

## Models
- BaseModel: a simple CNN used on CIFAR10
- eBaseModel: replace all relu() with square()
- MobileNetV2: based on original MobileNetV2 but the activation function is square()
**Note:** I find deep CNN with square() is hard to train

To train model and save:
```
python train.py --epochs 300 --lr 0.001 --model eBaseNet --save_dir ./checkpoint/eBaseNet_1e3.pth
```

## Compressed
After training model, we can run this code to compress the model
```
python compress_model.py --model_name eBaseNet --model_dir ../checkpoint/eBaseNet.pth --dataset_dir ../data/ --save_dir ../checkpoint/eBaseNet-8.pth
```

## SimBA on CIFAR10
```
python run_simba_cifar.py --targeted --model BaseNet
```