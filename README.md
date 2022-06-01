# Adversarial-Attack

## Models
- BaseModel: a simple CNN used on CIFAR10
- eBaseModel: replace all relu() with square()
- MobileNetV2: based on original MobileNetV2 but the activation function is square()

To train model and save:
```
python train.py --epochs 300 --lr 0.001 --model eBaseNet --save_dir ./checkpoint/eBaseNet.pth
```

## Compressed
After training model, we can run this to compress the model
```
python compress_model.py --model_name BaseNet --model_dir ../checkpoint/BaseNet.pth --dataset_dir ../data/ --save_dir ../checkpoint/BaseNet-8.pth --act_bits 8 --weight_bits 8
```
This code operation will compress the BaseNet into 8bits.

And I've found the best bits for BaseNet and eBaseNet, respectively.
![BaseNet](https://github.com/quliikay/Adversarial-Attack/blob/main/compressed/image/BaseNet.png?raw=true)
![eBaseNet](https://github.com/quliikay/Adversarial-Attack/blob/main/compressed/image/eBaseNet.png?raw=true)

I tend to use 8bits for BaseNet and 10bits for eBaseNet, because I want to lower the Loss and higher the Acc.

**What confuses me is why Loss increases with the number of bits while Acc stays the same, after about 10bits**

## SimBA on CIFAR10
We can run this code to attack our models: eBaseNet, BaseNet, eBaseNet-10, BaseNet8.

1. BaseNet
```
python run_simba_cifar.py --targeted --model BaseNet --model_ckpt ./checkpoint/BaseNet.pth --epsilon 0.2
```
2. eBaseNet
```
python run_simba_cifar.py --targeted --model eBaseNet --model_ckpt ./checkpoint/eBaseNet.pth --epsilon 5
```
3. BaseNet-8
```
python run_simba_cifar.py --targeted --compress --model BaseNet --model_ckpt ./checkpoint/BaseNet-8.pth --epsilon 0.6
```
4. eBaseNet-10
```
python run_simba_cifar.py --targeted --compress --model eBaseNet --model_ckpt ./checkpoint/eBaseNet-10.pth --epsilon 15
```
**I find that after compression, the attack efficiency against BaseModel will be significantly reduced. Only raising epsilon to 0.6 has an effect. For eBaseNet, the result is even stranger, the attack is ineffective no matter how large the epsilon is.**