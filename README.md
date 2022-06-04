# Adversarial-Attack

## CIFAR10

### Models
- BaseModel: a simple CNN used on CIFAR10    ACC: 81.15%
- eBaseModel: replace all relu() with square()   ACC: 77.27%

To train model and save:
```bash
python models/train.py --dataset CIFAR10 --epochs 100 --lr 0.001 --model eBaseNet --save_dir ./checkpoint/CIFAR10/eBaseNet.pth
```

### Compressed
After training model, we can run this to compress the model
```bash
python ./compressed/compress_model.py --dataset CIFAR10 --model BaseNet --model_dir ./checkpoint/CIFAR10/BaseNet.pth --dataset_dir ./data --save_dir ./checkpoint/CIFAR10/BaseNet-8.pth --act_bits 8 --weight_bits 8
```
This code operation will compress the BaseNet into 8bits.

And I've found the best bits for BaseNet and eBaseNet, respectively.
![BaseNet](https://github.com/quliikay/Adversarial-Attack/blob/main/compressed/image/BaseNet.png?raw=true)
![eBaseNet](https://github.com/quliikay/Adversarial-Attack/blob/main/compressed/image/eBaseNet.png?raw=true)

I tend to use 8bits for BaseNet and 10bits for eBaseNet, because I want to lower the Loss and higher the Acc.

**What confuses me is why Loss increases with the number of bits while Acc stays the same, after about 10bits.**

### SimBA

We can run this code to attack our models: eBaseNet, BaseNet, eBaseNet-10bit and BaseNet-8bit.

1. BaseNet
```bash
python attack/run_simba_cifar.py --targeted --model BaseNet --model_ckpt ./checkpoint/CIFAR10/BaseNet.pth --epsilon 0.2 
```
2. eBaseNet
```bash
python attack/run_simba_cifar.py --targeted --model eBaseNet --model_ckpt ./checkpoint/CIFAR10/eBaseNet.pth --epsilon 0.2
```
3. BaseNet-8
```bash
python attack/run_simba_cifar.py --targeted --compress --model BaseNet --model_ckpt ./checkpoint/BaseNet-8.pth --epsilon 0.7
```
4. eBaseNet-10
```bash
python attack/run_simba_cifar.py --targeted --compress --model eBaseNet --model_ckpt ./checkpoint/eBaseNet-10.pth --epsilon 0.7
```
**I find that after compression, the attack efficiency against BaseModel or eBaseModel will be significantly reduced. Only raising epsilon to 0.7 has a good effect. You can review this on the image below.**

![effect-epsilon](https://github.com/quliikay/Adversarial-Attack/blob/main/scores/images.png?raw=true)

The experiments were divided into two groups: the first group was targeted attack and the other group was untargeted attack.

In the targeted attack group, **remaining** represents the proportion of samples that the model did not predict as the targeted label; in the untargeted attack group, **remaining** represents the proportion of samples that the model did not predict incorrectly. No matter which group it is, the larger the **remaining**, the better the attack effect.

In the targeted attack group, **probability** represents the probability that the model predicts the sample as its targeted label; in the untargeted attack group, **probability** represents the probability that the model predicts the sample as its true label. In the targeted attack group, the larger the **probability**, the better the attack effect. In the other group, the smaller the **probability**, the better the attack effect.

From these experiments, we can know the compression operation has some **defense effect** on attack? No matter in BaseNet or eBaseNet.



