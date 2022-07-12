# Adversarial-Attack

## MNIST

### Models

- BaseModel: a simple CNN used on MNIST ACC: 98.32%
- eBaseModel: replace all relu() with square()   ACC: 98.21%

To train model and save:

```bash
python models/train.py --dataset MNIST --epochs 20 --lr 0.001 --model BaseNet --save_dir ./checkpoint/MNIST/BaseNet.pth
```

### Compressed

Same as CIFAR10, we can compress both model with 8bits:

```bash
python ./compressed/compress_model.py --dataset MNIST --model BaseNet --model_dir ./checkpoint/MNIST/BaseNet.pth --dataset_dir ./data --save_dir ./checkpoint/MNIST/BaseNet-8.pth --act_bits 8 --weight_bits 8
```

- BaseModel-8:   ACC: 97.59%
- eBaseModel-8:  ACC: 98.11%

### SimBA

Same as the CIFAR10, run the script **run_simba_mnist.py**

```bash
python attack/run_simba_mnist.py --targeted --model BaseNet --model_ckpt ./checkpoint/CIFAR10/BaseNet.pth --epsilon 0.2 
```

![effect-epsilon](https://github.com/quliikay/Adversarial-Attack/blob/main/scores/MNIST/images.png?raw=true)

## CIFAR10

### Models

- BaseModel: a simple CNN used on CIFAR10 ACC: 81.15%
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

**I find that after compression, the attack efficiency against BaseModel or eBaseModel will be significantly reduced.
Only raising epsilon to 0.7 has a good effect. You can review this on the image below.**

![effect-epsilon](https://github.com/quliikay/Adversarial-Attack/blob/main/scores/CIFAR10/images.png?raw=true)

The experiments were divided into two groups: the first group was targeted attack and the other group was untargeted
attack.

In the targeted attack group, **remaining** represents the proportion of samples that the model did not predict as the
targeted label; in the untargeted attack group, **remaining** represents the proportion of samples that the model did
not predict incorrectly. No matter which group it is, the larger the **remaining**, the better the attack effect.

In the targeted attack group, **probability** represents the probability that the model predicts the sample as its
targeted label; in the untargeted attack group, **probability** represents the probability that the model predicts the
sample as its true label. In the targeted attack group, the larger the **probability**, the better the attack effect. In
the other group, the smaller the **probability**, the better the attack effect.

From these experiments, we can know the compression operation has some **defense effect** on attack? No matter in
BaseNet or eBaseNet.

***[Fix]***: In my last experiment, I found that when using the Square activation function, the attack only works when
epsilon>=5. But after my later inspection, it was due to a **bug** in the code: the parameters of normalize were not
unified in training and attack. After fixing it, it can be found that using Square activation function does not
significantly improve the model's ability to resist attacks. The attack still works when epsilon is less than 1, as same
as using the Relu.

### Defense methods

I've implemented three different methods based on adding gaussian noise and compared them. They are:

1. adding gaussian noise at output layer, which means directly add noise on the confidence of each class.
2. adding gaussian noise at input layer, which means add noise on images.
3. adding gaussian noise at input layer and use the model trained with various and optimal noise.

#### exp1: add gaussian noise at output layer

In the past researches, a very common method to defense attack is adding random gaussian noise on the input image. The
reason behind this is that doing so can mislead the results of each query, allowing the attacker to make the opposite
decision. But, why don't we add gaussian noise directly to the last layer, the confidence layer, which is cheaper and
more straightforward. Based on this idea, I did a series of related experiments with various sigmas.

If I set epsilon=1 then fix it,

- targeted attack
  ![img_1.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/exp1_targeted.png?raw=true)
- untargeted attack
  ![img_2.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/exp1_untargeted.png?raw=true)

Through these figures, we can know this method is an effective way when sigma is optimal. But when sigma turn to larger
than 0.2, the defense will no longer effective.

#### exp2: add gaussian noise at input layer

In this part, I implemented the method mentioned in
this [paper](https://proceedings.neurips.cc/paper/2021/file/3eb414bf1c2a66a09c185d60553417b8-Paper.pdf) and the results
will be presented in the next section.

#### exp3: add gaussian noise at output layer

Firstly, in this [paper](https://proceedings.neurips.cc/paper/2021/file/3eb414bf1c2a66a09c185d60553417b8-Paper.pdf), the
author mentioned an improved version of RND. Training the model with gaussian noise in order to mitigate the effect of
adding noise on the model on clean images. Because the larger sigma is, the better efficiency of defense is, but the
lower accuracy on clean images is.

However, there is another question: how to decide the sigma of gaussian noise added on the training dataset. Dr. Lou
mentioned a new idea to add random gaussian noise on it; each training epoch chooses one sigma at random. Intuitively,
the model training in this way can have a better performance than using a fixed sigma.

Run code `tensorboard --logidr="./attack/logs"` to see the influence of adding gaussian noise on images.

![gaussian influence.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/logs/gaussian%20image.png?raw=true)

We can see that when sigma is greater than 0.19, the image quality is seriously degraded, so the Gaussian noise I add in
training is less than 0.19.

The attack results are showed below.

- targeted attack
  ![img_3.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/targeted.png?raw=true)
- untargeted attack
  ![img_4.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/untargeted.png?raw=true)

The exp1, exp2 and exp3 are my three experiments with adding noise at different layers and different ways. And we can
see the performance of model3 is the best among them.

But at this time, I have a new question: why is the efficiency of model3 so good? If that's what the paper says, this
could improve the accuracy of the model on clean images. Then the result should be that with a larger sigma, a better
defense effect is achieved, and the acc drop is not obvious. But why does this model perform so well at the beginning,
even if I don't add any noise and sigma=0, it still performs well. My guess is that the act of adding Gaussian noise
simulates different attack behaviors of the attacker, so the model learns how to resist during training. Therefore, in
the actual reasoning process, I don't need to add any noise at all, and it can still resist attacks.

### More Analysis

In the last part, I set the epsilon=1.0 and fix it to test different sigma.

In this part, I'll set sigma=0.0 and sigma=0.009 then fix it, to see this model's defensive ability against different
epsilon.

#### exp1: add gaussian noise at output layer

- targeted attack
  ![exp1_targeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp1_targeted.png?raw=true)
- untargeted attack
  ![exp1_untargeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp1_untargeted.png?raw=true)

#### exp2: add gaussian noise at input layer

- targeted attack
  ![exp2_targeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp2_targeted.png?raw=true)
- untargeted attack
  ![exp2_untargeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp1_untargeted.png?raw=true)

#### exp3: add gaussian noise at input layer

- targeted attack
  ![exp3_targeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp3_targeted.png?raw=true)
- untargeted attack
  ![exp3_untargeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp3_untargeted.png?raw=true)

#### exp4: combine exp1 and exp3

- targeted attack
  ![exp4_targeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp4_targeted.png?raw=true)
- untargeted attack
  ![exp4_untargeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp4_untargeted.png?raw=true)

#### exp5: attack training

- targeted attack
  ![exp5_targeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp5_targeted.png?raw=true)

- untargeted attack
  ![exp5_untargeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/exp5_untargeted.png?raw=true)

#### all data

- targeted attack
  ![all_targeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/all_targeted.png?raw=true)
- untargeted attack
  ![all_untargeted.png](https://github.com/quliikay/Adversarial-Attack/blob/main/attack/image/fix%20sigma/all_untargeted.png?raw=true)

## Skin Cancer Dataset

