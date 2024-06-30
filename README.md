# RPNet Defense

### Models

- BaseModel.pth: use square activation function trained with dataset.

- eBaseModel-10.pth: use square activation function and compressed to 10 bits (for CIFAR10).

- eBaseModel-8.pth: use square activation function and compressed to 8 bits (for MNIST).

- the specific implementation is in models/CIFAR10.py or MNIST.py

- pth files are in checkpoint/CIFAR10/ and checkpoint/MNIST/

- compressed process is in ./compreessed

- after training model, you can run this to compress the model

  ```bash
  python ./compressed/compress_model.py --dataset CIFAR10 --model BaseNet --model_dir ./checkpoint/CIFAR10/BaseNet.pth --dataset_dir ./data --save_dir ./checkpoint/CIFAR10/eBaseNet-10.pth --act_bits 10 --weight_bits 10
  ```

  This code operation will compress the BaseNet into 10bits.

- dataset: cifar or mnist

- model type CIFAR10 or MNIST

### Defense

```bash
python ./attack/simba_dev.py --num_runs 128 --batch_size 128 --data_root ./data --dataset cifar --model_type CIFAR10 --image_size 32 --targeted --model_ckpt ./checkpoint/CIFAR10/RND/eBaseNet-10.pth --sigma1 0.1 --sigma2 0.05 --T 400 -beta_min 0.5 -beta_max 1.5 --epsilon 1
```

- num_runs: number of image samples

- batch_size: batch size for parallel runs

- data_root: the location of your dataset

- targeted: targeted attack or untargeted attack

- model_ckpt: pth files.

- sigma1: gaussian noise adding on input layer

- sigma2: gaussian noise adding on confidence layer

- T: the cycle of epsilon schedule

- beta_min, beta_max: beta_range[beta_min, beta_max]

- image_size: 28 for mnist and 32 for CIFAR10