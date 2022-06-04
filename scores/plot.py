import matplotlib.pyplot as plt
import torch
import numpy as np

if __name__ == "__main__":
    BaseNet = torch.load('./targeted/BaseNet.pth')
    eBaseNet = torch.load('./targeted/eBaseNet.pth')
    BaseNet_8 = torch.load('./targeted/BaseNet-8.pth')
    eBaseNet_10 = torch.load('./targeted/eBaseNet-10.pth')
    uBaseNet = torch.load('./untargeted/BaseNet.pth')
    ueBaseNet = torch.load('./untargeted/eBaseNet.pth')
    uBaseNet_8 = torch.load('./untargeted/BaseNet-8.pth')
    ueBaseNet_10 = torch.load('./untargeted/eBaseNet-10.pth')
    data = {'BaseNet': BaseNet, 'eBaseNet': eBaseNet, 'BaseNet-8': BaseNet_8, 'eBaseNet-10': eBaseNet_10,
            'uBaseNet': uBaseNet, 'ueBaseNet': ueBaseNet, 'uBaseNet-8': uBaseNet_8, 'ueBaseNet-10': ueBaseNet_10}

    # plt.figure(figsize=(12, 8), dpi=200)
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16), dpi=80)

    axes[0][0].plot(data['BaseNet']['E'][:29], data['BaseNet']['R'][:29], label='BaseNet', c="blue", linestyle='--')
    axes[0][0].plot(data['eBaseNet']['E'][:29], data['eBaseNet']['R'][:29], label='eBaseNet', c="red", linestyle='--')
    axes[0][0].plot(data['BaseNet-8']['E'][:29], data['BaseNet-8']['R'][:29], label='BaseNet-8', c="blue", linestyle='-')
    axes[0][0].plot(data['eBaseNet-10']['E'][:29], data['eBaseNet-10']['R'][:29], label='eBaseNet-10', c="red", linestyle='-')

    axes[0][1].plot(data['BaseNet']['E'][:29], data['BaseNet']['P'][:29], label='BaseNet', c="blue", linestyle='--')
    axes[0][1].plot(data['eBaseNet']['E'][:29], data['eBaseNet']['P'][:29], label='eBaseNet', c="red", linestyle='--')
    axes[0][1].plot(data['BaseNet-8']['E'][:29], data['BaseNet-8']['P'][:29], label='BaseNet-8', c="blue", linestyle='-')
    axes[0][1].plot(data['eBaseNet-10']['E'][:29], data['eBaseNet-10']['P'][:29], label='eBaseNet-10', c="red", linestyle='-')

    axes[1][0].plot(data['uBaseNet']['E'][:29], data['uBaseNet']['R'][:29], label='BaseNet', c="blue", linestyle='--')
    axes[1][0].plot(data['ueBaseNet']['E'][:29], data['ueBaseNet']['R'][:29], label='eBaseNet', c="red", linestyle='--')
    axes[1][0].plot(data['uBaseNet-8']['E'][:29], data['uBaseNet-8']['R'][:29], label='BaseNet-8', c="blue", linestyle='-')
    axes[1][0].plot(data['ueBaseNet-10']['E'][:29], data['ueBaseNet-10']['R'][:29], label='eBaseNet-10', c="red", linestyle='-')

    axes[1][1].plot(data['uBaseNet']['E'][:29], data['uBaseNet']['P'][:29], label='BaseNet', c="blue", linestyle='--')
    axes[1][1].plot(data['ueBaseNet']['E'][:29], data['ueBaseNet']['P'][:29], label='eBaseNet', c="red", linestyle='--')
    axes[1][1].plot(data['uBaseNet-8']['E'][:29], data['uBaseNet-8']['P'][:29], label='BaseNet-8', c="blue", linestyle='-')
    axes[1][1].plot(data['ueBaseNet-10']['E'][:29], data['ueBaseNet-10']['P'][:29], label='eBaseNet-10', c="red", linestyle='-')

    axes[0][0].set_xticks(list(np.arange(0, 3.1, 0.1)))
    axes[0][0].set_yticks(list(np.arange(0, 1.1, 0.05)))
    axes[0][1].set_xticks(list(np.arange(0, 3.1, 0.1)))
    axes[0][1].set_yticks(list(np.arange(0, 1.1, 0.05)))
    axes[1][0].set_xticks(list(np.arange(0, 3.1, 0.1)))
    axes[1][0].set_yticks(list(np.arange(0, 1.1, 0.05)))
    axes[1][1].set_xticks(list(np.arange(0, 3.1, 0.1)))
    axes[1][1].set_yticks(list(np.arange(0, 1.1, 0.05)))

    axes[0][0].set_xlabel('epsilon', fontsize=12)
    axes[0][0].set_ylabel('remaining', fontsize=12)
    axes[0][1].set_xlabel('epsilon', fontsize=12)
    axes[0][1].set_ylabel('probability', fontsize=12)
    axes[1][0].set_xlabel('epsilon', fontsize=12)
    axes[1][0].set_ylabel('remaining', fontsize=12)
    axes[1][1].set_xlabel('epsilon', fontsize=12)
    axes[1][1].set_ylabel('probability', fontsize=12)

    axes[0][0].set_title('Targeted attack on CIFAR10', fontsize=15)
    axes[0][1].set_title('Targeted attack on CIFAR10', fontsize=15)
    axes[1][0].set_title('Untargeted attack on CIFAR10', fontsize=15)
    axes[1][1].set_title('Untargeted attack on CIFAR10', fontsize=15)

    axes[0][0].grid(True, linestyle='--', alpha=0.5)
    axes[0][1].grid(True, linestyle='--', alpha=0.5)
    axes[1][0].grid(True, linestyle='--', alpha=0.5)
    axes[1][1].grid(True, linestyle='--', alpha=0.5)

    axes[0][0].legend()
    axes[0][1].legend()
    axes[1][0].legend()
    axes[1][1].legend()

    plt.savefig('./images.png')
    plt.show()
