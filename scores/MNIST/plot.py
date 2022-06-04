import matplotlib.pyplot as plt
import torch
import numpy as np

if __name__ == "__main__":
    BaseNet = torch.load('./targeted/BaseNet.pth')
    eBaseNet = torch.load('./targeted/eBaseNet.pth')
    BaseNet_8 = torch.load('./targeted/BaseNet-8.pth')
    eBaseNet_8 = torch.load('./targeted/eBaseNet-8.pth')
    uBaseNet = torch.load('./untargeted/BaseNet.pth')
    ueBaseNet = torch.load('./untargeted/eBaseNet.pth')
    uBaseNet_8 = torch.load('./untargeted/BaseNet-8.pth')
    ueBaseNet_8 = torch.load('./untargeted/eBaseNet-8.pth')
    data = {'BaseNet': BaseNet, 'eBaseNet': eBaseNet, 'BaseNet-8': BaseNet_8, 'eBaseNet-8': eBaseNet_8,
            'uBaseNet': uBaseNet, 'ueBaseNet': ueBaseNet, 'uBaseNet-8': uBaseNet_8, 'ueBaseNet-8': ueBaseNet_8}

    # plt.figure(figsize=(12, 8), dpi=200)
    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 16), dpi=80)

    axes[0][0].plot(data['BaseNet']['E'][:99], data['BaseNet']['R'][:99], label='BaseNet', c="blue", linestyle='--')
    axes[0][0].plot(data['eBaseNet']['E'][:99], data['eBaseNet']['R'][:99], label='eBaseNet', c="red", linestyle='--')
    axes[0][0].plot(data['BaseNet-8']['E'][:99], data['BaseNet-8']['R'][:99], label='BaseNet-8', c="blue", linestyle='-')
    axes[0][0].plot(data['eBaseNet-8']['E'][:99], data['eBaseNet-8']['R'][:99], label='eBaseNet-8', c="red", linestyle='-')

    axes[0][1].plot(data['BaseNet']['E'][:99], data['BaseNet']['P'][:99], label='BaseNet', c="blue", linestyle='--')
    axes[0][1].plot(data['eBaseNet']['E'][:99], data['eBaseNet']['P'][:99], label='eBaseNet', c="red", linestyle='--')
    axes[0][1].plot(data['BaseNet-8']['E'][:99], data['BaseNet-8']['P'][:99], label='BaseNet-8', c="blue", linestyle='-')
    axes[0][1].plot(data['eBaseNet-8']['E'][:99], data['eBaseNet-8']['P'][:99], label='eBaseNet-8', c="red", linestyle='-')

    axes[1][0].plot(data['uBaseNet']['E'][:99], data['uBaseNet']['R'][:99], label='BaseNet', c="blue", linestyle='--')
    axes[1][0].plot(data['ueBaseNet']['E'][:99], data['ueBaseNet']['R'][:99], label='eBaseNet', c="red", linestyle='--')
    axes[1][0].plot(data['uBaseNet-8']['E'][:99], data['uBaseNet-8']['R'][:99], label='BaseNet-8', c="blue", linestyle='-')
    axes[1][0].plot(data['ueBaseNet-8']['E'][:99], data['ueBaseNet-8']['R'][:99], label='eBaseNet-8', c="red", linestyle='-')

    axes[1][1].plot(data['uBaseNet']['E'][:99], data['uBaseNet']['P'][:99], label='BaseNet', c="blue", linestyle='--')
    axes[1][1].plot(data['ueBaseNet']['E'][:99], data['ueBaseNet']['P'][:99], label='eBaseNet', c="red", linestyle='--')
    axes[1][1].plot(data['uBaseNet-8']['E'][:99], data['uBaseNet-8']['P'][:99], label='BaseNet-8', c="blue", linestyle='-')
    axes[1][1].plot(data['ueBaseNet-8']['E'][:99], data['ueBaseNet-8']['P'][:99], label='eBaseNet-8', c="red", linestyle='-')

    # axes[0][0].set_xticks(list(np.arange(0, 3.1, 0.1)))
    # axes[0][0].set_yticks(list(np.arange(0, 1.1, 0.05)))
    # axes[0][1].set_xticks(list(np.arange(0, 3.1, 0.1)))
    # axes[0][1].set_yticks(list(np.arange(0, 1.1, 0.05)))
    # axes[1][0].set_xticks(list(np.arange(0, 3.1, 0.1)))
    # axes[1][0].set_yticks(list(np.arange(0, 1.1, 0.05)))
    # axes[1][1].set_xticks(list(np.arange(0, 3.1, 0.1)))
    # axes[1][1].set_yticks(list(np.arange(0, 1.1, 0.05)))

    axes[0][0].set_xlabel('epsilon', fontsize=12)
    axes[0][0].set_ylabel('remaining', fontsize=12)
    axes[0][1].set_xlabel('epsilon', fontsize=12)
    axes[0][1].set_ylabel('probability', fontsize=12)
    axes[1][0].set_xlabel('epsilon', fontsize=12)
    axes[1][0].set_ylabel('remaining', fontsize=12)
    axes[1][1].set_xlabel('epsilon', fontsize=12)
    axes[1][1].set_ylabel('probability', fontsize=12)

    axes[0][0].set_title('Targeted attack on MNIST', fontsize=15)
    axes[0][1].set_title('Targeted attack on MNIST', fontsize=15)
    axes[1][0].set_title('Untargeted attack on MNIST', fontsize=15)
    axes[1][1].set_title('Untargeted attack on MNIST', fontsize=15)

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
