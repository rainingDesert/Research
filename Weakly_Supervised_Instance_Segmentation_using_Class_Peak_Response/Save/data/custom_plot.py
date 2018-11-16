import numpy as np
import matplotlib.pyplot as plt


def plot_train_val_ac_loss():
    def toFloat(x):
        return float(int(float(x) * 100)) / 100

    loss = []
    val_loss = []
    acc = []
    val_acc = []

    with open('train.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'val' in line:
                line_loss, line_acc = line.split(' ')[0].split(':')[-1], line.split(' ')[1].split(':')[-1]
                val_loss.append(toFloat(line_loss))
                val_acc.append(toFloat(line_acc))
            elif 'epoch' not in line:
                line_loss, line_acc = line.split(' ')[0].split(':')[-1], line.split(' ')[1].split(':')[-1]
                loss.append(toFloat(line_loss))
                acc.append(toFloat(line_acc))

    x_train = np.arange(0, len(loss), 100)
    x_val = np.arange(0, len(val_acc), 10)

    plt.figure(figsize=(10, 5))
    plt.plot(list(range(len(x_train))), np.array(loss)[x_train], color='r')
    plt.plot(list(range(len(x_train))), np.array(acc)[x_train], color='b')
    plt.plot(list(range(len(x_val))), np.array(val_loss)[x_val], color='g')
    plt.plot(list(range(len(x_val))), np.array(val_acc)[x_val], color='yellow')
    plt.savefig('loss_acc.png')

    # plt.figure(figsize=(10, 5))
    # plt.plot(list(range(len(x_val))), np.array(val_loss)[x_val], color='r')
    # plt.plot(list(range(len(x_val))), np.array(val_acc)[x_val], color='b')
    # plt.savefig('val_loss_acc.png')

    print('plot successfully!')


if __name__ == '__main__':
    print('1) plot train/val acc and loss')
    my_input = input('option:')
    if int(my_input) == 1:
        plot_train_val_ac_loss()
