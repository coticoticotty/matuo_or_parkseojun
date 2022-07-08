import matplotlib.pyplot as plt

def show_graphs(num_epochs, train_loss, val_loss, train_acc, val_acc, models_name):
    fig = plt.figure(figsize=(13, 5))
    x_value = list(range(1, num_epochs+1))
    ax_1 = fig.add_subplot(121)
    ax_1.plot(x_value, train_loss, label='Training')
    ax_1.plot(x_value, val_loss, label='Validation', linestyle='--')
    # plt.xticks(x_value, x_value)
    ax_1.set_ylabel('Loss')
    ax_1.set_xlabel('Epochs')
    ax_1.set_xlim([0, num_epochs+1])
    plt.legend(loc='upper right')
    loss_title = f'Train By {models_name}[Loss]'
    plt.title(loss_title)

    ax_2 = fig.add_subplot(122)
    ax_2.plot(x_value, train_acc, label='Training')
    ax_2.plot(x_value, val_acc, label='Validation', linestyle='--')
    # plt.xticks(x_value, x_value)
    ax_2.set_ylabel('Accuracy')
    ax_2.set_xlabel('Epochs')
    plt.legend(loc='lower right')
    ax_1.set_xlim([0, num_epochs+1])
    acc_title = f'Train By {models_name}[Acc]'
    plt.title(acc_title)

    plt.show()
    fig.savefig(f'{models_name}.png')