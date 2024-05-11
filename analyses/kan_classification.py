#!/usr/bin/env python3
"""
Train KANs to predict subjective text difficulty
"""

from analyses.utils.utils_analyses import prepare_kan_input, kan_train_test_split
from kan import KAN
import torch
import numpy as np
import matplotlib.pyplot as plt

# how accurate is this formula?
def acc(formula1, formula2, X, y):
    batch = X.shape[0]
    correct = 0
    for i in range(batch):
        logit1 = np.array(formula1.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
        logit2 = np.array(formula2.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
        correct += (logit2 > logit1) == y[i]
    return correct/batch


def main():

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device = f'cuda:{device_idx}'
    else:
        device = 'cpu'

    path_to_rms = 'data/reading_measures_corrected.csv'
    path_to_ratings = 'data/participant_info/participant_results.csv'

    exclude_subjects = ['ET_03', 'ET_11', 'ET_39', 'ET_49', 'ET_67', 'ET_83']

    data_dict = prepare_kan_input(
        path_to_rms=path_to_rms,
        path_to_ratings=path_to_ratings,
        exclude_subjects=exclude_subjects,
    )
    device = 'cpu'

    dataset = kan_train_test_split(
        data=data_dict,
        label='difficulty',
        device=device,
        task='classification',
    )


    # model = KAN(width=[78, 6, 1], grid=5, k=3, seed=0)
    # model.to(device)
    #
    # model.train(dataset, opt="Adam", steps=20, lamb=0.01, lamb_entropy=10.)
    # # initialize a more fine-grained KAN with G=10
    # model2 = KAN(width=[78, 6, 1], grid=10, k=3)
    # # initialize model2 from model
    # model2.initialize_from_another_model(model, dataset['train_input'])
    # model2.train(dataset, opt="Adam", steps=20)

    # grids = np.array([5, 10, 20, 50, 100])
    #
    # train_losses = []
    # test_losses = []
    # steps = 50
    # k = 3
    #
    #
    #
    # for i in range(grids.shape[0]):
    #     if i == 0:
    #         model = KAN(width=[78, 6, 1], grid=grids[i], k=k)
    #         model.to(device)
    #     if i != 0:
    #         model = KAN(width=[78, 6, 1], grid=grids[i], k=k).initialize_from_another_model(model,
    #                                                                                        dataset['train_input']).to(device)
    #     results = model.train(dataset, opt="LBFGS", steps=steps, stop_grid_update_step=30)
    #     train_losses += results['train_loss']
    #     test_losses += results['test_loss']
    #
    # plt.plot(train_losses)
    # plt.plot(test_losses)
    # plt.legend(['train', 'test'])
    # plt.ylabel('RMSE')
    # plt.xlabel('step')
    # plt.yscale('log')
    # plt.savefig('kan_losses.png')
    # plt.close()


    def train_acc():
        return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

    def test_acc():
        return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

    grids = np.array([5, 10, 20, 50, 100])

    train_losses = []
    test_losses = []
    steps = 50
    k = 3
    model = KAN(width=[2,2], grid=3, k=3)

    for i in range(grids.shape[0]):
        if i == 0:
            model = KAN(width=[78, 6, 5], grid=grids[i], k=k)
            model.to(device)
        if i != 0:
            model = KAN(width=[78, 6, 5], grid=grids[i], k=k).initialize_from_another_model(
                model, dataset['train_input'].to(device)
            )
        results = model.train(dataset, opt='LBFGS', steps=steps, metrics=(train_acc, test_acc),
                              loss_fn=torch.nn.CrossEntropyLoss())
        train_losses += results['train_loss']
        test_losses += results['test_loss']

    plt.plot(train_losses)
    plt.plot(test_losses)
    plt.legend(['train', 'test'])
    plt.ylabel('CE Loss')
    plt.xlabel('step')
    plt.yscale('log')
    plt.savefig('kan_losses_CE.png')
    plt.close()

    breakpoint()

    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
    model.auto_symbolic(lib=lib)
    formula1, formula2 = model.symbolic_formula()[0]
    print(formula1)
    print(formula2)
    breakpoint()
    train_acc = acc(formula1=formula2, formula2=formula2, X=dataset['train_input'], y=dataset['train_label'])
    test_acc = acc(formula1=formula2, formula2=formula2, X=dataset['test_input'], y=dataset['test_label'])
    print(f'train accuracy: {train_acc}')
    print(f'test accuracy: {test_acc}')
    breakpoint()






if __name__ == '__main__':
    raise SystemExit(main())
