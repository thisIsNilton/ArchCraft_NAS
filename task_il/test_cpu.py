from __future__ import print_function
import os
import sys
import numpy as np
import torch
import copy
import dataloaders.cifar100 as dataloader
from approaches import sgd as approach
import utils
import class_nilton as intruder


Inc_cls = 25  # Número de classes incrementadas a cada passo

def run_test():
    indi_no = 0
    code = [2, 40, [1, 1, 2, 2, 2], [0, 0, 0, 1, 2]]  # AlexAC-A, about 6.28M
    network_choices = ['arch_craft', 'alexnet']
    chosen_network = network_choices[0]
    m = TrainModel(code=code, indi_no=indi_no, network_name=chosen_network)
    m.process(1993)

class TrainModel(object):
    def __init__(self, code, indi_no, network_name):
        self.device = torch.device("cpu")  # Força o uso da CPU
        self.grad_clip = 10
        self.epoch = 2
        self.lr = 0.01
        self.code = code
        self.file_id = 'indiH%03d' % indi_no
        self.inc = Inc_cls
        self.network_name = network_name

    def process(self, s):
        print('\n\n')
        print(self.file_id)
        depth = self.code[0]
        width = self.code[1]
        pool_code = copy.deepcopy(self.code[2])
        double_code = copy.deepcopy(self.code[3])
        print(self.code)
        # Carga de dados
        data, taskcla, inputsize = dataloader.get(seed=s, pc_valid=0.15, inc=self.inc)
        # print(f">>> data\n----\n{data[0]}")
        # print(f"\n\n>>> taskcla\n----\n{taskcla}")
        # print(f"\n\n>>> inputsize\n----\n{inputsize}")
        # NILTON: Visualiza imagens
        # intruder.show_task_images(data=data, task_id=0, split='train', n=10)

        if self.network_name == 'arch_craft':
            from networks.arch_craft import Net
            net = Net(taskcla, depth, width, pool_code, double_code)
        elif self.network_name == 'alexnet':
            from networks.alexnet import Net
            net = Net(taskcla)
        else:
            raise NotImplementedError(f"Unknown type {self.network_name}")

        net.to(self.device)  # Move o modelo para CPU
        total = sum([param.nelement() for param in net.parameters()])
        print('Número de parâmetros: {:.4f}M'.format(total / 1e6))

        appr = approach.Appr(net, nepochs=self.epoch, sbatch=128, lr=self.lr, clipgrad=self.grad_clip)
        print(appr.criterion)
        utils.print_optimizer_config(appr.optimizer)
        print('-' * 100)

        acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        aps = []
        afs = []

        for t, ncla in taskcla:
            print('*' * 100)
            print('Task {:2d} ({:s})'.format(t, data[t]['name']))
            print('*' * 100)

            # Move dados para CPU
            xtrain = torch.tensor(data[t]['train']['x']).to(self.device)
            ytrain = torch.tensor(data[t]['train']['y']).to(self.device)
            xvalid = torch.tensor(data[t]['valid']['x']).to(self.device)
            yvalid = torch.tensor(data[t]['valid']['y']).to(self.device)
            task = t

            # Treinamento
            appr.train(task, xtrain, ytrain, xvalid, yvalid)
            print('-' * 100)

            # Teste
            for u in range(t + 1):
                xtest = torch.tensor(data[u]['test']['x']).to(self.device)
                ytest = torch.tensor(data[u]['test']['y']).to(self.device)
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                print('>>> Teste na tarefa {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(
                    u, data[u]['name'], test_loss, 100 * test_acc))
                acc[t, u] = test_acc
                lss[t, u] = test_loss

            now_acc = np.mean(acc[t, :t+1])
            aps.append(now_acc)
            print(f'ap: {now_acc:.5f}')

            if t != 0:
                f = sum(max(acc[j, k] for j in range(k, t)) - acc[t, k] for k in range(t)) / t
                afs.append(f)
                print(f'af: {f:.5f}')

        print('*' * 100)
        print('Acurácias =')
        for i in range(acc.shape[0]):
            print('\t', end='')
            for j in range(acc.shape[1]):
                print('{:5.1f}% '.format(100 * acc[i, j]), end='')
            print()
        print('*' * 100)

        aia = np.mean(aps)
        print(f'aia: {aia:.5f}')

        final_acc = np.mean(acc[-1, :])
        print(f'final_acc: {final_acc:.5f}')
        print('Finalizado!')
        return final_acc

if __name__ == '__main__':
    run_test()
