"""
from __future__ import print_function
import os
import sys
from datetime import datetime
import multiprocessing
import numpy as np
import dataloaders.cifar100 as dataloader
from networks.arch_craft import Net
from model_code import init_code
from approaches import sgd as approach
import utils
import copy
from evo_utils import StatusUpdateTool

Inc_cls = 5    # Number of classes incremented at each step

#generated_code
class TrainModel(object):
    def __init__(self):
        self.grad_clip = 10
        self.epoch = 20
        self.lr = 0.01
        self.code = code
        self.file_id = os.path.basename(__file__).split('.')[0]
        self.inc = Inc_cls

    def log_record(self, _str, first_time=None):
        dt = datetime.now()
        dt.strftime('%Y-%m-%d %H:%M:%S')
        file_mode = 'w' if first_time else 'a+'
        with open(f'./log/{self.file_id}.txt', file_mode) as f:
            f.write('[%s]-%s\n' % (dt, _str))

    def process(self, s):
        depth = self.code[0]
        width = self.code[1]
        pool_code = copy.deepcopy(self.code[2])
        double_code = copy.deepcopy(self.code[3])
        data, taskcla, inputsize = dataloader.get(seed=s, pc_valid=0, inc=self.inc)
        net = Net(taskcla, depth, width, pool_code, double_code)
        
        # MOD: Remove cudnn & CUDA
        # torch.backends.cudnn.benchmark = True
        # net = net.cuda()
        device = 'cpu'  # MOD
        net.to(device)  # MOD

        total = sum([param.nelement() for param in net.parameters()])
        self.log_record('Number of parameter: %.4fM' % (total / 1e6))

        appr = approach.Appr(net, nepochs=self.epoch, sbatch=128, lr=self.lr, clipgrad=self.grad_clip)
        self.log_record('-' * 100)

        acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        aps = []
        afs = []

        for t, ncla in taskcla:
            self.log_record('*' * 100)
            self.log_record('Task {:2d} ({:s})'.format(t, data[t]['name']))
            self.log_record('*' * 100)

            # MOD: Move data to CPU
            xtrain = data[t]['train']['x'].to(device)
            ytrain = data[t]['train']['y'].to(device)
            xvalid = data[t]['valid']['x'].to(device)
            yvalid = data[t]['valid']['y'].to(device)
            task = t

            # Train
            appr.train(task, xtrain, ytrain, xvalid, yvalid)
            self.log_record('-' * 100)

            # Test
            for u in range(t + 1):
                xtest = data[u]['test']['x'].to(device)  # MOD
                ytest = data[u]['test']['y'].to(device)  # MOD
                test_loss, test_acc = appr.eval(u, xtest, ytest)
                self.log_record('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(
                    u, data[u]['name'], test_loss, 100 * test_acc))
                acc[t, u] = test_acc
                lss[t, u] = test_loss

            now_acc = np.mean(acc[t, :t+1])
            aps.append(now_acc)
            self.log_record('ap:%.5f' % now_acc)

            if t != 0:
                f = sum([max(acc[j, k] for j in range(k, t)) - acc[t, k] for k in range(t)])
                af = f / t
                afs.append(af)
                self.log_record('af:%.5f' % af)

        self.log_record('*' * 100)
        self.log_record(str(aps))
        self.log_record(str(afs))
        
        aia = 100 * np.mean(aps)
        final_acc = 100 * np.mean(acc[-1, :])
        af = afs[-1] * 100 if afs else 0

        self.log_record('aia:%.3f' % aia)
        self.log_record('aa:%.3f' % final_acc)
        self.log_record('af:%.3f' % af)
        self.log_record('Done!')
        return round(aia, 3)

class RunModel(object):
    def do_work(self, gpu_id, file_id):
        # MOD: Disable GPU environment
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # MOD: No GPU usage
        best_acc = 0.0
        m = TrainModel()
        try:
            m.log_record('Running on CPU, worker name:%s[%d]' % (multiprocessing.current_process().name, os.getpid()), first_time=True)
            best_acc = m.process(s=0)
        except BaseException as e:
            print('Exception occurs, file:%s, pid:%d...%s' % (file_id, os.getpid(), str(e)))
            m.log_record('Exception occur:%s' % str(e))
        finally:
            m.log_record('Finished-Acc:%.4f' % best_acc)
            with open(f'./populations/after_{file_id[4:6]}.txt', 'a+') as f:
                f.write('%s=%.5f\n' % (file_id, best_acc))
"""


