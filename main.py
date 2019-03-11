import argparse
import torch
import torch.optim as optim
import os
import platform
import config
import models
import logging
import dataset
import copy
import scipy.io
import numpy as np

parser = argparse.ArgumentParser(description='MI')
parser.add_argument('--model', type=str, default='fullscale', metavar='M',
                    help='model name')
config.default = config.cfg(parser.parse_known_args()[0].model)
parser.add_argument('--batchsize', type=int, default=config.default.batch_size, metavar='B',
                    help='batch size')
parser.add_argument('--testbatchsize', type=int, default=config.default.test_batch_size, metavar='TB',
                    help='test batch size')
parser.add_argument('--epochs', type=int, default=config.default.epochs, metavar='N',
                    help='number of epochs to train (default: ' + str(config.default.epochs) + ')')
parser.add_argument('--lr', type=float, default=config.default.lr, metavar='LR',
                    help='learning rate (default: ' + str(config.default.lr) + ')')
parser.add_argument('--lrdecay', type=float, default=config.default.lr_decay, metavar='LRD',
                    help='learning rate decay (default: ' + str(config.default.lr_decay) + ')')
parser.add_argument('--momentum', type=float, default=config.default.momentum, metavar='M',
                    help='SGD momentum (default: ' + str(config.default.momentum) + ')')
parser.add_argument('--noise', type=float, default=config.default.noise, metavar='NE',
                    help='sMRI noise (default: ' + str(config.default.noise) + ')')
parser.add_argument('--ratio', type=float, default=config.default.data_ratio, metavar='DR',
                    help='data ratio (default: ' + str(config.default.data_ratio) + ')')
parser.add_argument('--reset', action='store_false' if config.default.data_reset else 'store_true',
                    default=config.default.data_reset, help='reset')
parser.add_argument('--no-cuda', action='store_false' if config.default.nocuda else 'store_true',
                    default=config.default.nocuda, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=config.default.seed, metavar='S',
                    help='random seed (default: ' + str(config.default.seed) + ')')
parser.add_argument('--ci', action='store_false' if config.default.ci else 'store_true',
                    default=config.default.ci, help='running CI')
args = parser.parse_args()
cfg = config.args2dataset(args)
if cfg.ci:
    cfg.batch_size = 8
    cfg.test_batch_size = 8
    cfg.epochs = 1

cfg.cuda = not cfg.nocuda and torch.cuda.is_available()

torch.manual_seed(cfg.seed)
device = torch.device("cuda" if cfg.cuda else "cpu")

save_folder = os.path.join(cfg.root_folder, cfg.paths.save_folder, args.model + '-' + str(cfg.batch_size)
                           + '-' + str(cfg.lr) + '-' + str(cfg.lr_decay) + '-' + str(cfg.momentum)
                           + '-' + str(cfg.noise) + '-' + str(int(cfg.is3d)) + '-' + str(int(cfg.nonorm)))
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

logging_file = os.path.join(save_folder, args.model + cfg.paths.logging_file)
if not os.path.exists(logging_file):
    if platform.system() == 'Windows':
        open(logging_file, 'w')
    else:
        os.mknod(logging_file)

logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s', datefmt='%Y-%m-%d %A %H:%M:%S',
                    filename=logging_file, filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s :  %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

logging.info(cfg)

kwargs = {'num_workers': 0, 'pin_memory': True} if cfg.cuda else {}
gaze = dataset.Gaze(cfg=cfg)
trainset = dataset.trainset(gaze)
testset = dataset.testset(trainset.dataset)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=True, **kwargs)

model = models.create_model(args.model, cfg=cfg)

if os.path.exists(os.path.join(save_folder, args.model + cfg.paths.check_file)):
    start_epoch = torch.load(os.path.join(save_folder, args.model + cfg.paths.check_file))
    model.load_state_dict(torch.load(os.path.join(save_folder, args.model + '_' + str(start_epoch) + '.pth')))
else:
    start_epoch = 0

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=cfg.lr)


def train(epoch):
    model.train()
    criterion = torch.nn.MSELoss().to(device)
    count = 0
    log_step = 100 if not cfg.ci else 1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(target.shape), target)
        loss.backward()
        optimizer.step()
        count += len(data)
        if batch_idx % log_step == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, count, len(train_loader.dataset),
                        100. * count / len(train_loader.dataset), loss.item()))

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    torch.save(model.state_dict(), os.path.join(save_folder, args.model + '_' + str(epoch) + '.pth'))
    torch.save(epoch, os.path.join(save_folder, args.model + cfg.paths.check_file))


def test(epoch, n=1):
    if n != 1:
        global testset
        testset = dataset.testset(trainset.dataset)
        global test_loader
        test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.test_batch_size, shuffle=True, **kwargs)
    model.eval()
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    test_loss = 0
    count = 0
    log_step = 10 if not cfg.ci else 1
    all_output = None
    all_target = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output.view(target.shape), target).item()
            if all_output is None:
                all_output = copy.deepcopy(output)
            else:
                all_output = torch.cat([all_output, copy.deepcopy(output)], 0)
            if all_target is None:
                all_target = copy.deepcopy(target)
            else:
                all_target = torch.cat([all_target, copy.deepcopy(target)], 0)
            count += len(data)
            if batch_idx % log_step == 0:
                logging.info('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                              epoch, count, len(test_loader.dataset),
                              100. * count / len(test_loader.dataset), test_loss / count))

    test_loss /= len(test_loader.dataset)
    logging.info('Test set: Average loss: {:.4f}'.format(test_loss))
    predict_file = os.path.join(save_folder, args.model + '_' + str(epoch) + (
        ('_' + str(n)) if n != 1 else '') + cfg.paths.predict_file)
    scipy.io.savemat(predict_file, {'predict': np.array(all_output.to('cpu')), 'real': np.array(all_target.to('cpu'))})


if __name__ == '__main__':
    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        train(epoch)
        # test(epoch)

    test(cfg.epochs)

    # for n in range(1, 2):
    #     test(cfg.epochs, n)
