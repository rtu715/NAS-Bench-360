from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
sys.path.insert(0, '../chrysalis')
#from chrysalis import Chrysalis
from nas import MixedOptimizer
from models import DeepConRddDistances
from generator import PDNetDataset
from dataio import load_list
from metrics import evaluate_distances
from eval_nb_orig import calculate_mae
from functools import partial
from timeit import default_timer
sys.path.insert(0, '../../autodeeplab')
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args


def logCoshLoss(y_t, y_prime_t, reduction='mean', eps=1e-12):
    if reduction == 'mean':
        reduce_fn = torch.mean
    elif reduction == 'sum':
        reduce_fn = torch.sum
    else:
        reduce_fn = lambda x: x
    x = y_prime_t - y_t
    return reduce_fn(torch.log((torch.exp(x) + torch.exp(-x)) / 2))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = logCoshLoss(output, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mae = F.l1_loss(output, target, reduction='mean')
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMAE: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), mae.item()))
            if args.dry_run:
                break


def test(model, device, test_loader, split='validation'):
    model.eval()
    test_loss = 0
    test_mae = 0
    w = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += logCoshLoss(output, target, reduction='sum').item()
            test_mae += F.l1_loss(output, target, reduction='sum').item()
            w = target.shape[-1]

    test_loss = test_loss / (len(test_loader.dataset) * w * w)
    test_mae = test_mae / (len(test_loader.dataset) * w * w)

    print(f'\n{split} loss: {test_loss:.4f}\t{split} MAE: {test_mae:.4f}\n')

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PDnet XD protein folding')
    parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--test-batch-size', type=int, default=2, 
                        metavar='N',
                        help='input batch size for testing (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--eval-only', action='store_true', default=False,
                        help='Only run evaluataion on saved model')
    parser.add_argument('--data-dir', 
                        default='/home/ubuntu/github.com/nick11roberts/generalizing-primitives/autodeeplab/',
                        help='Dataset directory')

    parser.add_argument('--training-window', type=int, default=128,
                        help='training window (default: 128)')
    parser.add_argument('--depth', type=int, default=8,
                        help='architecture depth (default: 8)')
    parser.add_argument('--width', type=int, default=16,
                        help='architecture width (default: 16)')
    parser.add_argument('--no-dilation', action='store_true', default=False,
                        help='do not use dilation over blocks')

    parser.add_argument('--arch', default='conv', 
                        help='architecture: conv | xd (default conv)')
    parser.add_argument('--arch-lr', type=float, default=0.001, 
                        help='arch learning rate (default: 0.001)')
    parser.add_argument('--arch-sgd', action='store_true', default=False,
                        help='Use SGD as arch optimizer')
    parser.add_argument('--arch-momentum', type=float, default=0.0, 
                        help='arch momentum if using SGD (default: 0.0)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='arch warmup epochs (default 0)')
    parser.add_argument('--window-eval', action='store_true', default=False,
                        help='Test evaluation using a sliding window')
    parser.add_argument('--window-stride', type=int, default=-1,
                        help='Stride of sliding window evaluation')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    training_window = args.training_window
    arch_depth = args.depth
    filters_per_layer = args.width #16
    expected_n_channels = 57
    pad_size = 10

    all_feat_paths = [args.data_dir + '/deepcov/features/', 
        args.data_dir + '/psicov/features/', args.data_dir + '/cameo/features/']
    all_dist_paths = [args.data_dir + '/deepcov/distance/', 
        args.data_dir + '/psicov/distance/', args.data_dir + '/cameo/distance/']

    deepcov_list = load_list(args.data_dir + '/deepcov.lst', -1)

    length_dict = {}
    for pdb in deepcov_list:
        (ly, seqy, cb_map) = np.load(
            args.data_dir + '/deepcov/distance/' + pdb + '-cb.npy', 
            allow_pickle = True)
        length_dict[pdb] = ly

    print('')
    print('Split into training and validation set..')
    valid_pdbs = deepcov_list[:int(0.3 * len(deepcov_list))]
    train_pdbs = deepcov_list[int(0.3 * len(deepcov_list)):]
    if len(deepcov_list) > 200:
        valid_pdbs = deepcov_list[:100]
        train_pdbs = deepcov_list[100:]

    print('Total validation proteins : ', len(valid_pdbs))
    print('Total training proteins   : ', len(train_pdbs))

    train_dataset = PDNetDataset(
        train_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, args.batch_size, expected_n_channels, label_engineering = '16.0')
    valid_dataset = PDNetDataset(
        valid_pdbs, all_feat_paths, all_dist_paths, training_window, pad_size, args.batch_size, expected_n_channels, label_engineering = '16.0')

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(valid_dataset, **test_kwargs)
    
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)

    def make_model():
        #model = DeepConRddDistances(
        #    L=training_window, 
        #    num_blocks=arch_depth, 
        #    width=filters_per_layer, 
        #    expected_n_channels=expected_n_channels,
        #    no_dilation=args.no_dilation
        #)
        #model.apply(weight_init) # Use Keras default initialization
        arguments = obtain_retrain_autodeeplab_args()
        arguments.num_classes = 1
        arguments.autodeeplab = 'train'
        arguments.exp = 'archs/prot_lr0.025/'
        model = Retrain_Autodeeplab(arguments, input_channels=57)
        args.arch_lr = 0.0
        model.cuda()
        return model

    model = make_model()
    print(model)
    #print(model.count_params())
    
    # Optimizer
    opts = [optim.RMSprop(model.parameters(), 
        lr=args.lr, alpha=0.9, eps=1e-7)]
    optimizer = MixedOptimizer(opts, op_decay=None)

    # Scheduler
    def weight_sched(epoch):
        return 1.0
    def arch_sched(epoch):
        return 0.0 if (epoch < args.warmup_epochs) else weight_sched(epoch)
    sched_groups = [weight_sched]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=sched_groups)

    # Training and validation
    best_loss = np.inf
    torch.save(model.state_dict(), f'pdnet_{args.seed}.pt')

    for epoch in range(1, args.epochs + 1):
        t1 = default_timer()
        train(args, model, device, train_loader, optimizer, epoch)
        loss = test(model, device, test_loader)
        if loss < best_loss:
            best_loss = loss
            # Save weights
            print('best loss, saving model...\n')
            torch.save(model.state_dict(), f'pdnet_{args.seed}.pt')
        scheduler.step()
        t2 = default_timer()
        print('time:', t2-t1)


    # Before testing, free up some storage
    train_loader = None
    train_dataset = None
    test_loader = None
    valid_dataset = None
    del train_loader
    del train_dataset
    del test_loader
    del valid_dataset

    # Test set evaluation
    psicov_list = load_list(args.data_dir + 'psicov.lst')
    psicov_length_dict = {}
    for pdb in psicov_list:
        (ly, seqy, cb_map) = np.load(args.data_dir + '/psicov/distance/' + pdb + '-cb.npy', allow_pickle = True)
        psicov_length_dict[pdb] = ly

    cameo_list = load_list(args.data_dir + 'cameo-hard.lst')
    cameo_length_dict = {}
    for pdb in cameo_list:
        (ly, seqy, cb_map) = np.load(args.data_dir + '/cameo/distance/' + pdb + '-cb.npy', allow_pickle = True)
        cameo_length_dict[pdb] = ly

    evalsets = {}
    ####
    ##evalsets['validation'] = {'LMAX': 512,  'list': valid_pdbs, 'lendict': length_dict}
    #evalsets['psicov'] = {'LMAX': 512,  'list': psicov_list, 'lendict': psicov_length_dict}
    #evalsets['cameo']  = {'LMAX': 1300, 'list': cameo_list,  'lendict': cameo_length_dict}
    ####

    evalsets['psicov'] = {'LMAX': 512, 
        'list': psicov_list, 'lendict': psicov_length_dict}
    #evalsets['cameo']  = {'LMAX': 1300, 'list': cameo_list,  'lendict': cameo_length_dict} # TODO just use psicov

    model = make_model()
    model.load_state_dict(torch.load(f'pdnet_{args.seed}.pt'))
    model.eval()

    for my_eval_set in evalsets:
        print('')
        print(f'Evaluate on the {my_eval_set} set..')
        my_list = evalsets[my_eval_set]['list']
        LMAX = evalsets[my_eval_set]['LMAX']
        length_dict = evalsets[my_eval_set]['lendict']
        print('L', len(my_list))
        print(my_list)

        # For AWS shared memory problem
        cuda_kwargs = {'batch_size': 2,
                       'num_workers': 0,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)
        
        test_dataset = PDNetDataset(
            my_list, all_feat_paths, all_dist_paths, LMAX, 
            pad_size, 1, expected_n_channels, label_engineering = None)
            # TODO clip at 16 ? 
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **test_kwargs)

        # Padded but full inputs/outputs
        with torch.no_grad():
            P = []
            targets = []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                for i in range(args.batch_size):
                    targets.append(
                        np.expand_dims(
                            target.cpu().numpy()[i].transpose(1, 2, 0), axis=0))
                if args.window_eval:
                    # If the window stride is mistakenly set to larger than 
                    # window size, set the stride to window size
                    if args.window_stride >= args.training_window:
                        args.window_stride = args.training_window
                    out = model.forward_window(data, stride=args.window_stride)
                else:
                    out = model(data)
                P.append(out.cpu().numpy().transpose(0, 2, 3, 1))
                
                # TODO remove, this is for testing
                #model.forward_window(data)
                # TODO remove, this is for testing

            # Combine P, convert to numpy
            P = np.concatenate(P, axis=0)

        Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
        for i, xy in enumerate(targets):
            Y[i, :, :, 0] = xy[0, :, :, 0]

        # Average the predictions from both triangles
        for j in range(0, len(P[0, :, 0, 0])):
            for k in range(j, len(P[0, :, 0, 0])):
                P[ :, j, k, :] = (P[ :, k, j, :] + P[ :, j, k, :]) / 2.0
        P[ P < 0.01 ] = 0.01

        # Remove padding, i.e. shift up and left by int(pad_size/2)
        P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
        Y[:, :LMAX-pad_size, :LMAX-pad_size, :] = Y[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]

        print('')
        print('Evaluating distances..')
        calculate_mae(P, Y, my_list, length_dict)

if __name__ == '__main__':
    main()
    