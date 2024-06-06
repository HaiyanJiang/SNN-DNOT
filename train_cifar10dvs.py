# -*- coding: utf-8 -*-


import datetime
import os
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import sys
from torch.cuda import amp
from models import spiking_vgg
from modules import neuron, surrogate
import argparse
import numpy as np
import math
import pickle

from models import spiking_vgg
from modules import neuron, surrogate
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, seed_all

# from data import datapool
from datasets.cifar10dvs_loader import data_cifar10_dvs_loader

from functions_cifar10dvs import train_cifar10dvs, evaluate_cifar10dvs
import data.utils as utils

seed_all(seed=2023, benchmark=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Classify DVS-CIFAR10')

    parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', type=str, default='online_spiking_vgg11_ws')

    # dataset is cifar10dvs
    parser.add_argument('--dataset', default='cifar10dvs', type=str)
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--optimizer', default='SGD', type=str, help='use which optimizer. SGD or Adam')

    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay for SGD or Adam')
    parser.add_argument('--j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4) workers')

    parser.add_argument('--lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('--step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('--step_gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('--T_max', default=300, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('--print_freq', default=100, type=int, help='print freq in train and evaluate')

    parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
    parser.add_argument('--resume', type=int, help='resume or not from the checkpoint path')
    parser.add_argument('--output_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    # ### Settings of the LIFSpike Neuron
    parser.add_argument('--T', default=6, type=int, help='simulating time-steps')
    parser.add_argument('--tau', default=2., type=float)    # The decay constant is lambda= 1.0-1.0/tau, lambda < 1.0


    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')  # default false
    parser.add_argument('--tb', action='store_true', help='using tensorboard')  # using tensorboard
    parser.add_argument('--autoaug', action='store_true', help='using auto augmentation')  # default false
    parser.add_argument('--cutout', action='store_true', help='using cutout')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='dropout rate')

    parser.add_argument('--test_only', action='store_true')  # 'store_true' means by default, it is false.

    # parser.add_argument('--stochdepth_rate', type=float, default=0.0)
    # parser.add_argument('--cnf', type=str)
    # parser.add_argument('--T_train', default=None, type=int)
    # parser.add_argument('--dts_cache', type=str, default='./dts_cache')


    parser.add_argument('--loss_lambda', type=float, default=0.001, help='weight of the mse_loss' ) #  ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
    parser.add_argument('--online_update', action='store_true',  help='use online update')   # default means false.

    parser.add_argument('--use_base', action='store_true',  help='use baseline model or not')   # default means false.
    parser.add_argument('--time_base', default=2, type=int, help='the time_base model')
    parser.add_argument('--reg_lambda', type=float, default=0.05, help='weight of the reg_loss. L1 or L2')   # first try L2 loss.


    args = parser.parse_args()

    # args = parser.parse_args([])  # for jupyter notebook
    print(args)

    return args




def logger_plot_save(logger_train_test_acc, logger_train_test_loss, prefix_name, plot_dir):
    # snames = ['Acc1.', 'Acc5.']
    # snames = ['Loss', 'Batch Loss', 'CE Loss', 'MSE Loss', 'Loss_0', 'Acc_0']

    # ## Plot the loss and accuracy
    fig = logger_train_test_acc.plot(['Train Acc1.', 'Test Acc1.'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_acc1.pdf')
    fig = logger_train_test_acc.plot(['Train Acc5.', 'Test Acc5.'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_acc5.pdf')
    fig = logger_train_test_acc.plot(['LR'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_LR.pdf')

    # # ### fig = logger.plot()  # this cause problems, as the scales are different
    fig = logger_train_test_loss.plot(['Train Loss', 'Test Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_loss.pdf')
    fig = logger_train_test_loss.plot(['Train Batch Loss', 'Test Batch Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_batch_loss.pdf')
    fig = logger_train_test_loss.plot(['Train CE Loss', 'Test CE Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_ce_loss.pdf')
    fig = logger_train_test_loss.plot(['Train MSE Loss', 'Test MSE Loss'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_mse_loss.pdf')

    fig = logger_train_test_loss.plot(['Train CLS Loss'])  # Only for train
    fig.savefig(plot_dir + f'/{prefix_name}_logger_cls_loss.pdf')
    fig = logger_train_test_loss.plot(['Train REG Loss'])  # Only for train
    fig.savefig(plot_dir + f'/{prefix_name}_logger_reg_loss.pdf')

    fig = logger_train_test_loss.plot(['Train Loss_0', 'Test Loss_0'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_loss_0.pdf')
    fig = logger_train_test_loss.plot(['Train Acc_0', 'Test Acc_0'])
    fig.savefig(plot_dir + f'/{prefix_name}_logger_acc_0.pdf')

    # ## Save it to a file
    fname = plot_dir + f'/{prefix_name}_logger_acc.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(logger_train_test_acc.numbers, f)
    fname = plot_dir + f'/{prefix_name}_logger_loss.pkl'
    with open(fname, 'wb') as f:
        pickle.dump(logger_train_test_loss.numbers, f)
    # ## and later you can load it
    # with open('test_logger.pkl', 'rb') as f:
    #     dt = pickle.load(f)




def main(args):
    # args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    num_classes = 10

    # Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
        print("CUDA (GPU) is available.")
    else:
        args.device = torch.device("cpu")
        print("CUDA (GPU) is not available. Using CPU.")
    device = args.device


    train_loader, test_loader = data_cifar10_dvs_loader(
        time_steps=args.T,
        batch_size=args.batch_size,
        num_workers=args.j)

    # train_loader, test_loader, train_sampler = datapool(
    #     args.dataset.lower(), batch_size=args.batch_size, num_workers=args.j,
    #     distributed=False, cache_dataset=False,  time_steps=args.T)


    model = spiking_vgg.__dict__[args.model](
        single_step_neuron=neuron.OnlineLIFNode,
        surrogate_function=surrogate.Sigmoid(alpha=4.),
        track_rate=True,
        c_in=2,
        num_classes=10,
        grad_with_rate=True,
        tau=args.tau,
        neuron_dropout=args.drop_rate,
        fc_hw=1,
        v_reset=None)


    print("===> Creating model")
    print(model)
    print('=== *** Total Parameters: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    # model.cuda()
    model = model.to(args.device)


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.optimizer)


    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)


    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    max_test_acc1 = 0.0
    max_test_acc5 = 0.0


    fname = f'_{args.dataset}_{args.model}_T_{args.T}_opt_{args.optimizer}' \
        f'_lr_{args.lr}_bs_{args.batch_size}_wd_{args.weight_decay}_epochs_{args.epochs}' \
        f'_autoaug_{args.autoaug}_coutout_{args.cutout}'
    output_dir = args.output_dir + fname

    if args.lr_scheduler == 'CosALR':
        output_dir += f'_CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        output_dir += f'_StepLR_{args.step_size}_{args.step_gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)
    if args.amp:
        output_dir += '_amp'

    prefix_name = f'{args.dataset}_{args.model}_T_{args.T}_tau_{args.tau}'
    utils.mkdir(output_dir)
    print(output_dir)
    plot_dir = os.path.join('./output_dir', fname + 'plots')
    utils.mkdir(plot_dir)


    # ## Resume, optionally resume from a checkpoint
    if args.resume:
        print('==> Resuming from checkpoint...')
        resume_name = os.path.join(output_dir, f'{prefix_name}_checkpoint_latest.pth')

        assert os.path.isfile(resume_name), 'Error: no checkpoint directory found!'

        # ## checkpoint = torch.load(args.resume)
        checkpoint = torch.load(resume_name, map_location='cpu')
        # model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        max_test_acc1 = checkpoint['max_test_acc1']
        max_test_acc5 = checkpoint['max_test_acc5']

        # ## The logger with 'append' mode
        logger = utils.get_logger(output_dir + f'/{prefix_name}_training_log.log', file_mode='a')
        logger.parent = None
        logger.info(output_dir)
        logger.info(args)
        logger.info("Resume training")

        try:
            print('==> Resuming logger_train and logger_test...')
            logger_train_test_acc = Logger(output_dir + f'/{prefix_name}_logger_train_test_acc.txt', title='TrainTestAcc', resume=True)
            logger_train_test_loss = Logger(output_dir + f'/{prefix_name}_logger_train_test_loss.txt', title='TrainTestLoss', resume=True)
            logger_train_test_acc.set_formats([
                '{0:d}', '{0:.6f}',
                '{0:.3f}', '{0:.3f}',
                '{0:.3f}', '{0:.3f}',
                ])
            logger_train_test_loss.set_formats([
                '{0:d}', '{0:.6f}',
                '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}', '{0:.4f}', '{0:.4f}',
                '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}',
                ])
        except:
            print('Cannot open the designated log file.')

    # ## If there is not from resume
    else:
        # ## The logger with 'write' mode
        logger = utils.get_logger(output_dir + f'/{prefix_name}_training_log.log', file_mode='w')
        logger.parent = None
        logger.info(output_dir)
        logger.info(args)
        logger.info("Start training")

        print('No existing log file, have to create a new one.')
        print('==> Trainig from epoch=0...')

        # #### New added, using the logger
        print('==> Creating new logger_train and logger_test...')
        logger_train_test_acc = Logger(output_dir + f'/{prefix_name}_logger_train_test_acc.txt', title='TrainTestAcc')
        snames = ['Acc1.', 'Acc5.']
        logger_train_test_acc.set_names(
            ['Epoch', 'LR'] + ['Train ' + i for i in snames] + ['Test ' + i for i in snames]
            )
        logger_train_test_acc.set_formats([
            '{0:d}', '{0:.6f}',
            '{0:.3f}', '{0:.3f}',
            '{0:.3f}', '{0:.3f}',
            ])

        logger_train_test_loss = Logger(output_dir + f'/{prefix_name}_logger_train_test_loss.txt', title='TrainTestLoss')
        train_snames = ['Loss', 'Batch Loss', 'CE Loss', 'MSE Loss', 'Loss_0', 'Acc_0', 'CLS Loss', 'REG Loss']
        test_snames = ['Loss', 'Batch Loss', 'CE Loss', 'MSE Loss', 'Loss_0', 'Acc_0']
        logger_train_test_loss.set_names(
            ['Epoch', 'LR'] + ['Train ' + i for i in train_snames] + ['Test ' + i for i in test_snames]
            )
        logger_train_test_loss.set_formats([
            '{0:d}', '{0:.6f}',
            '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}', '{0:.4f}', '{0:.4f}',
            '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.4f}', '{0:.3f}',
            ])
        # # #### New added




    if args.tb:  # using tensorboard
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + f'/{prefix_name}_train_logs', purge_step=purge_step_train)
        test_tb_writer = SummaryWriter(output_dir + f'/{prefix_name}_test_logs', purge_step=purge_step_te)
        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    with open(output_dir + f'/{prefix_name}_args.txt', 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))


    criterion_mse = nn.MSELoss(reduce=True)

    if args.test_only:
        results = evaluate_cifar10dvs(test_loader, model, criterion_mse, num_classes, device, args)
        print(results)
        return


    print('==> Start training')
    print('==> args that feed into training!')
    print(f'==> Trainig from epoch={args.start_epoch}...')
    print(args)

    start_time = time.time()
    writer = SummaryWriter(os.path.join(output_dir, f'{prefix_name}_logs.logs'), purge_step=args.start_epoch)

    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        cur_lr = optimizer.param_groups[0]["lr"]

        results = train_cifar10dvs(train_loader, model, optimizer, criterion_mse, num_classes, device, epoch, args, scaler)

        train_acc1, train_acc5 = results[:2]
        train_loss, train_loss_batch = results[2:4]
        train_loss_ce, train_loss_mse = results[4:6]
        train_loss_, train_acc_ = results[6:8]
        train_cls_loss, train_reg_loss = results[8:10]


        logger.info(
            'Train Epoch:[{}/{}]\t train_acc1={:.3f}\t train_acc5={:.3f}\t '
            'train_loss={:.5f}\t train_loss_batch={:.5f}\t train_loss_ce={:.5f}\t'
            'train_loss_mse={:.5f}\t train_loss_={:.5f}\t train_acc_={:.5f}\t'
            'train_cls_loss={:.5f}\t train_reg_loss_={:.5f}\t'
            .format(epoch, args.epochs, train_acc1, train_acc5,
                    train_loss, train_loss_batch, train_loss_ce,
                    train_loss_mse, train_loss_, train_acc_,
                    train_cls_loss, train_reg_loss,
                    )
            )
        print(
            'Train Epoch:[{}/{}]\t train_acc1={:.3f}\t train_acc5={:.3f}\t '
            'train_loss={:.5f}\t train_loss_batch={:.5f}\t train_loss_ce={:.5f}\t'
            'train_loss_mse={:.5f}\t train_loss_={:.5f}\t train_acc_={:.5f}\t'
            'train_cls_loss={:.5f}\t train_reg_loss_={:.5f}\n'
            .format(epoch, args.epochs, train_acc1, train_acc5,
                    train_loss, train_loss_batch, train_loss_ce,
                    train_loss_mse, train_loss_, train_acc_,
                    train_cls_loss, train_reg_loss,
                    )
            )
        if args.tb:
            train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
            train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            train_tb_writer.add_scalar('train_loss', train_loss, epoch)
            train_tb_writer.add_scalar('train_loss_batch', train_loss_batch, epoch)
            train_tb_writer.add_scalar('train_loss_ce', train_loss_ce, epoch)
            train_tb_writer.add_scalar('train_loss_mse', train_loss_mse, epoch)
            train_tb_writer.add_scalar('train_loss_', train_loss_, epoch)
            train_tb_writer.add_scalar('train_acc_', train_acc_, epoch)
            train_tb_writer.add_scalar('train_cls_loss', train_cls_loss, epoch)
            train_tb_writer.add_scalar('train_reg_loss', train_reg_loss, epoch)


        writer.add_scalar('train_loss', train_loss_, epoch)
        writer.add_scalar('train_acc', train_acc_, epoch)
        lr_scheduler.step()  # ## Updating the learning rate for the next epoch

#########################################

        results = evaluate_cifar10dvs(test_loader, model, criterion_mse, num_classes, device, args)

        test_acc1, test_acc5 = results[:2]
        test_loss, test_loss_batch = results[2:4]
        test_loss_ce, test_loss_mse = results[4:6]
        test_loss_, test_acc_ = results[6:8]


        # ## Append logger file
        logger_train_test_acc.append(
            [epoch, cur_lr,
             train_acc1, train_acc5,
             test_acc1, test_acc5,
             ])

        logger_train_test_loss.append(
            [epoch, cur_lr,
             train_loss, train_loss_batch, train_loss_ce, train_loss_mse, train_loss_, train_acc_,
             train_cls_loss, train_reg_loss,
             test_loss, test_loss_batch, test_loss_ce, test_loss_mse, test_loss_, test_acc_,

             ])


        logger.info(
            'Test Epoch:[{}/{}]\t test_acc1={:.3f}\t test_acc5={:.3f}\t '
            'test_loss={:.5f}\t test_loss_batch={:.5f}\t test_loss_ce={:.5f}\t'
            'test_loss_mse={:.5f}\t test_loss_={:.5f}\t test_acc_={:.5f}\t'
            .format(epoch, args.epochs, test_acc1, test_acc5,
                    test_loss, test_loss_batch, test_loss_ce,
                    test_loss_mse, test_loss_, test_acc_,
                    )
            )
        print(
            'Test Epoch:[{}/{}]\t test_acc1={:.3f}\t test_acc5={:.3f}\t '
            'test_loss={:.5f}\t test_loss_batch={:.5f}\t test_loss_ce={:.5f}\t'
            'test_loss_mse={:.5f}\t test_loss_={:.5f}\t test_acc_={:.5f}\t\n'
            .format(epoch, args.epochs, test_acc1, test_acc5,
                    test_loss, test_loss_batch, test_loss_ce,
                    test_loss_mse, test_loss_, test_acc_,
                    )
            )

        if args.tb and test_tb_writer is not None:
            test_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
            test_tb_writer.add_scalar('test_acc5', test_acc5, epoch)
            test_tb_writer.add_scalar('test_loss', test_loss, epoch)
            test_tb_writer.add_scalar('test_loss_batch', test_loss_batch, epoch)
            test_tb_writer.add_scalar('test_loss_ce', test_loss_ce, epoch)
            test_tb_writer.add_scalar('test_loss_mse', test_loss_mse, epoch)
            test_tb_writer.add_scalar('test_loss_', test_loss_, epoch)
            test_tb_writer.add_scalar('test_acc_', test_acc_, epoch)


        writer.add_scalar('test_loss', test_loss_, epoch)
        writer.add_scalar('test_acc', test_acc_, epoch)




        save_max = False
        if max_test_acc1 < test_acc1:
            max_test_acc1 = test_acc1
            max_test_acc5 = test_acc5
            save_max = True

        if output_dir:
            checkpoint = {
                'model_arch': args.model,
                'dataset': args.dataset,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'max_test_acc1': max_test_acc1,
                'max_test_acc5': max_test_acc5,
                'epoch': epoch,
                'args': args,
            }

            torch.save(checkpoint, os.path.join(output_dir, f'{prefix_name}_checkpoint_latest.pth'))

            save_flag = False
            if epoch % 64 == 0 or epoch == args.epochs - 1:
                save_flag = True
            if save_flag:
                torch.save(checkpoint, os.path.join(output_dir, f'{prefix_name}_checkpoint_{epoch}.pth'))

            if save_max:
                torch.save(checkpoint, os.path.join(output_dir, f'{prefix_name}_checkpoint_max_test_acc1.pth'))
        ## ####
        # for item in sys.argv:
        #     print(item, end=' ')
        # print('')

        print(args)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print(total_time_str)
        print(output_dir)

        total_time = time.time() - start_time
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc1={train_acc1}, '
              f'test_loss={test_loss}, test_acc1={test_acc1}, '
              f'max_test_acc1={max_test_acc1}, total_time={total_time}, '
              f'escape_time={(datetime.datetime.now()+datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}'
              )

        print('Training time {}\t max_test_acc1: {} \t max_test_acc5: {} \n'
              .format(total_time_str, max_test_acc1, max_test_acc5))

        logger.info('Training time {}\t max_test_acc1: {} \t max_test_acc5: {}'
                    .format(total_time_str, max_test_acc1, max_test_acc5))
        logger.info('\n')


    # ## Outside the for loop, Finish write information
    logger_train_test_acc.close()
    logger_train_test_loss.close()
    print('* Done training')


    # ## Print the best accuracy
    print('================================ Training finished! ================================')
    print(f'Best test acc:   {max_test_acc1} ')

    # ## Plot the loss and accuracy
    logger_plot_save(logger_train_test_acc, logger_train_test_loss, prefix_name, plot_dir)


if __name__ == '__main__':

    args = parse_args()
    main(args)

    print('*** Done __main__')



    #