# -*- coding: utf-8 -*-


import time
import torch.nn.functional as F
from torch.cuda import amp

import torch


from utils import Bar
import data.utils as utils



def train(train_loader, model, optimizer, criterion_mse, num_classes, device, epoch, t_step, args, scaler=None):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')

    losses = utils.AverageMeter('Loss', ':6.4f')  # final loss used to BP
    top1 = utils.AverageMeter('Acc@1', ':6.3f')
    top5 = utils.AverageMeter('Acc@5', ':6.3f')

    losses_batch = utils.AverageMeter('Batch Loss', ':6.4f')  # batch loss
    losses_ce = utils.AverageMeter('CE Loss', ':6.4f')        # ce loss, of the final_output
    losses_mse = utils.AverageMeter('MSE Loss', ':6.4f')      # mse loss,

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, losses_batch, losses_ce, losses_mse],
        prefix="Epoch: [{}]".format(epoch))

    bar = Bar('Processing', max=len(train_loader))

    model.train()
    end = time.time()

    train_loss = 0.
    train_acc = 0.
    train_samples = 0
    batch_idx = 0

    # for i, (image, target) in enumerate(train_loader):
    #     # measure data loading time
    #     data_time.update(time.time() - end)

    for i, (frame, label) in enumerate(train_loader):
        # ## measure data loading time
        data_time.update(time.time() - end)
        batch_idx += 1

        frame = frame.float().to(device)
        label = label.to(device)
        label_one_hot = F.one_hot(label, num_classes).float()

        batch_loss = 0.
        if not args.online_update:
            optimizer.zero_grad()

        for t in range(t_step):
            if args.online_update:
                optimizer.zero_grad()

            input_frame = frame
            if args.amp:
                with amp.autocast():
                    if t == 0:
                        out_fr = model(input_frame, init=True)
                        total_fr = out_fr.clone().detach()
                    else:
                        out_fr = model(input_frame)
                        total_fr += out_fr.clone().detach()
                        # # total_fr = total_fr * (1 - 1. / args.tau) + out_fr

                    ce_loss = F.cross_entropy(out_fr, label)
                    mse_loss = torch.tensor(0.0).to(device)
                    if args.loss_lambda > 0.0:
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss = ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss = ce_loss / t_step
                scaler.scale(loss).backward()

                if args.online_update:
                    scaler.step(optimizer)
                    scaler.update()
            else:
                if t == 0:
                    out_fr = model(input_frame, init=True)
                    total_fr = out_fr.clone().detach()
                else:
                    out_fr = model(input_frame)
                    total_fr += out_fr.clone().detach()
                    # # total_fr = total_fr * (1 - 1. / args.tau) + out_fr

                ce_loss = F.cross_entropy(out_fr, label)
                mse_loss = torch.tensor(0.0).to(device)
                if args.loss_lambda > 0.0:
                    mse_loss = criterion_mse(out_fr, label_one_hot)
                    loss = ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
                else:
                    loss = ce_loss / t_step
                loss.backward()

                if args.online_update:
                    optimizer.step()

            batch_loss += loss.item()
            train_loss += loss.item() * label.numel()
        if not args.online_update:
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()


        # batch_size = frame.shape[0]

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(total_fr.data, label.data, topk=(1, 5))
        losses.update(loss.item(), input_frame.size(0))
        top1.update(prec1.item(), input_frame.size(0))
        top5.update(prec5.item(), input_frame.size(0))

        losses_batch.update(batch_loss, input_frame.size(0))
        losses_ce.update(ce_loss.item(), input_frame.size(0))
        losses_mse.update(mse_loss.item(), input_frame.size(0))

        train_samples += label.numel()
        train_acc += (total_fr.argmax(1) == label).float().sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            # Print the GPU memory used in bytes
            print(f'=== GPU memory used in bytes: {torch.cuda.memory_allocated()}')
            # Print the GPU memory cached in bytes
            print(f'=== the GPU memory cached in bytes: {torch.cuda.memory_reserved()}')

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '\
            'Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx,
                size=len(train_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
                )
        bar.next()
    bar.finish()

    train_loss /= train_samples
    train_acc /= train_samples

    results = [
        top1.avg, top5.avg, losses.avg, losses_batch.avg,
        losses_ce.avg, losses_mse.avg,
        train_loss, train_acc,
        ]
    return results



    # writer.add_scalar('train_loss', train_loss, epoch)
    # writer.add_scalar('train_acc', train_acc, epoch)
    # lr_scheduler.step()





def evaluate(test_loader, model, criterion_mse, num_classes, device, t_step, args):
    model.eval()

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Date Time', ':6.3f')

    losses = utils.AverageMeter('Loss', ':6.4f')  # final loss used to BP
    top1 = utils.AverageMeter('Acc@1', ':6.3f')
    top5 = utils.AverageMeter('Acc@5', ':6.3f')

    losses_batch = utils.AverageMeter('Batch Loss', ':6.4f')  # batch loss
    losses_ce = utils.AverageMeter('CE Loss', ':6.4f')        # ce loss, of the final_output
    losses_mse = utils.AverageMeter('MSE Loss', ':6.4f')      # mse loss,

    progress = utils.ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5, losses_batch, losses_ce, losses_mse],
        prefix='Test: ')
    bar = Bar('Processing', max=len(test_loader))
    end = time.time()

    test_loss = 0.
    test_acc = 0.
    test_samples = 0
    batch_idx = 0

    with torch.no_grad():
        for i, (frame, label) in enumerate(test_loader):
            data_time.update(time.time() - end)

            batch_idx += 1
            frame = frame.float().to(device)
            label = label.to(device)
            total_loss = 0

            for t in range(t_step):
                input_frame = frame
                if t == 0:
                    out_fr = model(input_frame, init=True)
                    total_fr = out_fr.clone().detach()
                else:
                    out_fr = model(input_frame)
                    total_fr += out_fr.clone().detach()
                    # ## total_fr = total_fr * (1 - 1. / args.tau) + out_fr
                ce_loss = F.cross_entropy(out_fr, label)
                mse_loss = torch.tensor(0.0).to(device)
                if args.loss_lambda > 0.0:
                    label_one_hot = F.one_hot(label, num_classes).float()
                    mse_loss = criterion_mse(out_fr, label_one_hot)
                    loss = ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
                else:
                    loss = ce_loss / t_step
                total_loss += loss

            test_samples += label.numel()
            test_loss += total_loss.item() * label.numel()
            test_acc += (total_fr.argmax(1) == label).float().sum().item()

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(loss.item(), input_frame.size(0))
            top1.update(prec1.item(), input_frame.size(0))
            top5.update(prec5.item(), input_frame.size(0))

            losses_batch.update(total_loss, input_frame.size(0))
            losses_ce.update(ce_loss.item(), input_frame.size(0))
            losses_mse.update(mse_loss.item(), input_frame.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | '\
                ' Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
            bar.next()

    bar.finish()


    # TODO: this should also be done with the ProgressMeter
    print(' * Test Acc@1 {top1.avg:.3f} Test Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    test_loss /= test_samples
    test_acc /= test_samples

    results = [
        top1.avg, top5.avg, losses.avg, losses_batch.avg,
        losses_ce.avg, losses_mse.avg,
        test_loss, test_acc,
        ]
    return results