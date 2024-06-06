# -*- coding: utf-8 -*-


import time
import torch.nn.functional as F
from torch.cuda import amp

import torch


from utils import Bar
import data.utils as utils



# train_loader, model, optimizer, criterion_mse, num_classes, device, epoch, t_step, args,

# ### No t_step


def train_cifar10dvs(train_loader, model, optimizer, criterion_mse, num_classes, device, epoch, args, scaler=None):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':6.4f')  # final loss used to BP
    top1 = utils.AverageMeter('Acc@1', ':6.3f')
    top5 = utils.AverageMeter('Acc@5', ':6.3f')

    losses_batch = utils.AverageMeter('Batch Loss', ':6.4f')  # batch loss
    losses_ce = utils.AverageMeter('CE Loss', ':6.4f')        # ce loss, of the final_output
    losses_mse = utils.AverageMeter('MSE Loss', ':6.4f')      # mse loss,

    losses_cls = utils.AverageMeter('CLS Loss', ':6.4f')       #  classify loss
    losses_reg = utils.AverageMeter('REG Loss', ':6.4f')       # reg loss,

    progress = utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5,
         losses_batch, losses_ce, losses_mse,
         losses_cls, losses_reg,
         ],
        prefix="Epoch: [{}]".format(epoch))



    bar = Bar('Processing', max=len(train_loader))

    model.train()
    end = time.time()

    train_loss = 0.
    train_acc = 0.
    train_samples = 0
    batch_idx = 0

    # # # ### Save the current model parameters for the next iteration
    previous_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}

    for i, (frame, label) in enumerate(train_loader):
        # ## measure data loading time
        data_time.update(time.time() - end)
        batch_idx += 1

        # frame = frame.float().cuda()
        # label = label.cuda()
        frame = frame.float().to(device)
        label = label.to(device)
        label_one_hot = F.one_hot(label, num_classes).float()

        t_step = frame.shape[1]

        # if args.T_train:
        #     sec_list = np.random.choice(frame.shape[1], args.T_train, replace=False)
        #     sec_list.sort()
        #     frame = frame[:, sec_list]
        #     t_step = args.T_train

        batch_loss = 0.
        if not args.online_update:
            optimizer.zero_grad()

        # # # ### Save the current model parameters for the next iteration
        previous_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}

        for t in range(t_step):
            if args.online_update:
                optimizer.zero_grad()
            # ### This part is very different with cifar10 and cifar100
            input_frame = frame[:, t]
            if args.amp:
                with amp.autocast():
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
                        mse_loss = criterion_mse(out_fr, label_one_hot)
                        loss_classify = ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
                    else:
                        loss_classify = ce_loss / t_step

                    reg_loss = torch.tensor(0.0).to(device)
                    if args.online_update:
                        # ### Compute the loss term based on the parameter difference
                        if t == 0:
                            reg_loss = torch.tensor(0.0).to(device)
                        else:
                            # ### Calculate the L2 norm of the parameter difference
                            for name, param in model.named_parameters():
                                parameter_diff = torch.sub(param, previous_parameters[name])
                                temp = torch.sqrt(torch.tensor(1.0*param.numel()).to(device) )
                                # reg_loss += torch.norm(parameter_diff, p=1) / temp      #  _L1_avg_
                                reg_loss += torch.norm(parameter_diff, p=2) / temp      #  _sqrt_avg_
                                # reg_loss += torch.norm(parameter_diff, p=2) / param.numel()
                                # reg_loss += torch.norm(parameter_diff, p=2)   # ##  torch.sqrt(torch.sum(diff **2 ) )

                                # ##  reg_loss += torch.norm(param - previous_parameters[name], p=2) / param.numel()
                                # if param.requires_grad and (param.grad is not None):
                                #     reg_loss += torch.norm(param - previous_parameters[name], p=2) / param.numel()
                        # ## loss =  (1 - args.reg_lambda) * loss_classify + args.reg_lambda * reg_loss
                        loss =  loss_classify + 0.5*args.reg_lambda * reg_loss
                    else:
                        loss =  loss_classify



                # # # ### Save the current model parameters for the next iteration
                previous_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}

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
                    # ## total_fr = total_fr * (1 - 1. / args.tau) + out_fr

                ce_loss = F.cross_entropy(out_fr, label)
                mse_loss = torch.tensor(0.0).to(device)
                if args.loss_lambda > 0.0:
                    mse_loss = criterion_mse(out_fr, label_one_hot)
                    loss_classify = ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
                else:
                    loss_classify = ce_loss / t_step

                # ## New added reg_loss
                reg_loss = torch.tensor(0.0).to(device)
                if args.online_update:
                    # ### Compute the loss term based on the parameter difference
                    if t == 0:
                        reg_loss = torch.tensor(0.0).to(device)
                    else:
                        # ### Calculate the L2 norm of the parameter difference
                        for name, param in model.named_parameters():
                            parameter_diff = torch.sub(param, previous_parameters[name])
                            temp = torch.sqrt(torch.tensor(1.0*param.numel()).to(device) )
                            # reg_loss += torch.norm(parameter_diff, p=1) / temp      #  _L1_avg_
                            reg_loss += torch.norm(parameter_diff, p=2) / temp      #  _v_sqrt_
                            # reg_loss += torch.norm(parameter_diff, p=2) / param.numel()
                            # reg_loss += torch.norm(parameter_diff, p=2)   # ##  torch.sqrt(torch.sum(diff **2 ) )

                            # ##  reg_loss += torch.norm(param - previous_parameters[name], p=2) / param.numel()
                            # if param.requires_grad and (param.grad is not None):
                            #     reg_loss += torch.norm(param - previous_parameters[name], p=2) / param.numel()
                    # ## loss =  (1 - args.reg_lambda) * loss_classify + args.reg_lambda * reg_loss
                    loss =  loss_classify + 0.5*args.reg_lambda * reg_loss
                else:
                    loss =  loss_classify

                # # # ### Save the current model parameters for the next iteration
                previous_parameters = {name: param.clone().detach() for name, param in model.named_parameters()}

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

        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(total_fr.data, label.data, topk=(1, 5))
        losses.update(batch_loss, input_frame.size(0))
        top1.update(prec1.item(), input_frame.size(0))
        top5.update(prec5.item(), input_frame.size(0))

        losses_batch.update(batch_loss, input_frame.size(0))
        losses_ce.update(ce_loss.item(), input_frame.size(0))
        losses_mse.update(mse_loss.item(), input_frame.size(0))

        losses_cls.update(loss_classify.item(), input_frame.size(0))
        losses_reg.update(reg_loss.item(), input_frame.size(0))

        train_samples += label.numel()
        train_acc += (total_fr.argmax(1) == label).float().sum().item() * 100.0

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            progress.display(i)

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s |'\
            ' Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
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
        losses_cls.avg, losses_reg.avg,
        ]
    return results




def evaluate_cifar10dvs(test_loader, model, criterion_mse, num_classes, device, args):
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
            # frame = frame.float().cuda()
            # label = label.cuda()

            frame = frame.float().to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, num_classes).float()

            total_loss = 0.
            t_step = frame.shape[1]

            for t in range(t_step):
                input_frame = frame[:, t]

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
                    mse_loss = criterion_mse(out_fr, label_one_hot)
                    loss = ((1 - args.loss_lambda) * ce_loss + args.loss_lambda * mse_loss) / t_step
                else:
                    loss = ce_loss / t_step
                total_loss += loss

            test_samples += label.numel()
            test_loss += total_loss.item() * label.numel()
            test_acc += (total_fr.argmax(1) == label).float().sum().item() * 100.

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(total_fr.data, label.data, topk=(1, 5))
            losses.update(total_loss, input_frame.size(0))
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
                'Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(

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