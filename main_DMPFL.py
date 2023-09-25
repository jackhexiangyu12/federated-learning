#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


def getBinaryTensor(imgTensor, boundary=200):
    one = torch.ones_like(imgTensor)
    zero = torch.zeros_like(imgTensor)
    return torch.where(imgTensor > boundary, one, zero)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    net_client = copy.deepcopy(net_glob)

    # copy weights
    w_g = net_glob.state_dict()

    sparsity_ratio = 0.5
    readjust_masks = True
    a_s = 0.01
    mask_readjust_interval = 1

    # 初始化，把 w_glob中的参数的绝对值中的前sparsity_ratio%的参数置为1，其余的置为0
    m_g = copy.deepcopy(w_g)
    for k in m_g.keys():
        # mid是w_1[k]中绝对值第sparsity_ratio%大的数
        mid = torch.topk(m_g[k].abs().view(-1), int(sparsity_ratio * m_g[k].numel()))[0][-1]
        m_g[k] = getBinaryTensor(m_g[k].abs(), boundary=mid)

    # 初始化，把 m_c[idx]都设置为m_g
    m_c = [copy.deepcopy(m_g) for i in range(args.num_users)]
    w_c = [copy.deepcopy(w_g) for i in range(args.num_users)]

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # if args.all_clients:
    #     print("Aggregation over all clients")
    #     w_c = [w_glob for i in range(args.num_users)]
    for iteration in range(1):
        #大的iteration包含Train Dual Mask和Refine Dual Mask两个小的iteration
        for iter in range(args.epochs):
            loss_locals = []
            # if not args.all_clients:
            #     w_c = []
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)

            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])

                for k in w_c[idx].keys():
                    w_c_old=w_c[idx][k]
                    w_c[idx][k] = w_g[k] * (m_g[k] * m_c[idx][k]) + w_c[idx][k] * (1 - m_c[idx][k] * m_g[k])
                    # 比较w_c[idx]与w_c_old的差异
                    w_diff = w_c[idx][k] - w_c_old
                    #计算w_diff中有百分之多少不是0
                    w_diff_ratio = torch.sum(torch.abs(w_diff)) / w_diff.numel()

                wm_c_temp = copy.deepcopy(w_c[idx])
                for k in wm_c_temp.keys():
                    wm_c_temp[k] = wm_c_temp[k] * m_c[idx][k]
                net_client.load_state_dict(wm_c_temp)
                w_c_temp_old=wm_c_temp
                w_c_temp, loss = local.train(net=copy.deepcopy(net_client).to(args.device))
                for k in w_c[idx].keys():
                    w_c[idx][k] = w_c_temp[k] * m_c[idx][k] + w_c[idx][k] * (1 - m_c[idx][k])
                    #计算训练前后的差距
                    w_c_temp_differ=w_c_temp[k]-w_c_temp_old[k]


                if args.all_clients:
                    # w_c[idx] = copy.deepcopy(g_c)
                    # 报错，不知道怎么写：
                    exit('Error:todo')
                loss_locals.append(copy.deepcopy(loss))

                #比较w_c与w_g的差异
                # for i in range(len(w_c)):
                #     # 比较w_locals与w_glob的差异
                #     w_diff = 0
                #     for k in w_g.keys():
                #         w_diff += torch.sum(torch.abs(w_g[k] - w_c[i][k]))
                #     print("the difference between w_local ", i, " and w_glob: ", w_diff)

                # 判断是否readjust_masks=true:
                if readjust_masks == True:
                    if iter % mask_readjust_interval == 0:
                        for k in m_c[idx].keys():
                            m_c_old=m_c[idx][k]
                            # locate_min是w_1[k]中绝对值第a_s % 小的所有数的位置
                            locate_min = \
                                torch.topk((w_c[idx][k]*(1-m_c[idx][k])).abs().view(-1), int(a_s * w_c[idx][k].numel()), largest=False)[1]
                            # locate_max是w_1[k]中绝对值第a_s % 大的所有数的位置
                            locate_max = \
                                torch.topk((w_c[idx][k]*(1-m_c[idx][k])).abs().view(-1), int(a_s * w_c[idx][k].numel()), largest=True)[1]
                            #将m_c[idx][k]展开到一维然后还原
                            m_c_shape=m_c[idx][k].shape
                            m_c[idx][k] = m_c[idx][k].view(-1)
                            # 把m_c[idx]中locate_min中的位置置为0
                            m_c[idx][k]=torch.scatter(m_c[idx][k], 0, locate_min, 0)
                            # m_c[idx][k][locate_min] = 0
                            # 把m_c[idx]中locate_max中的位置置为1
                            m_c[idx][k]=torch.scatter(m_c[idx][k], 0, locate_max, 1)
                            # m_c[idx][k][locate_max] = 1

                            m_c[idx][k] = m_c[idx][k].view(m_c_shape)
                            m_c_diff=m_c[idx][k]-m_c_old
                            # print()

            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
            w_old_glob=w_g
            #w_c_select是w_c中下标为idxs_users的元素
            w_c_select=[w_c[i] for i in idxs_users]
            w_g = FedAvg(w_c_select)
            # 比较w_old_glob和w_glob的差异
            w_diff = 0
            for k in w_g.keys():
                w_diff += torch.sum(torch.abs(w_g[k] - w_old_glob[k]))
            print("w_diff: ", w_diff)
            # 计算w_old_glob和w_glob的相似度
            w_sim = 0
            for k in w_g.keys():
                w_sim += torch.sum(torch.abs(w_g[k] * w_old_glob[k]))
            print("w_sim: ", w_sim)
            # for idx in range(args.num_users):
            #     # 计算m_c中为1的元素占所有元素的占比
            #     m_c_ratio = 0
            #     for k in m_c[idx].keys():
            #         m_c_ratio += torch.sum(m_c[idx][k])
            #         m_c_ratio = m_c_ratio / m_c[idx][k].numel()
            #         print(idx,',',k,":m_c_ratio: ", m_c_ratio)
        print('algorithm 1')
        net_glob.load_state_dict(w_g)
        net_glob.eval()
        acc_train, loss_train1 = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

        for iter in range(args.epochs):
            # algorithm 2
            w_g = copy.deepcopy(w_g)
            for k in w_g.keys():
                w_g[k] = w_g[k] * m_g[k]
            wm_g_temp = copy.deepcopy(w_g)
            for k in wm_g_temp.keys():
                wm_g_temp[k] = wm_g_temp[k] * m_g[k]
            net_client.load_state_dict(wm_g_temp)
            w_g, loss = local.train(net=copy.deepcopy(net_client).to(args.device))
            for k in w_c[idx].keys():
                w_g[k] = w_g[k] * m_c[idx][k] + w_c[idx][k] * (1 - m_c[idx][k])
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
            w_g = FedAvg(w_c)
        print('algorithm 2')
        net_glob.load_state_dict(w_g)
        net_glob.eval()
        acc_train, loss_train1 = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

        for iter in range(args.epochs):
            # algorithm 3
            w_c[idx] = copy.deepcopy(w_g)
            for k in w_c[idx].keys():
                w_c[idx][k] = w_c[idx][k] * m_g[k]
                w_c[idx][k] = w_c[idx][k] * m_c[idx][k]
            cita_c = copy.deepcopy(w_g)
            for k in cita_c.keys():
                cita_c[k] = cita_c[k] * m_g[k]
                cita_c[k] = cita_c[k] * m_c[idx][k]
                cita_c[k] = cita_c[k] + w_c[idx][k] * (m_c[idx][k] - m_g[k])

            net_client.load_state_dict(cita_c)
            w_c_temp, loss = local.train(net=copy.deepcopy(net_client).to(args.device))
            for k in w_c[idx].keys():
                w_c[idx][k] = w_c_temp[k] * (m_c[idx][k] - m_g[k]) + w_c[idx][k] * m_g[k]
            loss_avg = sum(loss_locals) / len(loss_locals)
            print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
            loss_train.append(loss_avg)
        print('algorithm 3')
        net_glob.load_state_dict(w_g)
        net_glob.eval()
        acc_train, loss_train1 = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    # testing 每个客户都测试一下，acc_train与acc_test都是所有客户的平均值

    # net_glob.load_state_dict(w_g)
    # net_glob.eval()
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    # acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    # print("Testing accuracy: {:.2f}".format(acc_test))
