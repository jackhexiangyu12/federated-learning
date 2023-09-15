#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def DM_PFL(mc,wc,mg,wg):
    wc_next=copy.deepcopy(wc)
    wc_next=wg(mg)

    # w_avg = copy.deepcopy(w[0])
    for k in wc.keys():
        # wc[k]
        # for i in range(1, len(wc)):#遍历n个客户，此处不需要，只需要遍历每一个tensor
        wc_shape = wc.shape
        wc_next = wc_next.reshape(-1)
        mg = mg.reshape(-1)
        mc = mc.reshape(-1)
        wg = wg.reshape(-1)
        # a = torch.tensor([[0.0, 0.1, 0.1], [1.0, 1.5, 1.4], [2.1, 2.3, 2.4], [3.9, 3.9, 3.8]])

        # 对张量进行reshape，转成一维张量
        # b = a.reshape(-1)

        # 遍历一维张量
        dim0 = wc_next.shape
        for i in range(dim0):
            if (mg[i] * mc[i] == 1):
                wc_next[i] = wg[i]
        wc_next = wc_next.reshape(wc_shape)
        mc = mc.reshape(wc_shape)


        #     w_avg[k] += w[i][k]
        # w_avg[k] = torch.div(w_avg[k], len(w))
    return wc_next,mc
