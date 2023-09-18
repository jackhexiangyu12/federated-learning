#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def DM_PFL(mc,wc,mg,wg):
    wc_next=copy.deepcopy(wc)
    wc_next=wg(mg)

    for k in wc.keys():
        wc_shape = wc.shape
        wc_next = wc_next.reshape(-1)
        mg = mg.reshape(-1)
        mc = mc.reshape(-1)
        wg = wg.reshape(-1)

        # 遍历一维张量
        dim0 = wc_next.shape
        for i in range(dim0):
            if (mg[i] * mc[i] == 1):
                wc_next[i] = wg[i]
        wc_next = wc_next.reshape(wc_shape)
        mc = mc.reshape(wc_shape)

    return wc_next,mc
