#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:44:10 2026

@author: remi
"""

import scanpy as sc

from config.config import H5AD
from data_io import load_dataset_from_config_and_gate


def build_all_bcells(cfg):
    adatas = []
    for ds_name, ds_cfg in cfg['datasets'].items():
        ad = load_dataset_from_config_and_gate(ds_name, ds_cfg)
        adatas.append(ad)
    ad_all = sc.concat(adatas, join='outer', label='dataset', index_unique='-')
    ad_all.write(H5AD['ALL_Bcells'])
    return ad_all


def run_umap(ad):
    sc.pp.highly_variable_genes(ad, flavor='seurat_v3', n_top_genes=3000)
    ad = ad[:, ad.var['highly_variable']].copy()
    sc.pp.scale(ad, max_value=10)
    sc.tl.pca(ad, n_comps=50)
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=30)
    sc.tl.umap(ad)
    return ad