import pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

import pdb
import subprocess

# from FCN import FCNNet
from Network import U_Net as FCNNet
#from Network3 import U_Net_FP as FCNNet

from ufold.utils import *
from ufold.config import process_config
from ufold.postprocess import postprocess_new as postprocess

from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
import collections


def train(model_path, contact_net, train_merge_generator, epoches_first, device):
    epoch = 0
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    u_optimizer = optim.Adam(contact_net.parameters())
    
    steps_done = 0
    print('start training...')
    # There are three steps of training
    # step one: train the u net
    epoch_rec = []
    for epoch in range(epoches_first):
        contact_net.train()
   
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
   
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            
            pred_contacts = contact_net(seq_embedding_batch)
    
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
    
            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
    
            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done=steps_done+1
    
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
                    epoch, steps_done-1, loss_u))
        if epoch > -1:
            torch.save(contact_net.state_dict(),  f'{model_path}/ufold_train_{epoch}.pt')

def main():
    
    args = get_args()
    
    config_file = args.config
    
    config = process_config(config_file)
    print("#####Stage 1#####")
    print('Here is the configuration of this run: ')
    print(config)
            
    BATCH_SIZE = config.batch_size_stage_1
    epoches_first = config.epoches_first

    train_files = args.train_files
    model_path = args.model_path
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    device = config.device
    
    seed_torch()
    
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
    
    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ',file_item)
        if file_item == 'RNAStralign' or file_item == 'ArchiveII':
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
    print('Data Loading Done!!!')
    
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}

    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    
    contact_net = FCNNet(img_ch=17)
    contact_net.to(device)

    train(model_path, contact_net, train_merge_generator, epoches_first, device)

if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()
