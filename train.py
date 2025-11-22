# -*- coding: utf-8 -*-
# @Author  : Alisa
# @File    : main(pretrain).py
# @Software: PyCharm
import warnings
from evaluate import evaluate, train_test_split_few, context_inference
from sklearn.metrics import f1_score, roc_auc_score
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import torch
import itertools
from torch.optim import Adam
from pargs import pargs
from load_data import load_datasets_with_prompts, TreeDataset, HugeDataset, TreeDataset_PHEME, TreeDataset_UPFD, \
    CovidDataset
from model import BiGCN_graphcl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from augmentation import augment
from torch_geometric import seed_everything

warnings.filterwarnings("ignore")


def pre_train(loaders, aug1, aug2, model, optimizer, device):
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    # for i, batches in enumerate(itertools.zip_longest(*loaders, fillvalue=None)):
    for loader in loaders:
        for batch in loader:
            optimizer.zero_grad()

            # augmented_data1 = []
            # augmented_data2 = []
            # for idx, batch in enumerate(batches):
            #     if batch is not None:  # Ensure the batch is not None (handle shorter datasets)
            #         batch = batch.to(device)
            #         aug_data1 = augment(batch, augs1)
            #         aug_data2 = augment(batch, augs2)
            #         augmented_data1.append(aug_data1)
            #         augmented_data2.append(aug_data2)
            #############################################
            batch = batch.to(device)
            augmented_data1 = augment(batch, augs1)
            augmented_data2 = augment(batch, augs2)

            # Model forward pass
            out1 = model(augmented_data1)
            out2 = model(augmented_data2)

            # Compute the loss using the contrastive loss function
            loss = model.loss_graphcl(out1, out2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
    total_loss /= len(loaders)
    # scheduler.step()
    return total_loss


def pre_trains(loaders, aug1, aug2, model, optimizer, device):
    model.train()
    total_loss = 0

    augs1 = aug1.split('||')
    augs2 = aug2.split('||')

    for data in loaders:
        optimizer.zero_grad()
        data = data.to(device)

        aug_data1 = augment(data, augs1)
        aug_data2 = augment(data, augs2)

        out1 = model(aug_data1)
        out2 = model(aug_data2)
        loss = model.loss_graphcl(out1, out2)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    total_loss /= len(loaders)
    return total_loss


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_millions(num):
    return round(num / 1e6, 2)


def finetune_and_evaluate(model, target_data, fewshot_k, batch_size, finetune_epochs, args, device, seed):
    model.freeze_backbone()

    # 优化器只针对非冻结参数
    lr = 0.01
    finetune_optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                              weight_decay=args.weight_decay)

    # 损失
    criterion = torch.nn.CrossEntropyLoss()

    # 数据split
    if args.dataset in ['WeiboCOVID19', 'TwitterCOVID19']:
        all_labels = np.array([sample.y.item() for sample in target_data])
    else:
        all_labels = target_data.y.cpu().numpy()

    mask = train_test_split_few(all_labels, seed=0, train_examples_per_class=fewshot_k, val_size=500, test_size=None)
    train_mask = mask['train'].astype(bool)
    val_mask = mask['val'].astype(bool)
    test_mask = mask['test'].astype(bool)

    train_loader = DataLoader(target_data[train_mask], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(target_data[val_mask], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(target_data[test_mask], batch_size=batch_size, shuffle=False)

    # 微调
    best_val_loss = float('inf')
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model.finetune(batch)
            labels = batch.y.long()
            loss = criterion(logits, labels)
            # print(loss.item())
            finetune_optimizer.zero_grad()
            loss.backward()
            finetune_optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits = model.finetune(batch)
                labels = batch.y.long()
                loss = criterion(logits, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            # torch.save(model.state_dict(), f"./{args.dataset}_finetuned_seed{seed}.pt")

    # model.load_state_dict(torch.load(f"./{args.dataset}_finetuned_seed{seed}.pt", map_location=device))
    model.load_state_dict(best_model)
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model.finetune(batch)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = batch.y.cpu().numpy()
            test_preds.extend(preds)
            test_probs.extend(probs)
            test_labels.extend(labels)

    micro_f1 = f1_score(test_labels, test_preds, average='micro')
    macro_f1 = f1_score(test_labels, test_preds, average='macro')
    auc = roc_auc_score(test_labels, test_probs)

    return micro_f1, macro_f1, auc


if __name__ == '__main__':

    f1_macros_5 = []
    f1_macros_1 = []
    args = pargs()
    seed_everything(0)
    dataset = args.dataset
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    batch_size = args.batch_size

    weight_decay = args.weight_decay
    epochs = args.epochs

    # Initialize datasets
    # data = TreeDataset("./Data/DRWeiboV3/")
    # data = TreeDataset("../ACL/Data/Weibo/")
    # data = TreeDataset("./Data/Twitter15-tfidf/")
    # data = TreeDataset("./Data/Twitter16-tfidf/")
    # data = TreeDataset_PHEME("../ACL/Data/pheme/")
    # data = TreeDataset_UPFD("./Data/politifact/")
    # data = TreeDataset_UPFD("./Data/gossipcop/")
    # data = CovidDataset("../ACL/Data/Twitter-COVID19/Twittergraph")
    # data = CovidDataset("./Data/Weibo-COVID19/Weibograph")
    # data = HugeDataset("./Data/Tree/")
    # train_loaders = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=48)
    # target_loader = DataLoader(data, batch_size=32, shuffle=False)

    train_loaders, target_data, _ = load_datasets_with_prompts(args)

    # Model and optimizer initialization
    model = BiGCN_graphcl(768, args.out_feat).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    for epoch in range(1, epochs + 1):
        pretrain_loss = pre_trains(train_loaders,
                                  args.aug1, args.aug2, model, optimizer, device)
        # scheduler.step()
        print(f"Epoch: {epoch}, loss: {pretrain_loss}")
    # # torch.save(model.state_dict(), f"./{dataset}_concat.pt")
    print(args.dataset)

    fewshot_k = 1
    micro_list, macro_list, auc_list = [], [], []

    for run in range(1):
        # model.load_state_dict(torch.load(f"./{args.dataset}_prompt.pt", map_location=device))

        micro, macro, auc = finetune_and_evaluate(model, target_data, fewshot_k, batch_size, finetune_epochs=50,
                                                  args=args, device=device, seed=run)
        micro_list.append(micro)
        macro_list.append(macro)
        auc_list.append(auc)

    print(f"Average Micro F1: {np.mean(micro_list):.4f}, Std = {np.std(micro_list):.4f}")
    print(f"Average Macro F1: {np.mean(macro_list):.4f}, Std = {np.std(macro_list):.4f}")
    print(f"Average AUC: {np.mean(auc_list):.4f}, Std = {np.std(auc_list):.4f}")
    # model.eval()
    # x_list = []
    # y_list = []
    # for data in target_data:
    #     data = data.to(device)
    #     embeds = model.get_embeds(data)
    #     y_list.append(data.y)
    #     x_list.append(embeds.detach())
    # x = torch.cat(x_list, dim=0)
    # y = torch.cat(y_list, dim=0)
    # ################################################################################################
    # for r in [1, 5]:
    #     mask = train_test_split_few(y.cpu().numpy(), seed=0,
    #                                 train_examples_per_class=r,
    #                                 val_size=500, test_size=None)
    #     train_mask_l = f"{r}_train_mask"
    #     train_mask = mask['train'].astype(bool)
    #     val_mask_l = f"{r}_val_mask"
    #     val_mask = mask['val'].astype(bool)
    #
    #     test_mask_l = f"{r}_test_mask"
    #     test_mask = mask['test'].astype(bool)
    #
    #     evaluate(x, y, device, 0.01, 0.0, train_mask, val_mask, test_mask)

# Weibo
# Average Micro F1: 0.7398, Std = 0.0000
# Average Macro F1: 0.7397, Std = 0.0000
# Average AUC: 0.8201, Std = 0.0000
# Politifact
# Average Micro F1: 0.6934, Std = 0.0000
# Average Macro F1: 0.6933, Std = 0.0000
# Average AUC: 0.7393, Std = 0.0000
# Gossipcop
# Average Micro F1: 0.7701, Std = 0.0000
# Average Macro F1: 0.7670, Std = 0.0000
# Average AUC: 0.8800, Std = 0.0000
# DRWeiboV3
# Average Micro F1: 0.7821, Std = 0.0000
# Average Macro F1: 0.7819, Std = 0.0000
# Average AUC: 0.8813, Std = 0.0000
# PHEME
# Average Micro F1: 0.9670, Std = 0.0000
# Average Macro F1: 0.9644, Std = 0.0000
# Average AUC: 0.9935, Std = 0.0000
# WeiboCOVID19
# Average Micro F1: 0.6229, Std = 0.0000
# Average Macro F1: 0.6095, Std = 0.0000
# Average AUC: 0.6693, Std = 0.0000
# TwitterCOVID19
# Average Micro F1: 0.5940, Std = 0.0000
# Average Macro F1: 0.5911, Std = 0.0000
# Average AUC: 0.6298, Std = 0.0000


# Concat
# DRWeiboV3
# Average Micro F1: 0.7910, Std = 0.0000
# Average Macro F1: 0.7910, Std = 0.0000
# Average AUC: 0.8908, Std = 0.0000
# Weibo
# Average Micro F1: 0.8208, Std = 0.0000
# Average Macro F1: 0.8206, Std = 0.0000
# Average AUC: 0.9061, Std = 0.0000
# WeiboCOVID19
# Average Micro F1: 0.6801, Std = 0.0000
# Average Macro F1: 0.6613, Std = 0.0000
# Average AUC: 0.7129, Std = 0.0000
# TwitterCOVID19
# Average Micro F1: 0.6040, Std = 0.0000
# Average Macro F1: 0.6014, Std = 0.0000
# Average AUC: 0.6626, Std = 0.0000
# PHEME
# Average Micro F1: 0.9642, Std = 0.0002
# Average Macro F1: 0.9616, Std = 0.0002
# Average AUC: 0.9944, Std = 0.0001
# Politifact
# Average Micro F1: 0.7264, Std = 0.0000
# Average Macro F1: 0.7258, Std = 0.0000
# Average AUC: 0.7885, Std = 0.0000
# Gossipcop
# Average Micro F1: 0.9191, Std = 0.0006
# Average Macro F1: 0.9190, Std = 0.0006
# Average AUC: 0.9707, Std = 0.0002



# GraphPrompt 768*x
# WeiboCOVID19
# Average Micro F1: 0.5926, Std = 0.0000
# Average Macro F1: 0.5714, Std = 0.0000
# Average AUC: 0.5965, Std = 0.0000
# TwitterCOVID19
# Average Micro F1: 0.4933, Std = 0.0000
# Average Macro F1: 0.4912, Std = 0.0000
# Average AUC: 0.5841, Std = 0.0000


# 1-shot Politifact
# Average Micro F1: 0.7264, Std = 0.0000
# Average Macro F1: 0.7258, Std = 0.0000
# Average AUC: 0.7885, Std = 0.0000
# 2
# Average Micro F1: 0.7377, Std = 0.0025
# Average Macro F1: 0.7368, Std = 0.0024
# Average AUC: 0.8255, Std = 0.0004
# 5
# Average Micro F1: 0.7377, Std = 0.0025
# Average Macro F1: 0.7368, Std = 0.0024
# Average AUC: 0.8255, Std = 0.0004
# 10-shot
# Average Micro F1: 0.7856, Std = 0.0025
# Average Macro F1: 0.7856, Std = 0.0025
# Average AUC: 0.8409, Std = 0.0030
# 15-shot
# Average Micro F1: 0.8315, Std = 0.0000
# Average Macro F1: 0.8314, Std = 0.0000
# Average AUC: 0.8621, Std = 0.0001

# DRWeibo
# 1
# Average Micro F1: 0.7910, Std = 0.0000
# Average Macro F1: 0.7910, Std = 0.0000
# Average AUC: 0.8908, Std = 0.0000
# 2
# Average Micro F1: 0.7867, Std = 0.0000
# Average Macro F1: 0.7867, Std = 0.0000
# Average AUC: 0.8954, Std = 0.0000
# 5
# Average Micro F1: 0.7984, Std = 0.0000
# Average Macro F1: 0.7984, Std = 0.0000
# Average AUC: 0.8947, Std = 0.0000
# 10
# Average Micro F1: 0.8021, Std = 0.0000
# Average Macro F1: 0.8021, Std = 0.0000
# Average AUC: 0.8973, Std = 0.0000
# 15
# Average Micro F1: 0.7975, Std = 0.0000
# Average Macro F1: 0.7975, Std = 0.0000
# Average AUC: 0.8952, Std = 0.0000


# Gossipcop
# Average Micro F1: 0.9191, Std = 0.0006
# Average Macro F1: 0.9190, Std = 0.0006
# Average AUC: 0.9707, Std = 0.0002
# 2-shot
# Average Micro F1: 0.9327, Std = 0.0000
# Average Macro F1: 0.9326, Std = 0.0000
# Average AUC: 0.9776, Std = 0.0000
# 5
# Average Micro F1: 0.9338, Std = 0.0000
# Average Macro F1: 0.9337, Std = 0.0000
# Average AUC: 0.9787, Std = 0.0000
# 10
# Average Micro F1: 0.9341, Std = 0.0000
# Average Macro F1: 0.9340, Std = 0.0000
# Average AUC: 0.9805, Std = 0.0000
# 15
# Average Micro F1: 0.9325, Std = 0.0000
# Average Macro F1: 0.9325, Std = 0.0000
# Average AUC: 0.9805, Std = 0.0000

# DRWeiboV3
# Average Micro F1: 0.7910, Std = 0.0000
# Average Macro F1: 0.7910, Std = 0.0000
# Average AUC: 0.8908, Std = 0.0000
# Target:DRWeibo
# no Weibo
# Average Micro F1: 0.7483, Std = 0.0000
# Average Macro F1: 0.7476, Std = 0.0000
# Average AUC: 0.8551, Std = 0.0000
# no Weibo, W19
# Average Micro F1: 0.7160, Std = 0.0000
# Average Macro F1: 0.7156, Std = 0.0000
# Average AUC: 0.8184, Std = 0.0000
# no Weibo, W19, PHEME
# Average Micro F1: 0.7176, Std = 0.0000
# Average Macro F1: 0.7172, Std = 0.0000
# Average AUC: 0.8214, Std = 0.0000
# no Weibo, W19, PHEME, Politifact
# Average Micro F1: 0.7205, Std = 0.0000
# Average Macro F1: 0.7199, Std = 0.0000
# Average AUC: 0.8282, Std = 0.0000
# no Weibo, W19, PHEME, Politifact, Gossipcop
# Average Micro F1: 0.7012, Std = 0.0000
# Average Macro F1: 0.7009, Std = 0.0000
# Average AUC: 0.8024, Std = 0.0000


# Weibo
# Average Micro F1: 0.8208, Std = 0.0000
# Average Macro F1: 0.8206, Std = 0.0000
# Average AUC: 0.9061, Std = 0.0000
# Target Weibo
# no DRWeibo
# Average Micro F1: 0.7556, Std = 0.0000
# Average Macro F1: 0.7551, Std = 0.0000
# Average AUC: 0.8597, Std = 0.0000
# no DRWeibo W19
# Average Micro F1: 0.7309, Std = 0.0000
# Average Macro F1: 0.7307, Std = 0.0000
# Average AUC: 0.8281, Std = 0.0000
# no DRWeibo W19 PHEME
# Average Micro F1: 0.7732, Std = 0.0000
# Average Macro F1: 0.7724, Std = 0.0000
# Average AUC: 0.8811, Std = 0.0000
# no DRWeibo W19 PHEME Politifact
# Average Micro F1: 0.7542, Std = 0.0000
# Average Macro F1: 0.7536, Std = 0.0000
# Average AUC: 0.8618, Std = 0.0000
# no DRWeibo W19 PHEME Politifact Gossipcop
# Average Micro F1: 0.7511, Std = 0.0000
# Average Macro F1: 0.7504, Std = 0.0000
# Average AUC: 0.8512, Std = 0.0000

# TwitterCOVID19
# Average Micro F1: 0.6040, Std = 0.0000
# Average Macro F1: 0.6014, Std = 0.0000
# Average AUC: 0.6626, Std = 0.0000
# Target:TwitterCOVID19
# no DRWeibo
# Average Micro F1: 0.5940, Std = 0.0000
# Average Macro F1: 0.5923, Std = 0.0000
# Average AUC: 0.6399, Std = 0.0000
# no DRWeibo Weibo
# Average Micro F1: 0.5537, Std = 0.0000
# Average Macro F1: 0.5533, Std = 0.0000
# Average AUC: 0.5937, Std = 0.0000
# no DRWeibo Weibo WeiboCOVID19
# Average Micro F1: 0.5604, Std = 0.0000
# Average Macro F1: 0.5590, Std = 0.0000
# Average AUC: 0.5923, Std = 0.0000
# no DRWeibo Weibo WeiboCOVID19 PHEME
# Average Micro F1: 0.5336, Std = 0.0000
# Average Macro F1: 0.5333, Std = 0.0000
# Average AUC: 0.5844, Std = 0.0000
# no DRWeibo Weibo WeiboCOVID19 PHEME Politifact
# Average Micro F1: 0.5336, Std = 0.0000
# Average Macro F1: 0.5334, Std = 0.0000
# Average AUC: 0.5884, Std = 0.0000


# DRWeiboV3
# Average Micro F1: 0.7906, Std = 0.0000
# Average Macro F1: 0.7905, Std = 0.0000
# Average AUC: 0.8884, Std = 0.0000
# Weibo
# Average Micro F1: 0.8275, Std = 0.0000
# Average Macro F1: 0.8266, Std = 0.0000
# Average AUC: 0.9078, Std = 0.0000
# TwitterCOVID19
# Average Micro F1: 0.7081, Std = 0.0000
# Average Macro F1: 0.6883, Std = 0.0000
# Average AUC: 0.7473, Std = 0.0000
# Politifact
# Average Micro F1: 0.9528, Std = 0.0000
# Average Macro F1: 0.9526, Std = 0.0000
# Average AUC: 0.9957, Std = 0.0000
# Gossipcop
# Average Micro F1: 0.9506, Std = 0.0000
# Average Macro F1: 0.9506, Std = 0.0000
# Average AUC: 0.9810, Std = 0.0000
# WeiboCOVID19
# Average Micro F1: 0.6734, Std = 0.0000
# Average Macro F1: 0.6691, Std = 0.0000
# Average AUC: 0.7391, Std = 0.0000
# PHEME
# Average Micro F1: 0.9306, Std = 0.0000
# Average Macro F1: 0.9260, Std = 0.0000
# Average AUC: 0.9846, Std = 0.0000
