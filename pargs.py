import argparse


def pargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PHEME')

    parser.add_argument('--cuda', type=int, default=3)

    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--out_feat', type=int, default=128)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)

    # Node centrality metric can be chosen from "Degree", "PageRank", "Eigenvector", "Betweenness".
    parser.add_argument('--centrality', type=str, default="PageRank")
    # Augmentation can be chosen from "DropEdge,mean,0.3,0.7", "NodeDrop,0.3,0.7", "AttrMask,0.3,0.7",
    # or augmentation combination "DropEdge,mean,0.3,0.7||NodeDrop,0.3,0.7", "DropEdge,mean,0.3,0.7||AttrMask,0.3,0.7".
    # Str like "DropEdge,mean,0.3,0.7" means "AugName,[aggr,]p,threshold".
    parser.add_argument('--aug1', type=str, default="DropEdge,mean,0.3,0.7")
    parser.add_argument('--aug2', type=str, default="NodeDrop,0.3,0.7")

    args = parser.parse_args()
    return args
