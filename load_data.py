import os
import json
import pickle
import numpy
import collections
from sklearn.preprocessing import StandardScaler
import random
import torch
import numpy as np
from datetime import datetime
import re
from collections import Counter
from scipy.stats import entropy
import scipy.sparse as sp
from torch_geometric.io import read_txt_array
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.data import DataLoader
from torch.utils.data import Sampler
from torch_geometric.utils import degree, to_undirected, coalesce, cumsum, to_networkx, k_hop_subgraph
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from sklearn.decomposition import PCA
from torch_scatter import scatter
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import ConcatDataset


def get_word_count(text):
    """Counts words in a string. Handles both English and Chinese."""
    # For English, split by space. For Chinese, just count characters as a proxy.
    # A more advanced CJK tokenizer could be used, but this is a robust simple approach.
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text)
    return len(words)


def get_char_count(text):
    """Counts total characters in a string."""
    return len(text)


def get_type_token_ratio(text):
    """Calculates vocabulary richness (TTR)."""
    words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+', text)
    if not words:
        return 0
    unique_words = set(words)
    return len(unique_words) / len(words)


def get_shannon_entropy(text):
    """Calculates the Shannon Entropy of the text based on character frequency."""
    if not text:
        return 0
    counts = Counter(text)
    frequencies = [count / len(text) for count in counts.values()]
    return entropy(frequencies, base=2)


def pca_compression(seq, k):
    pca = PCA(n_components=k)
    seq = pca.fit_transform(seq)

    # print(pca.explained_variance_ratio_.sum())
    return seq


def min_max_normalize(features):
    """
    对特征进行最大-最小归一化，将值缩放到 [0, 1]。
    """
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    normalized = (features - min_vals) / (max_vals - min_vals + 1e-8)  # 避免除以零
    return normalized


def standardize(features):
    """
    对特征进行标准化，调整为均值为 0，标准差为 1。
    """
    mean_vals = features.mean(axis=0)
    std_vals = features.std(axis=0) + 1e-8  # 避免除以零
    standardized = (features - mean_vals) / std_vals
    return standardized


def svd_compression(seq, k):
    res = np.zeros_like(seq)
    # 进行奇异值分解, 从svd函数中得到的奇异值sigma 是从大到小排列的
    U, Sigma, VT = np.linalg.svd(seq, full_matrices=False)
    # print(U[:, :k].shape)
    # print(VT[:k, :].shape)
    # res = U[:, :k].dot(np.diag(Sigma[:k]))
    # 只保留前 k 个奇异值对应的部分
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    # VT_k = VT[:k, :]

    # 压缩后的特征
    compressed_seq = U_k.dot(Sigma_k)

    return compressed_seq


def text_to_vector(texts, model, tokenizer, max_length=128, batch_size=16):
    """
    使用 XLM-RoBERTa 将文本列表转化为向量，支持批量处理。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(
            device)
        with torch.no_grad():
            outputs = model(**inputs)
        vectors.append(outputs.last_hidden_state[:, 0, :].cpu())  # 使用 [CLS] token 的输出
    return torch.cat(vectors, dim=0)


def parse_date(date_str):
    """
    Parses the date string in the format 'Sat Aug 09 22:33:06 +0000 2014'.
    """
    return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")


def preprocess_text(value):
    """综合清洗函数：处理 Unicode 引号、非 ASCII 字符、URL、提及、主题标签、标点和空格"""
    if isinstance(value, str):
        # 替换特殊的 Unicode 引号
        value = value.replace('\u201c', '').replace('\u201d', '')
        value = value.replace('\u2018', "").replace('\u2019', "")

        # 移除非 ASCII 字符
        value = re.sub(r'[^\x00-\x7F]+', '', value)

        # 移除 URL
        value = re.sub(r'http\S+|www\.\S+', '', value)

        # 移除提及和主题标签
        value = re.sub(r'@\w+', '', value)  # 删除 @提及
        value = re.sub(r'#\w+', '', value)  # 删除 #主题标签

        # 标准化标点符号
        value = re.sub(r'\.{2,}', '.', value)  # 替换多个点

        # 删除反斜杠
        value = value.replace('//', '')  # 删除反斜杠

        # 去除多余空格
        value = ' '.join(value.split())

    return value


def trans_time(t, t_init):
    """
    Converts time strings into seconds since a reference time (t_init).
    """
    try:
        t_seconds = int(datetime.strptime(t, "%a %b %d %H:%M:%S +0000 %Y").timestamp())
        t_init_seconds = int(datetime.strptime(t_init, "%a %b %d %H:%M:%S +0000 %Y").timestamp())
        return t_seconds - t_init_seconds
    except Exception as e:
        print(f"Error parsing time: {e}")
        return None


class TreeDataset(InMemoryDataset):
    def __init__(self, root, centrality_metric="PageRank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        all_data_names = self.raw_file_names
        xx = []
        # for filename in all_data_names:
        #     tweets = os.listdir(os.path.join(self.raw_dir, filename))
        #     print(filename)
        filename = self.root
        root_word_counts = []
        root_char_counts = []
        reply_word_counts = []
        reply_char_counts = []
        root_ttrs = []
        reply_ttrs = []
        root_entropies = []
        reply_entropies = []
        for tweet in all_data_names:
            print(tweet)
            y = []
            centrality = None
            row = []
            col = []
            no_root_row = []
            no_root_col = []
            edges = []
            post = json.load(open(os.path.join(self.raw_dir, tweet), 'r', encoding='utf-8'))

            if "15" in filename or "16" in filename:
                tfidf = post['source']['content']
                indices = [[0, int(index_freq.split(':')[0])] for index_freq in tfidf.split()]
                values = [int(index_freq.split(':')[1]) for index_freq in tfidf.split()]
            else:
                # 提取内容
                texts = [post["source"]["content"]]  # 添加主内容
                # texts += [comment["content"] for comment in post["comment"] if comment["content"].strip()]  # 添加非空评论
                root_word_counts.append(get_word_count(post["source"]["content"]))
                root_char_counts.append(get_char_count(post["source"]["content"]))
                root_ttrs.append(get_type_token_ratio(post["source"]["content"]))
                root_entropies.append(get_shannon_entropy(post["source"]["content"]))
            if 'label' in post['source'].keys():
                y.append(post['source']['label'])
            else:
                y.append(-1)

            if "time" in post["source"]:
                # 定义原始时间格式
                original_format = "%y-%m-%d %H:%M"

                # 解析为 datetime 对象
                parsed_time = datetime.strptime(post["source"]["time"], original_format)

                # 转换为目标格式
                init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")

            for i, comment in enumerate(post['comment']):
                if "time" in post["source"]:
                    # 解析为 datetime 对象
                    parsed_time = datetime.strptime(comment["time"], original_format)

                    # 转换为目标格式
                    post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                    edge_time = trans_time(post_time, init_time)
                    # edge_time = np.log(1 + np.abs(edge_time))
                    edges.append(edge_time)
                else:
                    edges.append(1.0)
                if "15" in filename or "16" in filename:
                    indices += [[i + 1, int(index_freq.split(':')[0])] for index_freq in comment['content'].split()]
                    values += [int(index_freq.split(':')[1]) for index_freq in comment['content'].split()]
                elif comment['content'] == "":
                    txt = "转发"
                    texts += [txt]
                    reply_word_counts.append(get_word_count(txt))
                    reply_char_counts.append(get_char_count(txt))
                    root_ttrs.append(get_type_token_ratio(txt))
                    root_entropies.append(get_shannon_entropy(txt))
                else:
                    texts += [comment["content"]]  # 添加非空评论
                    reply_word_counts.append(get_word_count(comment["content"]))
                    reply_char_counts.append(get_char_count(comment["content"]))
                    reply_ttrs.append(get_type_token_ratio(comment["content"]))
                    reply_entropies.append(get_shannon_entropy(comment["content"]))

                if comment['parent'] != -1:
                    no_root_row.append(comment['parent'] + 1)
                    no_root_col.append(comment['comment id'] + 1)
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)

            # if self.centrality_metric == "Degree":
            #     centrality = torch.tensor(post['centrality']['Degree'], dtype=torch.float32)
            # elif self.centrality_metric == "PageRank":
            centrality = torch.tensor(post['centrality']['Pagerank'], dtype=torch.float32)
            # elif self.centrality_metric == "Eigenvector":
            #     centrality = torch.tensor(post['centrality']['Eigenvector'], dtype=torch.float32)
            # elif self.centrality_metric == "Betweenness":
            #     centrality = torch.tensor(post['centrality']['Betweenness'], dtype=torch.float32)

            edge_index = [row, col]
            y = torch.LongTensor(y)
            edge_index = to_undirected(torch.LongTensor(edge_index)) if self.undirected else torch.LongTensor(
                edge_index)
            # if "15" in filename or "16" in filename:
            #     x = torch.sparse_coo_tensor(torch.tensor(indices).t(), values, (len(post['comment']) + 1, 5000),
            #                                 dtype=torch.float32).to_dense()
            # else:
            #     x = text_to_vector(texts, self.model, self.tokenizer)
            # if "Weibo" in filename:
            #     lang = torch.tensor([1])
            # else:
            #     lang = torch.tensor([0])
            #
            # one_data = Data(x=x, y=y, edge_index=edge_index,
            #                 edge_attr=torch.FloatTensor(edges), lang=lang, centrality=centrality)
            # data_list.append(one_data)
        print(f"\n--- Type-Token Ratio (TTR) Statistics ---")
        print(f"Root Posts:  Mean={np.mean(root_ttrs):.2f}, Median={np.median(root_ttrs):.2f}")
        print(f"Replies:     Mean={np.mean(reply_ttrs):.2f}, Median={np.median(reply_ttrs):.2f}")

        print(f"\n--- Shannon Entropy Statistics (bits) ---")
        print(f"Root Posts:  Mean={np.mean(root_entropies):.2f}, Median={np.median(root_entropies):.2f}")
        print(f"Replies:     Mean={np.mean(reply_entropies):.2f}, Median={np.median(reply_entropies):.2f}")
        print(f"--- Word Count Statistics ---")
        print(
            f"Root Posts:  Mean={np.mean(root_word_counts):.2f}, Median={np.median(root_word_counts):.2f}, Std={np.std(root_word_counts):.2f}")
        print(
            f"Replies:     Mean={np.mean(reply_word_counts):.2f}, Median={np.median(reply_word_counts):.2f}, Std={np.std(reply_word_counts):.2f}")

        print(f"\n--- Character Count Statistics ---")
        print(
            f"Root Posts:  Mean={np.mean(root_char_counts):.2f}, Median={np.median(root_char_counts):.2f}, Std={np.std(root_char_counts):.2f}")
        print(
            f"Replies:     Mean={np.mean(reply_char_counts):.2f}, Median={np.median(reply_char_counts):.2f}, Std={np.std(reply_char_counts):.2f}")

        if "15" in filename or "16" in filename:
            # 合并所有图的节点特征
            for data in data_list:
                xx.append(data.x)  # data.x 是节点特征矩阵
            all_features = torch.cat(xx).numpy()
            # normalized_features = min_max_normalize(all_features)
            # all_features = standardize(normalized_features)

            # 对合并的特征矩阵进行 PCA 降维
            pca = PCA(n_components=768)
            reduced_features = pca.fit_transform(all_features)
            reduced_features = torch.FloatTensor(reduced_features)

            # 将降维后的特征分配回每个图
            start_idx = 0
            for data in data_list:
                num_nodes = data.x.shape[0]
                data.x = reduced_features[start_idx: start_idx + num_nodes]  # 取对应行
                start_idx += num_nodes

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])


def read_json_file(file_path):
    """
    Reads a JSON file and returns its content as a dictionary.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        dict or None: The content of the JSON file, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file '{file_path}': {e}")
        return None


# need normalization
# root centrality = no children 1 level reply centrality
def pagerank_centrality(data, damp=0.85, k=10):
    device = data.x.device
    bu_edge_index = data.edge_index.clone()
    bu_edge_index[0], bu_edge_index[1] = data.edge_index[1], data.edge_index[0]

    num_nodes = data.num_nodes
    deg_out = degree(bu_edge_index[0])
    centrality = torch.ones((num_nodes,)).to(device).to(torch.float32)

    for i in range(k):
        edge_msg = centrality[bu_edge_index[0]] / deg_out[bu_edge_index[0]]
        agg_msg = scatter(edge_msg, bu_edge_index[1], reduce='sum')
        pad = torch.zeros((len(centrality) - len(agg_msg),)).to(device).to(torch.float32)
        agg_msg = torch.cat((agg_msg, pad), 0)

        centrality = (1 - damp) * centrality + damp * agg_msg

    centrality[0] = centrality.min().item()
    return centrality


class TreeDataset_PHEME(InMemoryDataset):
    def __init__(self, root, centrality_metric="Pagerank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        event_list = ['germanwings-crash-all-rnr-threads', 'charliehebdo-all-rnr-threads',
                      'sydneysiege-all-rnr-threads', 'ebola-essien-all-rnr-threads',
                      'gurlitt-all-rnr-threads', 'putinmissing-all-rnr-threads',
                      'ferguson-all-rnr-threads', 'ottawashooting-all-rnr-threads',
                      'prince-toronto-all-rnr-threads']
        root_word_counts = []
        root_char_counts = []
        reply_word_counts = []
        reply_char_counts = []
        root_ttrs = []
        reply_ttrs = []
        root_entropies = []
        reply_entropies = []
        for event in event_list:
            event_path = os.path.join(self.raw_dir, self.raw_file_names[0], event)
            if not os.path.exists(event_path):
                continue

            # Process non-rumor data
            non_rumor_path = os.path.join(event_path, 'non-rumours')
            if os.path.exists(non_rumor_path):

                # Iterate through each news item in the event directory
                for news in os.listdir(non_rumor_path):
                    print(news)
                    # if news == "580320890086383616":
                    #     print("DDDD")
                    json_content = {"source": {}, "comment": []}
                    if not news.startswith('._') and news != '.DS_Store':
                        if len(os.listdir(os.path.join(event_path, 'non-rumours', news, 'reactions'))) == 0:
                            continue
                        source_tweets_path = os.path.join(event_path, 'non-rumours', news, 'source-tweets',
                                                          f'{news}.json')
                        try:
                            source_tweets_data = read_json_file(source_tweets_path)
                            if source_tweets_data:
                                json_content["source"]["t"] = source_tweets_data["created_at"]
                                json_content["source"]["id"] = source_tweets_data["id"]
                                json_content["source"]["parent"] = source_tweets_data["in_reply_to_status_id"]
                                json_content["source"]["text"] = source_tweets_data["text"]

                        except Exception as e:
                            print(f"Error reading source tweet {source_tweets_path}: {e}")
                            continue

                        reactions_path = os.path.join(event_path, 'non-rumours', news, 'reactions')
                        node_index = {}
                        node_index[int(news)] = 0
                        json_content["source"]["id"] = 0
                        if os.path.exists(reactions_path):

                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            if int(comment_data["id"]) not in node_index:
                                                node_index[int(comment_data["id"])] = len(node_index)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            content = {
                                                "t": comment_data["created_at"],
                                                "id": node_index[int(comment_data["id"])],
                                                "text": comment_data["text"],
                                                # "parent": node_index[int(comment_data["in_reply_to_status_id"])]
                                            }
                                            if comment_data["in_reply_to_status_id"] == None:
                                                content["parent"] = 0
                                            elif int(comment_data["in_reply_to_status_id"]) in node_index.keys():
                                                content["parent"] = node_index[
                                                    int(comment_data["in_reply_to_status_id"])]
                                            else:
                                                content["parent"] = 0
                                            json_content["comment"].append(content)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                        row = []
                        col = []
                        edge_attr = []
                        texts = [json_content["source"]["text"]]

                        root_word_counts.append(get_word_count(json_content["source"]["text"]))
                        root_char_counts.append(get_char_count(json_content["source"]["text"]))
                        root_ttrs.append(get_type_token_ratio(json_content["source"]["text"]))
                        root_entropies.append(get_shannon_entropy(json_content["source"]["text"]))

                        # 定义原始时间格式
                        original_format = "%a %b %d %H:%M:%S %z %Y"

                        # 解析为 datetime 对象
                        parsed_time = datetime.strptime(json_content["source"]["t"], original_format)

                        # 转换为目标格式
                        init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                        for post in json_content["comment"]:
                            row.append(post["parent"])
                            col.append(post["id"])

                            parsed_time = datetime.strptime(post["t"], original_format)

                            # 转换为目标格式
                            post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                            edge_time = trans_time(post_time, init_time)
                            edge_attr.append(edge_time)

                            if post['text'] == "":
                                txt = "Relay"
                                texts += [txt]
                                reply_word_counts.append(get_word_count(txt))
                                reply_char_counts.append(get_char_count(txt))
                                root_ttrs.append(get_type_token_ratio(txt))
                                root_entropies.append(get_shannon_entropy(txt))
                            else:
                                texts += [post["text"]]  # 添加非空评论
                                reply_word_counts.append(get_word_count(post["text"]))
                                reply_char_counts.append(get_char_count(post["text"]))
                                root_ttrs.append(get_type_token_ratio(post["text"]))
                                root_entropies.append(get_shannon_entropy(post["text"]))
                        # x = text_to_vector(texts, self.model, self.tokenizer)
                        # lang = torch.tensor([0])
                        # edge_index = torch.LongTensor([row, col])
                        # one_data = Data(x=x, y=torch.LongTensor([0]), edge_index=edge_index,
                        #                 edge_attr=torch.FloatTensor(edge_attr), lang=lang)
                        # if one_data.num_nodes > 1:
                        #     pc = pagerank_centrality(one_data)
                        #
                        # centrality = torch.tensor(pc, dtype=torch.float32)
                        # one_data["centrality"] = centrality
                        # print(one_data)
                        # data_list.append(one_data)
            # Process rumor data
            rumor_path = os.path.join(event_path, 'rumours')
            if os.path.exists(rumor_path):

                # Iterate through each news item in the event directory
                for news in os.listdir(rumor_path):
                    # if news == "580320890086383616":
                    #     print("DDDD")
                    print(news)

                    json_content = {"source": {}, "comment": []}
                    if not news.startswith('._') and news != '.DS_Store':
                        if len(os.listdir(os.path.join(event_path, 'rumours', news, 'reactions'))) == 0:
                            continue
                        source_tweets_path = os.path.join(event_path, 'rumours', news, 'source-tweets',
                                                          f'{news}.json')
                        try:
                            source_tweets_data = read_json_file(source_tweets_path)
                            if source_tweets_data:
                                json_content["source"]["t"] = source_tweets_data["created_at"]
                                json_content["source"]["id"] = source_tweets_data["id"]
                                json_content["source"]["parent"] = source_tweets_data[
                                    "in_reply_to_status_id"]
                                json_content["source"]["text"] = source_tweets_data["text"]


                        except Exception as e:
                            print(f"Error reading source tweet {source_tweets_path}: {e}")
                            continue

                        reactions_path = os.path.join(event_path, 'rumours', news, 'reactions')
                        node_index = {}
                        node_index[int(news)] = 0
                        json_content["source"]["id"] = 0
                        if os.path.exists(reactions_path):

                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            if int(comment_data["id"]) not in node_index:
                                                node_index[int(comment_data["id"])] = len(node_index)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                            for comment in os.listdir(reactions_path):
                                if not comment.startswith('._') and comment != '.DS_Store':
                                    comment_path = os.path.join(reactions_path, comment)
                                    try:
                                        comment_data = read_json_file(comment_path)
                                        if comment_data:
                                            content = {
                                                "t": comment_data["created_at"],
                                                "id": node_index[int(comment_data["id"])],
                                                "text": comment_data["text"],
                                                # "parent": node_index[
                                                #     int(comment_data["in_reply_to_status_id"])]
                                            }
                                            if comment_data["in_reply_to_status_id"] == None:
                                                content["parent"] = 0
                                            elif int(comment_data["in_reply_to_status_id"]) in node_index.keys():
                                                content["parent"] = node_index[
                                                    int(comment_data["in_reply_to_status_id"])]
                                            else:
                                                content["parent"] = 0
                                            json_content["comment"].append(content)
                                    except Exception as e:
                                        print(f"Error reading comment {comment_path}: {e}")
                                        continue
                        row = []
                        col = []
                        edge_attr = []
                        texts = [json_content["source"]["text"]]

                        reply_word_counts.append(get_word_count(json_content["source"]["text"]))
                        reply_char_counts.append(get_char_count(json_content["source"]["text"]))
                        reply_ttrs.append(get_type_token_ratio(json_content["source"]["text"]))
                        reply_entropies.append(get_shannon_entropy(json_content["source"]["text"]))

                        # 定义原始时间格式
                        original_format = "%a %b %d %H:%M:%S %z %Y"

                        # 解析为 datetime 对象
                        parsed_time = datetime.strptime(json_content["source"]["t"], original_format)

                        # 转换为目标格式
                        init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                        for post in json_content["comment"]:
                            row.append(post["parent"])
                            col.append(post["id"])

                            parsed_time = datetime.strptime(post["t"], original_format)

                            # 转换为目标格式
                            post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                            edge_time = trans_time(post_time, init_time)
                            edge_attr.append(edge_time)

                            if post['text'] == "":
                                txt = "Relay"
                                texts += [txt]
                                reply_word_counts.append(get_word_count(txt))
                                reply_char_counts.append(get_char_count(txt))
                                root_ttrs.append(get_type_token_ratio(txt))
                                root_entropies.append(get_shannon_entropy(txt))
                            else:
                                texts += [post["text"]]  # 添加非空评论
                                reply_word_counts.append(get_word_count(post["text"]))
                                reply_char_counts.append(get_char_count(post["text"]))
                                root_ttrs.append(get_type_token_ratio(post["text"]))
                                root_entropies.append(get_shannon_entropy(post["text"]))
                        # x = text_to_vector(texts, self.model, self.tokenizer)
                        # lang = torch.tensor([0])
                        # edge_index = torch.LongTensor([row, col])
                        # one_data = Data(x=x, y=torch.LongTensor([1]), edge_index=edge_index,
                        #                 edge_attr=torch.FloatTensor(edge_attr), lang=lang)
                        # if one_data.num_nodes > 1:
                        #     pc = pagerank_centrality(one_data)
                        #
                        # centrality = torch.tensor(pc, dtype=torch.float32)
                        # one_data["centrality"] = centrality
                        # print(one_data)
                        # data_list.append(one_data)
        print(f"\n--- Type-Token Ratio (TTR) Statistics ---")
        print(f"Root Posts:  Mean={np.mean(root_ttrs):.2f}, Median={np.median(root_ttrs):.2f}")
        print(f"Replies:     Mean={np.mean(reply_ttrs):.2f}, Median={np.median(reply_ttrs):.2f}")

        print(f"\n--- Shannon Entropy Statistics (bits) ---")
        print(f"Root Posts:  Mean={np.mean(root_entropies):.2f}, Median={np.median(root_entropies):.2f}")
        print(f"Replies:     Mean={np.mean(reply_entropies):.2f}, Median={np.median(reply_entropies):.2f}")
        print(f"--- Word Count Statistics ---")
        print(
            f"Root Posts:  Mean={np.mean(root_word_counts):.2f}, Median={np.median(root_word_counts):.2f}, Std={np.std(root_word_counts):.2f}")
        print(
            f"Replies:     Mean={np.mean(reply_word_counts):.2f}, Median={np.median(reply_word_counts):.2f}, Std={np.std(reply_word_counts):.2f}")

        print(f"\n--- Character Count Statistics ---")
        print(
            f"Root Posts:  Mean={np.mean(root_char_counts):.2f}, Median={np.median(root_char_counts):.2f}, Std={np.std(root_char_counts):.2f}")
        print(
            f"Replies:     Mean={np.mean(reply_char_counts):.2f}, Median={np.median(reply_char_counts):.2f}, Std={np.std(reply_char_counts):.2f}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])


class TreeDataset_UPFD(InMemoryDataset):
    def __init__(self, root, centrality_metric="Pagerank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        x = sp.load_npz(os.path.join(self.raw_dir, 'new_bert_feature.npz'))
        x = torch.from_numpy(x.todense()).to(torch.float)

        edge_index = read_txt_array(os.path.join(self.raw_dir, 'A.txt'), sep=',',
                                    dtype=torch.long).t()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))

        y = np.load(os.path.join(self.raw_dir, 'graph_labels.npy'))
        y = torch.from_numpy(y).to(torch.long)
        _, y = y.unique(sorted=True, return_inverse=True)

        batch = np.load(os.path.join(self.raw_dir, 'node_graph_id.npy'))
        batch = torch.from_numpy(batch).to(torch.long)

        # Create individual graphs as Data objects
        data_list = []
        for graph_id in batch.unique():
            node_mask = batch == graph_id
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

            graph_x = x[node_mask]
            graph_edge_index = edge_index[:, edge_mask] - node_mask.nonzero(as_tuple=False)[0].min()
            graph_y = y[graph_id]

            data = Data(x=graph_x, edge_index=graph_edge_index, y=graph_y)
            if data.num_nodes > 1:
                pc = pagerank_centrality(data)

            centrality = torch.tensor(pc, dtype=torch.float32)
            data["centrality"] = centrality
            data["lang"] = torch.tensor([0])
            data.edge_attr = torch.ones(graph_edge_index.size(1), dtype=torch.float)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        all_data, slices = self.collate(data_list)
        torch.save((all_data, slices), self.processed_paths[0])


class HugeDataset(Dataset):
    def __init__(self, root, centrality_metric="Pagerank", undirected=False, transform=None, pre_transform=None,
                 pre_filter=None):
        self.undirected = undirected
        self.centrality_metric = centrality_metric
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
        # self.model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """List all raw files in the raw directory."""
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        """Define processed file names based on dataset size."""
        return [f'data_{i}.pt' for i in range(len(self.raw_file_names))]

    def download(self):
        """Skip download logic as data is assumed to be locally available."""
        pass

    def process(self):
        """Process raw files into individual PyG Data objects."""
        for i, raw_file in enumerate(self.raw_file_names):
            print(f"Processing file: {raw_file}")
            post = json.load(open(os.path.join(self.raw_dir, raw_file), 'r', encoding='utf-8'))
            data = self.process_single_file(post)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            # Save each processed graph individually
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

    def len(self):
        """Return the number of processed graphs."""
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a single graph from processed files."""
        data_path = os.path.join(self.processed_dir, f'data_{idx}.pt')
        return torch.load(data_path)

    def process_single_file(self, post):
        """Process a single JSON file into a PyG Data object."""
        try:
            # Extract label
            y = [post['source'].get('label', -1)]

            # Extract node texts
            texts = [post["source"]["content"]]
            row, col, edges = [], [], []
            for i, comment in enumerate(post['comment']):
                row.append(comment['parent'] + 1)
                col.append(comment['comment id'] + 1)
                edges.append(self.calculate_edge_time(comment, post))

                if comment['content']:
                    texts.append(comment['content'])
                else:
                    texts.append('回复')

            # Create edge index tensors
            edge_index = torch.LongTensor([row, col])
            if self.undirected:
                edge_index = to_undirected(edge_index)

            # Generate node features
            x = self.create_node_features(texts)

            # Extract centrality
            # centrality = torch.tensor(post['centrality'].get(self.centrality_metric, []), dtype=torch.float32)

            # Language type
            lang = torch.tensor([1 if "Weibo" in self.root else 0])
            data = Data(
                x=x,
                y=torch.LongTensor(y),
                edge_index=edge_index,
                edge_attr=torch.FloatTensor(edges),
                lang=lang
            )
            if data.num_nodes > 1:
                pc = pagerank_centrality(data)

            centrality = torch.tensor(pc, dtype=torch.float32)
            data["centrality"] = centrality
            # Create the Data object
            return data
        except Exception as e:
            print(f"Error processing file: {e}")
            return None

    def calculate_edge_time(self, comment, post):
        """Calculate edge time or assign a default value."""
        try:
            original_format = "%y-%m-%d %H:%M"
            source_time = datetime.strptime(post["source"]["time"], original_format)
            comment_time = datetime.strptime(comment["time"], original_format)
            return (comment_time - source_time).total_seconds()
        except:
            return 1.0

    def create_node_features(self, texts):
        """Convert texts into node feature tensors."""
        return text_to_vector(texts, self.model, self.tokenizer)


class CovidDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        Custom dataset class for graph data.
        :param root: Root directory containing .npz files.
        :param transform: Transformation function for the data.
        :param pre_transform: Preprocessing function for the data.
        """
        super().__init__(root, transform, pre_transform)
        self.data_dir = os.path.join(root)  # Directory for dataset files

    @property
    def raw_file_names(self):
        """
        Returns a list of all .npz files in the directory.
        """
        return [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]

    def len(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.raw_file_names)

    def get(self, idx):
        """
        Loads a single data sample by index.
        :param idx: Index of the sample.
        :return: A torch_geometric.data.Data object.
        """
        file_name = self.raw_file_names[idx]  # Get the file name
        file_path = os.path.join(self.data_dir, file_name)  # Construct the full file path

        data = np.load(file_path, allow_pickle=True)

        # Extract features, edges, and labels
        x = torch.stack(data['x'].tolist(), dim=0).float()
        edge_index = torch.tensor(data['edgeindex'], dtype=torch.long)
        y = torch.tensor([int(data['y'])], dtype=torch.long)

        # Create a torch_geometric.data.Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # Compute PageRank centrality if the graph has more than one node
        if data.num_nodes > 1:
            pc = pagerank_centrality(data)

        # Add centrality and language attributes to the data object
        centrality = torch.tensor(pc, dtype=torch.float32)
        data["centrality"] = centrality
        lang = torch.tensor([1 if "Weibo" in self.root else 0])
        data["lang"] = lang
        data.edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)

        return data


class DomainWrapperDataset(Dataset):
    """
    A wrapper that takes a dataset and a domain ID.
    When an item is requested, it fetches the item from the original dataset
    and dynamically adds the 'domain_id' attribute to it.
    """

    def __init__(self, dataset, domain_id):
        self.dataset = dataset
        # Store domain_id as a tensor for easy batching by the DataLoader
        self.domain_id = torch.tensor([domain_id])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get the original data object (e.g., a PyG Data object)
        data = self.dataset[idx]

        # It's good practice to clone to avoid modifying the cached original object
        data = data.clone()

        # Add the domain ID attribute
        data.domain_id = self.domain_id
        return data


def load_datasets_with_prompts(args):
    """
    Load all datasets, dynamically exclude the target dataset based on args.dataset,
    and assign corresponding prompt keys to each DataLoader.

    Args:
        args: Argument parser containing the target dataset name and batch_size.

    Returns:
        train_loaders: List of DataLoaders for training datasets, each with a `prompt_key` attribute.
        target_loader: DataLoader for the target dataset.
    """
    # Define all datasets and their corresponding paths
    dataset_mapping = {
        "DRWeiboV3": {"dataset": TreeDataset("../ACL/Data/DRWeiboV3/"), "prompt_key": "DRWeiboV3_prompt"},
        "Weibo": {"dataset": TreeDataset("../ACL/Data/Weibo/"), "prompt_key": "Weibo_prompt"},
        "WeiboCOVID19": {"dataset": CovidDataset("../ACL/Data/Weibo-COVID19/Weibograph"), "prompt_key": "W19_prompt"},
        "PHEME": {"dataset": TreeDataset_PHEME("../ACL/Data/pheme/"), "prompt_key": "PHEME_prompt"},
        "Politifact": {"dataset": TreeDataset_UPFD("../ACL/Data/politifact/"), "prompt_key": "Politifact_prompt"},
        "Gossipcop": {"dataset": TreeDataset_UPFD("../ACL/Data/gossipcop/"), "prompt_key": "Gossipcop_prompt"},
        "TwitterCOVID19": {"dataset": CovidDataset("../ACL/Data/Twitter-COVID19/Twittergraph"),
                           "prompt_key": "T19_prompt"},
        # "ShallowAug": {"dataset": ShallowAugDataset(root="../ACL/Data/ShallowAugData/"), "prompt_key": "Shallow_prompt"}
        # "Twitter15-tfidf": {"dataset": TreeDataset("../ACL/Data/Twitter15-tfidf/"), "prompt_key": "en_prompt"}
    }

    # Check if the target dataset exists
    if args.dataset not in dataset_mapping:
        raise ValueError(f"Dataset '{args.dataset}' not found in the available datasets.")

    # Split the datasets into training and target (test)
    target_info = dataset_mapping[args.dataset]
    train_datasets = {key: info for key, info in dataset_mapping.items() if key != args.dataset}
    # train_datasets = {"DRWeiboV3": {"dataset": TreeDataset("../ACL/Data/DRWeiboV3/"), "prompt_key": "DRWeiboV3_prompt"}}
    # Create DataLoaders
    target_loader = DataLoader(target_info["dataset"], batch_size=args.batch_size, shuffle=False)
    target_loader.prompt_key = target_info["prompt_key"]  # Attach the prompt_key for the target dataset

    # train_loaders = []
    # for key, info in train_datasets.items():
    #     loader = DataLoader(info["dataset"], batch_size=args.batch_size, shuffle=True)
    #     loader.prompt_key = info["prompt_key"]  # Attach the prompt_key for each training dataset
    #     loader.name = key
    #     train_loaders.append(loader)

    train_domain_names = sorted(train_datasets.keys())
    domain_to_id = {name: i for i, name in enumerate(train_domain_names)}
    #
    # 2. Wrap each training dataset to add the domain ID
    # wrapped_datasets = []
    # for domain_name in train_domain_names:
    #     info = train_datasets[domain_name]
    #     domain_id = domain_to_id[domain_name]
    #     wrapped_ds = DomainWrapperDataset(dataset=info["dataset"], domain_id=domain_id)
    #     wrapped_datasets.append(wrapped_ds)
    # train_loaders = DataLoader(
    #     ConcatDataset(wrapped_datasets),
    #     batch_size=args.batch_size,
    #     shuffle=True
    # )

    train_loaders = DataLoader(
        ConcatDataset([info["dataset"] for key, info in train_datasets.items()]),
        batch_size=args.batch_size, shuffle=True
    )
    # train_loaders = DataLoader(target_info["dataset"], batch_size=args.batch_size, shuffle=True)
    return train_loaders, target_info["dataset"], train_domain_names


def preprocess_data(raw_dir, cache_file):
    """
    Preprocess raw data and save it as a pickle file.

    Args:
    raw_dir (str): Path to the raw data directory.
    model: The model used to convert text to vectors.
    tokenizer: The tokenizer used for text tokenization.
    trans_time: Function to convert time format.
    cache_file (str): Path to save the processed data as a pickle file.

    Returns:
    None
    """
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base', cache_dir="./")
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base', cache_dir="./")

    datas = []
    text = []
    ys = []

    # Iterate through all tweets in the raw directory
    for tweet in os.listdir(raw_dir):
        print(f"Processing tweet: {tweet}")
        post = json.load(open(os.path.join(raw_dir, tweet), "r", encoding="utf-8"))
        row = []
        col = []
        edges = []
        combined_text = post["source"]["content"]
        texts = [post["source"]["content"]]
        label = post["source"]["label"]
        ys.append(label)

        # Handle the timestamp if available
        if "time" in post["source"]:
            original_format = "%y-%m-%d %H:%M"
            parsed_time = datetime.strptime(post["source"]["time"], original_format)
            init_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")

        # Process comments and related information
        for i, comment in enumerate(post['comment']):
            if "time" in post["source"]:
                parsed_time = datetime.strptime(comment["time"], original_format)
                post_time = parsed_time.strftime("%a %b %d %H:%M:%S +0000 %Y")
                edge_time = trans_time(post_time, init_time)
                edges.append(edge_time)
            else:
                edges.append(1.0)
            if comment['content'] == "":
                txt = "转发"  # "Forward" for empty comments
                texts += [txt]
            else:
                texts += [comment["content"]]

            row.append(comment['parent'] + 1)
            col.append(comment['comment id'] + 1)

        # Construct graph data
        edge_index = [row, col]
        y = torch.LongTensor(label)
        edge_index = torch.LongTensor(edge_index)
        x = text_to_vector(texts, model, tokenizer)

        # Create a Data object to store the graph data
        one_data = Data(x=x, y=y, edge_index=edge_index, edge_attr=torch.FloatTensor(edges))
        datas.append(one_data)
        text.append(combined_text)

    # Save processed data to a pickle file
    with open(cache_file, "wb") as f:
        pickle.dump({
            "graph": datas,
            "texts": text,
            "labels": ys
        }, f)

    print(f"Data has been saved to {cache_file}")


def analyze_dataset(dataset: Dataset, name: str):
    """
    遍历数据集中的所有图，计算一系列核心统计指标。
    """
    print(f"Analyzing dataset: {name}...")
    stats = []

    # 使用 tqdm 显示进度条
    for i in tqdm(range(len(dataset)), desc=f"Processing {name}"):
        data = dataset[i]

        num_nodes = data.num_nodes
        num_edges = data.num_edges

        if num_nodes == 0:
            continue

        # 计算平均度数
        avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

        # --- 计算图的深度 (从根节点到最远节点的距离) ---
        graph_depth = 0
        if num_edges > 0:
            try:
                # 找到根节点 (入度为0的节点)
                in_degree = degree(data.edge_index[1], num_nodes=num_nodes)
                root_nodes = (in_degree == 0).nonzero(as_tuple=False).squeeze(-1)

                if root_nodes.numel() > 0:
                    root_node = root_nodes[0].item()  # 取第一个根节点

                    # 将图转换为 networkx 格式以便计算最短路径
                    G = to_networkx(data, to_undirected=False)

                    # 计算从根节点到所有其他节点的最短路径长度
                    path_lengths = nx.shortest_path_length(G, source=root_node)

                    # 图的深度 = 最长的最短路径
                    if path_lengths:
                        graph_depth = max(path_lengths.values())
            except Exception as e:
                # 处理图不连通或无根节点的特殊情况
                # print(f"Warning: Could not compute depth for graph {i} in {name}. Reason: {e}")
                graph_depth = -1  # 用一个特殊值标记

        stats.append({
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "graph_depth": graph_depth,
        })

    return pd.DataFrame(stats)


class StructuralPruning:
    """
    一个 PyG 变换，通过将图剪枝到随机的k跳邻域，
    来将一个“深层”图，人为地改造成一个“浅层”图。
    """

    def __init__(self, max_hops=2):
        # 对于模拟“浅滩”，一个较小的 max_hops (如1或2) 效果最好
        self.max_hops = max_hops

    def __call__(self, data):
        # 随机选择一个剪枝深度 k
        k = random.randint(1, self.max_hops)

        # 找到根节点 (入度为0)
        in_degree = degree(data.edge_index[1], num_nodes=data.num_nodes)
        root_nodes = (in_degree == 0).nonzero(as_tuple=False).squeeze(-1)

        if root_nodes.numel() == 0:
            # 如果没有严格的根，就从度数最高的节点开始，模拟“意见领袖”
            out_degree = degree(data.edge_index[0], num_nodes=data.num_nodes)
            center_node_idx = torch.argmax(out_degree).unsqueeze(0)
        else:
            center_node_idx = root_nodes[0].unsqueeze(0)

        edge_index = data.edge_index
        device = data.x.device
        num_nodes = data.num_nodes
        queue = [(center_node_idx, 0)]  # 队列中存储 (节点ID, 当前深度)
        visited = {center_node_idx}  # 记录已访问的节点，防止重复
        subset = [center_node_idx]  # 最终子图包含的节点列表

        head = 0
        while head < len(queue):
            current_node, current_depth = queue[head]
            head += 1

            if current_depth >= k:
                continue

            # 找到当前节点的所有出边邻居 (顺着箭头的方向)
            mask = edge_index[0] == current_node
            neighbors = edge_index[1][mask]

            for neighbor in neighbors:
                neighbor_idx = neighbor.item()
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    subset.append(neighbor_idx)
                    queue.append((neighbor_idx, current_depth + 1))

        subset = torch.tensor(subset, dtype=torch.long, device=device)

        # --- 步骤 3: 筛选出只存在于子图内部的边 ---
        source_nodes, target_nodes = edge_index

        # 检查每条边的起点和终点是否都在我们找到的子图节点集合中
        source_in_subset = torch.isin(source_nodes, subset)
        target_in_subset = torch.isin(target_nodes, subset)
        edge_mask = torch.logical_and(source_in_subset, target_in_subset)

        # 应用掩码，得到子图的边
        subgraph_edge_index = edge_index[:, edge_mask]

        # 如果筛选后没有边，直接返回一个只有节点的图
        if subgraph_edge_index.numel() == 0 and subset.numel() > 1:
            # 这是一个安全保护，虽然在树结构下不太可能发生
            return None

        # --- 步骤 4: 重新标记节点索引 (Relabeling) ---
        # 这是至关重要的一步，确保新图的节点索引是从 0 连续开始的
        node_map = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_map[subset] = torch.arange(subset.size(0), device=device)

        relabeled_edge_index = node_map[subgraph_edge_index]

        # 创建新的、被“压平”的 Data 对象
        data = Data(x=data.x[subset], edge_index=relabeled_edge_index, y=data.y)

        # Compute PageRank centrality if the graph has more than one node
        if data.num_nodes > 1:
            pc = pagerank_centrality(data)
        else:
            return None
        # Add centrality and language attributes to the data object
        centrality = torch.tensor(pc, dtype=torch.float32)
        data["centrality"] = centrality
        lang = torch.tensor([0])
        data["lang"] = lang
        data.edge_attr = torch.ones(data.edge_index.size(1), dtype=torch.float)

        return data


def plot_comparison(df1, name1, df2, name2, metric):
    """
    在同一个图上绘制两个数据集的指标分布，以供比较。
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df1[metric], color="skyblue", label=name1, kde=True, stat="density", element="step")
    sns.histplot(df2[metric], color="red", label=name2, kde=True, stat="density", element="step")

    mean1 = df1[metric].mean()
    mean2 = df2[metric].mean()

    plt.axvline(mean1, color='blue', linestyle='--', label=f'{name1} Mean: {mean1:.2f}')
    plt.axvline(mean2, color='darkred', linestyle='--', label=f'{name2} Mean: {mean2:.2f}')

    plt.title(f"{name1} vs {name2}: {metric.replace('_', ' ').title()} 分布对比", fontsize=16)
    plt.xlabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("密度 (Density)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


class ShallowAugDataset(InMemoryDataset):
    """
    一个【已修正】的、能够正确生成和加载“浅层”增强图的 InMemoryDataset。
    """

    def __init__(self, root, transform=None, pre_transform=None):
        # __init__ 只需要调用父类的构造函数即可。
        # PyG 的内部逻辑会自动处理“是否需要处理”或“直接加载”
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # 这个数据集没有“原始”文件，因为它是在其他数据集的基础上生成的。
        # 返回一个空列表即可。
        return []

    @property
    def processed_file_names(self):
        # 这是我们最终要生成的文件名。
        return ['shallow_augmented_data.pt']

    def download(self):
        # 我们不需要从网上下载任何东西。
        pass

    def process(self):
        """
        这个方法是 InMemoryDataset 的核心！
        它只会在 processed_file_names() 返回的文件不存在时，被【自动调用一次】。
        它的职责是：生成数据，并以 PyG 指定的格式保存。
        """
        print("Processed data not found. Starting one-time generation of ShallowAugDataset...")

        source_dataset_paths = {
            "DRWeiboV3": TreeDataset("../ACL/Data/DRWeiboV3/"),
            "Weibo": TreeDataset("../ACL/Data/Weibo/"),
            "WeiboCOVID19": CovidDataset("../ACL/Data/Weibo-COVID19/Weibograph"),
            "PHEME": TreeDataset_PHEME("../ACL/Data/pheme/"),
            "Politifact": TreeDataset_UPFD("../ACL/Data/politifact/"),
            "Gossipcop": TreeDataset_UPFD("../ACL/Data/gossipcop/"),
            "TwitterCOVID19": CovidDataset("../ACL/Data/Twitter-COVID19/Twittergraph"),
        }
        # 2. 初始化结构增强变换
        shallow_augment = StructuralPruning(max_hops=2)

        # 3. 遍历所有源图，生成新的浅层图
        all_shallow_graphs = []
        for name, dataset in source_dataset_paths.items():
            print(f"正在处理数据集: {name}...")
            for i in tqdm(range(len(dataset))):
                original_graph = dataset[i]
                # 对每个图应用剪枝，生成一个新的浅层图
                shallow_graph = shallow_augment(original_graph)
                if shallow_graph is not None:
                    all_shallow_graphs.append(shallow_graph)

        random.shuffle(all_shallow_graphs)
        end_shallow_graph = all_shallow_graphs[:6000]

        # 4. 【核心修正】使用 self.collate() 将图列表打包成 PyG 的标准格式
        #    这个函数会自动创建那个巨大的 data 对象和 slices 字典
        data, slices = self.collate(end_shallow_graph)

        # 5. 保存这个【元组】，而不是列表
        torch.save((data, slices), self.processed_paths[0])
        print("ShallowAugDataset has been successfully generated and saved!")


class DomainWrapperDataset(Dataset):
    """
    一个【已最终修正】的包装器，解决了“就地修改”的梯度错误。
    """

    def __init__(self, dataset, domain_id, dataset_name="Unknown", num_workers=4):
        self.dataset = dataset
        self.domain_id = domain_id
        self.dataset_name = dataset_name
        self.num_workers = num_workers
        self.mean = None
        self.std = None

        self._calculate_stats_fast()

    def _calculate_stats_fast(self):
        print(f"为数据集 '{self.dataset_name}' 高效计算标准化统计量...")

        temp_loader = DataLoader(self.dataset, batch_size=64, shuffle=False)

        all_x_tensors = []
        for batch in tqdm(temp_loader, desc=f"处理 {self.dataset_name}"):
            if batch.x is not None and batch.x.numel() > 0:
                all_x_tensors.append(batch.x)

        if not all_x_tensors:
            print(f"警告: 数据集 '{self.dataset_name}' 中没有找到有效的节点特征，跳过标准化。")
            self.mean = 0
            self.std = 1
            return

        all_x = torch.cat(all_x_tensors, dim=0)
        self.mean = torch.mean(all_x, dim=0)

        std_original = torch.std(all_x, dim=0)

        # --- 核心修正：使用 torch.where 进行非就地修改 ---
        # self.std[self.std == 0] = 1e-8  # 错误的、就地修改 (In-place)

        # 正确的、非就地修改 (Out-of-place):
        # torch.where 会创建一个【新】的张量，而不是修改原来的。
        self.std = torch.where(std_original == 0, 1e-8, std_original)

        print(f"数据集 '{self.dataset_name}' 的统计量计算完成。")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx].clone()
        if self.mean is not None and self.std is not None and data.x is not None:
            with torch.no_grad():
                data.x = (data.x - self.mean) / self.std
        data.domain_id = self.domain_id
        return data


class DomainStratifiedSampler(Sampler):
    """【已修正】的采样器，确保每个批次都包含来自N个不同领域的样本。"""

    def __init__(self, domain_indices, num_domains, batch_size):
        # ... (init 部分的代码完全不变) ...
        self.domain_indices = np.array(domain_indices)
        self.num_domains = num_domains
        self.batch_size = batch_size
        if self.batch_size % self.num_domains != 0:
            raise ValueError("batch_size must be a multiple of num_domains.")
        self.samples_per_domain_per_batch = self.batch_size // self.num_domains
        self.indices_per_domain = [np.where(self.domain_indices == i)[0] for i in range(self.num_domains)]
        # 找到拥有足够样本进行至少一次抽样的最小领域大小
        min_valid_domain_size = min(
            len(indices) for indices in self.indices_per_domain if len(indices) >= self.samples_per_domain_per_batch)
        if min_valid_domain_size == float('inf'):
            raise ValueError("No domain has enough samples for a single batch.")
        self.steps_per_epoch = min_valid_domain_size // self.samples_per_domain_per_batch

    def __iter__(self):
        for i in range(self.num_domains): np.random.shuffle(self.indices_per_domain[i])
        for step in range(self.steps_per_epoch):
            batch_indices = []
            for i in range(self.num_domains):
                start = step * self.samples_per_domain_per_batch
                end = start + self.samples_per_domain_per_batch
                batch_indices.extend(self.indices_per_domain[i][start:end])
            np.random.shuffle(batch_indices)

            # --- 核心修正：从 yield from 改为 yield ---
            # 我们现在遵守合同，返回一个完整的列表“包裹”
            yield batch_indices

    def __len__(self):
        # 注意：当使用 batch_sampler 时，DataLoader 会忽略 __len__
        # 真正的长度由 __iter__ 的步数决定
        return self.steps_per_epoch


def plot_heterogeneity_overlaid(df, metrics_to_plot, palette_name="viridis", alpha_fill=0.3, line_width=2.0):
    """
    【全新重写】绘制多源域异质性的重叠核密度估计图。
    每个指标一个子图，每个子图内包含所有数据集的分布曲线。

    Args:
        df (pd.DataFrame): 包含所有数据集统计信息的DataFrame。
        metrics_to_plot (list): 要绘制的指标列表。
        palette_name (str): Seaborn调色板名称。
        alpha_fill (float): 填充区域的透明度。
        line_width (float): 曲线的宽度。
    """
    num_metrics = len(metrics_to_plot)
    # 根据指标数量调整行数和列数，例如最多3列
    ncols = min(num_metrics, 3)
    nrows = int(np.ceil(num_metrics / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)  # squeeze=False 确保 axes 总是 2D 数组
    axes = axes.flatten()  # 展平成 1D 数组方便索引

    # 获取数据集名称的唯一列表和颜色映射
    unique_datasets = df['dataset'].unique()
    # 如果想手动排序
    # unique_datasets = ['PHEME', 'Politifact', 'WeiboCOVID19', 'Gossipcop']
    # df['dataset'] = pd.Categorical(df['dataset'], categories=unique_datasets, ordered=True)
    # df = df.sort_values('dataset')

    palette = sns.color_palette(palette_name, n_colors=len(unique_datasets))
    dataset_color_map = dict(zip(unique_datasets, palette))

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]

        # --- 核心改动：使用 sns.kdeplot 并设置 hue ---
        sns.kdeplot(data=df, x=metric, hue="dataset", ax=ax,
                    fill=True, common_norm=False,  # common_norm=False 让每个KDE独立归一化
                    palette=dataset_color_map,  # 使用我们定义的颜色
                    alpha=alpha_fill,
                    linewidth=line_width,
                    hue_order=unique_datasets  # 确保颜色和图例顺序一致
                    )

        # --- （可选）添加均值/中位数标记 ---
        for j, ds_name in enumerate(unique_datasets):
            subset = df[df['dataset'] == ds_name]
            if not subset[metric].empty:
                mean_val = subset[metric].mean()
                ax.axvline(mean_val, color=palette[j], linestyle='--', linewidth=1.0, alpha=0.7)
                # 您也可以选择绘制中位数: median_val = subset[metric].median() ... ax.axvline(median_val, ...)
        ax.set_xlim(-1, 10)
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=14)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_xlabel("")  # 先移除X轴标签
        # ax.grid(True, linestyle='--', alpha=0.6)
        ax.grid(axis='y')

        # 移除图例，将在图外统一添加
        ax.get_legend().remove()

        # 隐藏多余的子图（如果存在）
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # --- 添加统一的图例 ---
    handles = [plt.Rectangle((0, 0), 1, 1, color=dataset_color_map[ds]) for ds in unique_datasets]
    fig.legend(handles, unique_datasets, loc='upper center', bbox_to_anchor=(0.5, 1.05 if nrows == 1 else 1.02),
               ncol=len(unique_datasets), frameon=False, fontsize=11)

    # --- 设置统一的X轴标签（如果所有图共享X轴含义） ---
    # fig.text(0.5, 0.01, 'Metric Value', ha='center', va='center', fontsize=12)
    # 或者为最后一个可见的轴设置xlabel
    axes[i].set_xlabel('Metric Value', fontsize=12)

    plt.suptitle("Heterogeneity of Graph Structures Across Datasets", fontsize=16, y=1.1 if nrows == 1 else 1.05)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为图例和标题留空间
    plt.show()


def calculate_graph_width(G, root_node):
    """
    使用BFS计算图的最大宽度。
    对于有向图，宽度定义为在某个层次上拥有最多节点的数量。
    """
    if root_node not in G:
        return 0

    max_width = 0
    q = collections.deque([(root_node, 0)])  # (node, depth)
    visited = {root_node}

    # 统计每个深度的节点数量
    nodes_per_level = collections.defaultdict(int)
    nodes_per_level[0] = 1  # 根节点在深度0

    head = 0  # 模拟队列的头部指针
    q_list = [(root_node, 0)]

    while head < len(q_list):
        curr_node, curr_depth = q_list[head]
        head += 1

        for neighbor in G.successors(curr_node):
            if neighbor not in visited:
                visited.add(neighbor)
                q_list.append((neighbor, curr_depth + 1))
                nodes_per_level[curr_depth + 1] += 1

    if nodes_per_level:
        max_width = max(nodes_per_level.values())

    return max_width


def analyze_single_graph(data, name_for_display):
    """
    计算单个图的核心统计指标。
    """
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    if num_nodes == 0:
        return {
            "dataset": name_for_display, "num_nodes": 0, "num_edges": 0,
            "avg_degree": 0, "graph_depth": 0, "graph_width": 0
        }

    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

    graph_depth = 0
    graph_width = 0

    # 转换为 networkx 图 (确保是有向图)
    G = to_networkx(data, to_undirected=False)

    # 找到根节点 (入度为0的节点)
    in_degree_dict = G.in_degree()
    root_candidates = [node for node, deg in in_degree_dict if deg == 0]

    root_node = -1
    if root_candidates:
        root_node = root_candidates[0]  # 取第一个根节点
    elif num_nodes > 0 and num_edges == 0:  # 如果是孤立节点图，默认0为根
        root_node = 0
    elif num_nodes > 0:  # 如果没有入度为0的节点（如环），则随便选一个，但深度宽度可能不准确
        root_node = 0  # 默认从节点0开始

    if root_node != -1 and root_node in G:
        # 计算深度 (最长最短路径)
        try:
            path_lengths = nx.shortest_path_length(G, source=root_node)
            if path_lengths:
                graph_depth = max(path_lengths.values())
        except nx.NetworkXNoPath:
            # 如果从根节点无法到达所有节点 (图不连通)，深度可能为0或取决于可达部分
            # 为了引言图，我们更关注能到达的结构深度
            graph_depth = 0
        except Exception:  # 其他异常，比如图是空
            graph_depth = 0

        # 计算宽度
        graph_width = calculate_graph_width(G, root_node)
    else:  # 图是空，或者根节点不在图中
        graph_depth = 0 if num_nodes == 0 else 1  # 单节点深度为0
        graph_width = 0 if num_nodes == 0 else 1  # 单节点宽度为1

    return {
        "dataset": name_for_display,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": avg_degree,
        "graph_depth": graph_depth,
        "graph_width": graph_width,
    }


def analyze_all_datasets_for_plot(datasets_dict):
    """
    分析所有数据集并返回一个用于绘图的DataFrame。
    """
    all_stats = []
    for name, dataset in datasets_dict.items():
        print(f"Analyzing dataset: {name}...")
        for i in tqdm(range(len(dataset)), desc=f"Processing {name}"):
            data = dataset[i]
            stats = analyze_single_graph(data, name)
            all_stats.append(stats)

    combined_df = pd.DataFrame(all_stats)

    # 清理异常值：将计算失败的深度/宽度（如果设置为-1或其他特殊值）替换为NaN
    # 或者对于深度和宽度，过滤掉0值（如果是有效单节点图）或者异常值
    combined_df['graph_depth'] = combined_df['graph_depth'].replace(-1, np.nan)
    combined_df['graph_width'] = combined_df['graph_width'].replace(-1, np.nan)

    # 可以对数值进行剪裁或转换，以更好地可视化
    # 例如，深度和宽度可能存在极端值，可以剪裁掉最高百分位
    for col in ['num_nodes', 'graph_depth', 'graph_width', 'avg_degree']:
        if col in combined_df.columns and not combined_df[col].isnull().all():
            upper_bound = combined_df[col].quantile(0.99)  # 裁剪掉最高的1%
            combined_df[col] = combined_df[col].clip(upper=upper_bound)
            # 对于深度和宽度，过滤掉非常小的值（可能是计算异常或单节点图）
            if col in ['graph_depth', 'graph_width']:
                combined_df[col] = combined_df[col].replace(0, np.nan)  # 0深度/宽度可能代表无效或极小的图，可根据实际情况保留或替换

    combined_df = combined_df.dropna(subset=['graph_depth', 'graph_width', 'num_nodes', 'avg_degree'])

    return combined_df


if __name__ == '__main__':
    # data = HugeDataset("./Data/UWeibo/")
    # data = TreeDataset_PHEME("../ACL/Data/pheme/")
    # data = UPFD("./Data/", "gossipcop", "bert")
    # data = TreeDataset_UPFD('./Data/politifact/')
    # root_path = "D://ACLR4RUMOR_datasets//Twitter-COVID19//Twitter-COVID19//Twittergraph"
    # data = CovidDataset('./Data/Twitter-COVID19/Twittergraph')
    # data = TreeDataset("../ACL/Data/DRWeiboV3/")

    # PHEME_NAME = "TwitterCOVID19"
    # COVID_NAME = "WeiboCOVID19"
    #
    # pheme_dataset = CovidDataset("../ACL/Data/Twitter-COVID19/Twittergraph")
    # covid_dataset = CovidDataset("../ACL/Data/Weibo-COVID19/Weibograph")
    #
    # # 分析数据集并获取统计数据
    # pheme_stats_df = analyze_dataset(pheme_dataset, PHEME_NAME)
    # covid_stats_df = analyze_dataset(covid_dataset, COVID_NAME)
    #
    # # --- 打印量化对比的统计摘要 ---
    # print("\n" + "=" * 50)
    # print(f" {PHEME_NAME} 统计摘要:")
    # print("=" * 50)
    # print(pheme_stats_df.describe())
    #
    # print("\n" + "=" * 50)
    # print(f" {COVID_NAME} 统计摘要:")
    # print("=" * 50)
    # print(covid_stats_df.describe())
    # print("\n" + "=" * 50)
    #
    # # --- 可视化对比 ---
    # metrics_to_plot = ["num_nodes", "num_edges", "avg_degree", "graph_depth"]
    # for metric in metrics_to_plot:
    #     plot_comparison(pheme_stats_df, PHEME_NAME, covid_stats_df, COVID_NAME, metric)
    #
    # print("\n所有对比图已生成。请查看弹出的绘图窗口。")
    # plt.tight_layout()
    # plt.show()

    # output_path = "../ACL/Data/ShallowAugData"
    # data = ShallowAugDataset(output_path)

    datasets_to_analyze = {
        "DRWeiboV3": TreeDataset("../ACL/Data/DRWeiboV3/"),
        "Weibo": TreeDataset("../ACL/Data/Weibo/"),
        "WeiboCOVID19": CovidDataset("../ACL/Data/Weibo-COVID19/Weibograph"),
        "PHEME": TreeDataset_PHEME("../ACL/Data/pheme/"),
        "Politifact": TreeDataset_UPFD("../ACL/Data/politifact/"),
        "Gossipcop": TreeDataset_UPFD("../ACL/Data/gossipcop/"),
        "TwitterCOVID19": CovidDataset("../ACL/Data/Twitter-COVID19/Twittergraph"),
    }
    # 执行分析并获取合并后的 DataFrame (与之前相同)
    combined_stats_df = analyze_all_datasets_for_plot(datasets_to_analyze)

    # --- 打印量化对比的统计摘要 (与之前相同) ---
    print("\n" + "=" * 70)
    print(" Overall Statistics Summary Grouped by Dataset:")
    print("=" * 70)
    summary_stats = combined_stats_df.groupby("dataset").agg(['mean', 'std', 'median', 'min', 'max'])
    print(summary_stats)
    print("\n" + "=" * 70)

    # --- 可视化对比 ---
    # metrics_to_visualize = ["graph_depth", "graph_width"]
    metrics_to_visualize = ["graph_depth"]
    print(f"\nGenerating overlaid heterogeneity plots for metrics: {', '.join(metrics_to_visualize)}")

    # 【核心改动】调用新的绘图函数
    plot_heterogeneity_overlaid(combined_stats_df, metrics_to_visualize, palette_name="viridis")

    print("\nAnalysis complete.")
