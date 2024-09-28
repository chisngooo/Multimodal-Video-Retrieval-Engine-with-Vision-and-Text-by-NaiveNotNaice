import os
import json
import numpy as np
import faiss
import torch
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import math
from langdetect import detect
from sentence_transformers import SentenceTransformer

class Myfaiss_robert:
    def __init__(self, bin_file: str, bin_file2: str, id2img_fps, device, translater):
        self.device = device
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = id2img_fps
        self.index2 = self.load_bin_file(bin_file2)
        self.model = SentenceTransformer("sentence-transformers/all-roberta-large-v1")
        self.translater = translater

    def load_bin_file(self, bin_file: str):
        index = faiss.read_index(bin_file)
        #if self.device == "cuda":
        #    res = faiss.StandardGpuResources()
        #   index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths) / columns))

        for i in range(1, columns * rows + 1):
            if i - 1 < len(image_paths):
                img_path = image_paths[i - 1]
                try:
                    img = plt.imread(img_path)
                except FileNotFoundError:
                    print(f"Image file {img_path} not found.")
                    continue

                ax = fig.add_subplot(rows, columns, i)
                ax.set_title('/'.join(img_path.split('/')[-3:]))
                plt.imshow(img)
                plt.axis("off")
        plt.show()

    def normalize(self, x):
        return normalize(x, norm='l2')

    def extract_text_features(self, text):
        if detect(text) == 'vi':
            text = self.translater(text)
        text_features = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        text_features = text_features.astype(np.float32)
        return text_features

    def image_search(self, id_query, k):
        query_feats = self.index2.reconstruct(id_query).reshape(1, -1)
        query_feats = query_feats.astype(np.float32)
        if self.device == "cuda":
            query_feats = torch.tensor(query_feats).cuda()
            query_feats = query_feats.cpu().numpy()
        else:
            query_feats = torch.tensor(query_feats).numpy()

        scores, idx_image = self.index2.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = [self.id2img_fps.get(i) for i in idx_image]
        image_paths = infos_query

        return scores, idx_image, infos_query, image_paths

    def text_search(self, text, k):
        text_features = self.extract_text_features(text)
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        infos_query = [self.id2img_fps.get(i) for i in idx_image]
        image_paths = infos_query
        return scores, idx_image, infos_query, image_paths
