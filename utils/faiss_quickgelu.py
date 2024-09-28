import os
import json
import numpy as np
from PIL import Image
import faiss
import torch
import open_clip
import matplotlib.pyplot as plt
import math
from langdetect import detect

class Myfaiss_quigelu:
    def __init__(self, bin_file: str, id2img_fps, device, translater, clip_backbone):
        self.device = device
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = id2img_fps
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_backbone, pretrained='dfn5b')
        self.model.to(self.device).eval()  # Set model to evaluation mode
        self.tokenizer = open_clip.get_tokenizer(clip_backbone)
        self.translater = translater

    def load_bin_file(self, bin_file: str):
        index = faiss.read_index(bin_file)
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
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.amp.autocast(self.device):
            image_features = self.model.encode_image(image_tensor).cpu().numpy()
        return self.normalize(image_features)

    def image_search(self, id_query, k):
        query_feats = self.index.reconstruct(id_query).reshape(1, -1)
        query_feats = query_feats.astype(np.float32)
        if self.device == "cuda":
            query_feats = torch.tensor(query_feats).cuda()
            query_feats = query_feats.cpu().numpy()
        else:
            query_feats = torch.tensor(query_feats).numpy()

        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = [self.id2img_fps.get(i) for i in idx_image]
        image_paths = infos_query

        return scores, idx_image, infos_query, image_paths

    def text_search(self, text, k):
        if detect(text) == 'vi':
            text = self.translater(text)
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).cpu().numpy().astype(np.float32)
        text_features = self.normalize(text_features)
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()
        infos_query = [self.id2img_fps.get(i) for i in idx_image]
        image_paths = infos_query
        return scores, idx_image, infos_query, image_paths
