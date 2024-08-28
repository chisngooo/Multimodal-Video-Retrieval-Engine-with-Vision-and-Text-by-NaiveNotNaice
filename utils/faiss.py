import faiss
import matplotlib.pyplot as plt
import math
import numpy as np
import clip
from langdetect import detect
import torch

class Myfaiss:
    def __init__(self, bin_file: str, id2img_fps, device, translater, clip_backbone="ViT-B/32"):
        self.device = device
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = id2img_fps
        self.model, _ = clip.load(clip_backbone, device=self.device)
        self.translater = translater

    def load_bin_file(self, bin_file: str):
        """Load FAISS index from binary file and move to GPU if available."""
        index = faiss.read_index(bin_file)
        if self.device == "cuda":
            # Convert to GPU index if GPU is available
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index
    
    def show_images(self, image_paths):
        """Display images using matplotlib."""
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths) / columns))

        for i in range(1, columns * rows + 1):
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
        """Normalize vectors to unit length."""
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    def image_search(self, id_query, k): 
        """Search for images based on a query ID."""
        query_feats = self.index.reconstruct(id_query).reshape(1, -1)
        query_feats = query_feats.astype(np.float32)

        # Normalize query features
        query_feats = self.normalize(query_feats)

        if self.device == "cuda":
            query_feats = torch.tensor(query_feats).cuda()
            # Move to CPU for FAISS search
            query_feats = query_feats.cpu().numpy()
        else:
            query_feats = torch.tensor(query_feats).numpy()

        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = [self.id2img_fps.get(i) for i in idx_image]
        image_paths = infos_query
        
        return scores, idx_image, infos_query, image_paths
    
    def text_search(self, text, k):
        """Search for images based on a text query."""
        if detect(text) == 'vi':
            text = self.translater(text)

        # Text features extraction
        text_tokens = clip.tokenize([text]).to(self.device)
        text_features = self.model.encode_text(text_tokens).cpu().detach().numpy().astype(np.float32)

        # Normalize text features
        text_features = self.normalize(text_features)

        # Searching
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()

        # Get image paths
        infos_query = [self.id2img_fps.get(i) for i in idx_image]
        image_paths = infos_query

        return scores, idx_image, infos_query, image_paths
