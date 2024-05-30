import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoFeatureExtractor, \
    AutoModel, AutoTokenizer, ViTForImageClassification
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm


import sys
sys.path.append('..')
from utils.config import Conf as conf
from utils.train_utils import *

device = select_gpu()
set_seed(888)


class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.vit_encoder = AutoModel.from_pretrained(conf.paths('vit-base-patch16-224'))

    def forward(self, img_tensors):
        outputs = self.vit_encoder(pixel_values=img_tensors.to(device))

        last_hidden_state = outputs.last_hidden_state
        cls_vector = last_hidden_state[:, 0, :]
        return cls_vector

    def forward_paths(self, img_paths):
        imgs = [Image.open(img_path) for img_path in img_paths]
        inputs = self.feature_extractor(images=imgs, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        cls_vector = last_hidden_state[:, 0, :]
        return cls_vector

    def img_classify(self, img_path, return_logits=False):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(conf.paths('vit-base-patch16-224'))
        self.vit_img_classify = AutoModelForImageClassification.from_pretrained(conf.paths('vit-base-patch16-224'))
        img = Image.open(img_path)
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = self.vit_img_classify(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predict_class = self.vit_img_classify.config.id2label[predicted_class_idx]
        return predict_class if not return_logits else (predict_class, logits)


class TextualEncoder(nn.Module):
    def __init__(self):
        super(TextualEncoder, self).__init__()
        self.bert_tokenizer = AutoTokenizer.from_pretrained(conf.paths('bert-base-uncased'))
        self.bert_encoder = AutoModel.from_pretrained(conf.paths('bert-base-uncased'))
    
    def forward(self, texts):
        inputs = self.bert_tokenizer(texts, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_encoder(**inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_vector = last_hidden_state[:, 0, :]
        return cls_vector


class CLIP(nn.Module):
    def __init__(self, img_dim, text_dim, fuse_dim) -> None:
        super().__init__()
        self.visual_encoder = VisualEncoder()
        self.textual_encoder = TextualEncoder()
        
        self.W_i2fuse = nn.Linear(img_dim, fuse_dim)
        self.W_t2fuse = nn.Linear(text_dim, fuse_dim)
    
    def get_clip_matrix(self, img_pixel_values, texts):
        img_vectors = self.visual_encoder(img_pixel_values)
        text_vectors = self.textual_encoder(texts)
        img_fuse = self.W_i2fuse(img_vectors) # (batch_size, fuse_dim) or eval: (eval_bs, fuse_dim)
        text_fuse = self.W_t2fuse(text_vectors) # (batch_size, fuse_dim) or eval: (label_bs, fuse_dim)
        matrix = torch.matmul(img_fuse, text_fuse.T) # (bs, bs) or eval: (eval_bs, label_bs)
        # [i, j] stands for the similarity between i-th image and j-th text
        return matrix

    def get_clip_loss(self, img_pixel_values, texts):
        matrix = self.get_clip_matrix(img_pixel_values, texts)
        bs = matrix.shape[0]
        choose_partner_labels = torch.arange(bs).to(device)
        loss_i2t = F.cross_entropy(matrix, choose_partner_labels)
        loss_t2i = F.cross_entropy(matrix.T, choose_partner_labels)
        loss = (loss_i2t + loss_t2i) / 2
        return loss


def load_cifar10_and_transform():
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_dataset = CIFAR10(root=conf.paths('cifar10'), train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root=conf.paths('cifar10'), train=False, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    # classes = train_dataset.classes
    # print(classes)
    print('train/test samples: ', len(train_dataset), len(test_dataset))
    return train_dataset, train_loader, test_dataset, test_loader


class CLIPTrainer(nn.Module):
    def __init__(self, model, opt, grad_norm=1.0, train_id=None) -> None:
        super().__init__()
        # memorize basic info
        self.model = model
        self.opt = opt
        self.grad_norm = grad_norm
        self.train_id = train_id
        # memorize dataset
        self.dataset, self.loader, self.test_dataset, self.test_loader \
            = load_cifar10_and_transform()


    def normal_backward_and_step(self, loss):
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_norm)
        self.opt.step()
        return

    def train_epoch(self):
        self.model.train()

        train_log, step = [], 0
        eval_step = self.loader.batch_size * (len(self.loader) // 10)
        for img_pixel_values, labels in tqdm(self.loader, desc=f'Train Clip Epoch-1'):
            if step % eval_step == 0:
                self.eval()
            texts = [self.dataset.classes[l] for l in labels]
            loss = self.model.get_clip_loss(img_pixel_values, texts)
            self.normal_backward_and_step(loss)
            step += labels.shape[0]
            train_log.append({'step': step, 'loss': loss.item()})
        
        plot_loss(train_log, keys=['loss'], train_id=self.train_id)
        return
    
    def eval(self):
        labels_space = self.test_dataset.classes
        self.model.eval()
        with torch.no_grad():
            all_labels, all_predictions = [], []
            for img_pixel_values, labels in tqdm(self.test_loader, desc=f'Test Clip Epoch-1'):
                all_labels.append(labels)
                matrix = self.model.get_clip_matrix(img_pixel_values, labels_space) # (eval_bs, label_bs)
                predictions = torch.argmax(matrix, dim=1)
                all_predictions.append(predictions)
                """for idx, prediction in enumerate(predictions):
                    print(f"Image {idx} is predicted as {labels[prediction]}")"""
            all_labels, all_predictions = torch.cat(all_labels), torch.cat(all_predictions)
            eval_classify(all_labels, all_predictions, labels_space)
        return
    

if __name__ == '__main__':
    # 1. test vit
    # ve = VisualEncoder()
    # print(ve.forward('hello_vit.jpeg'))
    
    # 2. test trainer of clip
    train_id = 'clip_vit_bert_' + get_time()
    clip_model = CLIP(img_dim=768, text_dim=768, fuse_dim=1024).to(device)
    opt = get_optimizer(clip_model)
    clip_trainer = CLIPTrainer(clip_model, opt, train_id=train_id)
    clip_trainer.train_epoch()