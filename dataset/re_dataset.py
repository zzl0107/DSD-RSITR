import json
import os
import torch
from torch.utils.data import Dataset
from jieba import analyse
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        # n = 0
        # for ann in self.ann:
        #     img_id = ann['image_id']
        #     if img_id not in self.img_ids.keys():
        #         self.img_ids[img_id] = n
        #         n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        # t = analyse.extract_tags(caption, topK=4, withWeight=False)
        # ii = caption.split(' ')
        # k = ""
        # fl = 0
        # for j in range(len(ii)):
        #     if fl == 1:
        #         k += " "
        #     fl = 1
        #     if ii[j] not in t:
        #         k += "[MASK]"
        #     else:
        #         k += ii[j]
        #
        # mask_text = pre_caption(k, self.max_words)
        # print('caption: {}'.format(caption))
        # print('mask_texts: {}'.format(mask_texts))

        # label = torch.tensor(ann['label'])

        ## if no need label, set value to zero or others:
        # label = 0
        # return image, caption, mask_text, self.img_ids[ann['image_id']], label
        # return image, caption, self.img_ids[ann['image_id']], label
        return image, caption


class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        # self.mask_text = []
        self.image = []
        # self.image_data = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

                # t = analyse.extract_tags(caption, topK=4, withWeight=False)
                # ii = caption.split(' ')
                # k = ""
                # fl = 0
                # for j in range(len(ii)):
                #     if fl == 1:
                #         k += " "
                #     fl = 1
                #     if ii[j] not in t:
                #         k += "[MASK]"
                #     else:
                #         k += ii[j]
                # self.mask_text.append(pre_caption(k, self.max_words))

                # image_path = os.path.join(self.image_root, ann['image'])
                # image = Image.open(image_path).convert('RGB')
                # image = self.transform(image)
                # self.image_data.append(image.unsqueeze(dim=0))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
