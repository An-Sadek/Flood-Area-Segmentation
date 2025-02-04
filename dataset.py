import os
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL import ImageFilter

import tensorflow as tf
from tensorflow import keras
from keras import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FloodDataset(utils.Sequence):

    def __init__(self, root, batch_sz: int, img_sz: int|tuple, is_trainset=True, channels=1):
        super().__init__() # Khoi tao lop utils.Sequence
        
        # Kiem tra duong dan root 
        assert os.path.exists(root), f"Duong dan khong ton tai: {root}"
        self.root = root

        # Doc duong dan
        metadata_path = os.path.join(root, "metadata.csv")
        assert os.path.exists(metadata_path), f"Khong tim thay metadata.csv: {metadata_path}"
        self.metadata = pd.read_csv(metadata_path) # Cot 0 la Image, cot 1 la Mask

        # Batch size
        self.batch_sz = batch_sz

        # Kiem tra img_sz va dieu chinh
        if isinstance(img_sz, int):
            self.img_sz = (img_sz, img_sz)
        if isinstance(img_sz, tuple):
            assert len(img_sz) == 2, f"Kich thuoc cua tuple khong hop le {len(img_sz)}"
            assert img_sz[0] == img_sz[1], "Chieu dai phai bang chieu rong"
            self.img_sz = img_sz

        # Danh dau la train set hoac test set (khong co nhan)
        self.is_trainset = is_trainset

        # RGB hoac mau xam
        assert channels in [1, 3], "Khong gian mau la 1 (anh xam) hoac 3 (RGB"
        self.channels = channels

    def __len__(self):
        return math.ceil(len(self.metadata) / self.batch_sz)

    def __getitem__(self, idx):
        # Dieu chinh lai idx neu lay cai cuoi cung
        idx = self.__len__() -1 if idx < 0 else idx
        start_idx = idx * self.batch_sz
        end_idx = min(start_idx + self.batch_sz, len(self.metadata))

        # Tao batch img thu idx
        # Su dung try de tranh lon xon voi mask
        try:
            img_dir = os.path.join(self.root, "Image")
            curr_img_paths = self.metadata.iloc[start_idx: end_idx, 0] # Cot Image
            curr_img_paths = [os.path.join(img_dir, img_path) for img_path in curr_img_paths]
            curr_batch_img = np.zeros(
                shape=(len(curr_img_paths), self.img_sz[0], self.img_sz[1], self.channels),
                dtype=float
            )
            
            for i in range(len(curr_img_paths)):
                img = Image.open(curr_img_paths[i])
                img = img.resize(self.img_sz) # Chuan hoa kich thuoc
                if self.channels == 1:
                    img = ImageOps.grayscale(img) # Chuyen anh xam neu channel = 1
                img = img.filter(ImageFilter.BoxBlur(1)) # Lam mo trung binh 3 x3
                img = np.asarray(img) # Chuyen ve np array, mang 2d
                img = img / 255 # Chuan hoa min-max
                img = np.expand_dims(img, axis=2) # Chuyen ve mang 3d
                
                curr_batch_img[i] = img
                
        except Exception as e:
            print("Co loi xay ra", e)

        curr_batch_img = tf.constant(curr_batch_img, dtype=float)

        # Tra ve neu la test set
        if not self.is_trainset:
            return (curr_batch_img,)

        # Tao batch mask thu idx
        try:
            mask_dir = os.path.join(self.root, "Mask")
            curr_mask_paths = self.metadata.iloc[start_idx: end_idx, 1] # Cot Mask
            curr_mask_paths = [os.path.join(mask_dir, mask_path) for mask_path in curr_mask_paths]
            curr_batch_mask = np.zeros(
                shape=(len(curr_mask_paths), self.img_sz[0], self.img_sz[1], self.channels),
                dtype=int
            )
    
            for i in range(len(curr_mask_paths)):
                mask = Image.open(curr_mask_paths[i]).convert("L")
                mask = mask.resize(self.img_sz)
                mask = np.asarray(mask)
                mask = np.where(mask >= 127, 1, 0) # 1 = trang (lu), 2 = den (khong lu)
                mask = mask[:, :, np.newaxis] # Chuyen ve dang 3 chieu
                
                curr_batch_mask[i] = mask
                
        except Exception as e:
            print("Co loi xay ra: ", e)

        curr_batch_mask = tf.constant(curr_batch_mask, dtype=float)

        return (curr_batch_img, curr_batch_mask)
        
        
if __name__ == "__main__":
    dataset = FloodDataset("../FloodDataset", 16, 576)
    print("So luong anh: ", len(dataset.metadata))
    print("So luong batch: ", len(dataset))
    
    print("\nidx=0")
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)

    print("\nidx=18")
    print(dataset[18][0].shape)
    print(dataset[18][1].shape)

    print("\nidx=-1")
    print(dataset[-1][0].shape)
    print(dataset[-1][1].shape)
    
        