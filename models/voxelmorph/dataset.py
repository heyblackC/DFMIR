import os
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize
import random

class dataset_single(data.Dataset):
  def __init__(self, opts, setname, input_dim):
    self.dataroot = opts.dataroot
    images = os.listdir(os.path.join(self.dataroot, opts.phase + setname))
    self.img = [os.path.join(self.dataroot, opts.phase + setname, x) for x in images]
    self.size = len(self.img)
    self.input_dim = input_dim

    # setup image transformation
    transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms.append(CenterCrop(opts.crop_size))
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    print('%s: %d images'%(setname, self.size))
    return

  def __getitem__(self, index):
    data = self.load_img(self.img[index], self.input_dim)
    return data

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name).convert('RGB')
    img = self.transforms(img)
    if input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    return img

  def __len__(self):
    return self.size

class medical_dataset_unpair(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    # A
    images_A = os.listdir(os.path.join(self.dataroot, opts.phase + 'A'))
    images_A.sort()
    self.A = [os.path.join(self.dataroot, opts.phase + 'A', x) for x in images_A]

    # B
    images_B = os.listdir(os.path.join(self.dataroot, opts.phase + 'B'))
    images_B.sort()
    self.B = [os.path.join(self.dataroot, opts.phase + 'B', x) for x in images_B]

    # C 其实就是原始模态T1
    images_C = os.listdir(os.path.join(self.dataroot, 'trainA'))
    images_C.sort()
    self.C = [os.path.join(self.dataroot, 'trainA', x) for x in images_C]

    self.A_size = len(self.A)
    self.B_size = len(self.B)
    self.dataset_size = max(self.A_size, self.B_size)
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms = [(CenterCrop(opts.crop_size))]
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5], std=[0.5]))
    self.transforms = Compose(transforms)
    print('A: %d, B: %d images'%(self.A_size, self.B_size))
    return

  def __getitem__(self, index):
    if self.dataset_size == self.A_size:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      # data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    else:
      data_A = self.load_img(self.A[index], self.input_dim_A)
      data_B = self.load_img(self.B[index], self.input_dim_B)
    data_C = self.load_img(self.C[index], self.input_dim_B)
    return data_A, data_B, data_C

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name)
    img = self.transforms(img)
    return img

  def __len__(self):
    return self.dataset_size


class medical_dataset_patient_site_random(data.Dataset):
  def __init__(self, opts):
    self.dataroot = opts.dataroot

    site_patient_names = os.listdir(os.path.join(self.dataroot))

    self.A = []
    self.B = []
    for dir in site_patient_names:
      # A
      images_A = os.listdir(os.path.join(opts.dataroot, str(dir), "t1"))
      A_path = [os.path.join(self.dataroot, str(dir), "t1", x) for x in images_A]
      self.A.append(A_path)

      # B
      images_B = os.listdir(os.path.join(self.dataroot, str(dir), "t2"))
      B_path = [os.path.join(self.dataroot, str(dir), "t2", x) for x in images_B]
      self.B.append(B_path)

    # # C 其实就是原始模态T1
    # images_C = os.listdir(os.path.join(self.dataroot, 'trainA'))
    # self.C = [os.path.join(self.dataroot, 'trainA', x) for x in images_C]

    # self.A_size = len(self.A)
    # self.B_size = len(self.B)
    self.dataset_size = len(self.A) *len(self.A[0])
    self.dir_size = len(self.A[0])
    self.input_dim_A = opts.input_dim_a
    self.input_dim_B = opts.input_dim_b

    # setup image transformation
    # transforms = [Resize((opts.resize_size, opts.resize_size), Image.BICUBIC)]
    transforms = [(CenterCrop(opts.crop_size))]
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5], std=[0.5]))
    self.transforms = Compose(transforms)
    # print('A: %d, B: %d images'%(self.len(self.A), self.B_size))
    return

  def __getitem__(self, index):
    # 从二维到1维

    data_A = self.load_img(self.A[int(index/self.dir_size)][index%self.dir_size], self.input_dim_A)
    # data_B = self.load_img(self.B[random.randint(0, self.B_size - 1)], self.input_dim_B)
    data_B = self.load_img(self.B[random.randint(0, len(self.B) - 1)][index%self.dir_size], self.input_dim_B)
    return data_A, data_B, data_A

  def load_img(self, img_name, input_dim):
    img = Image.open(img_name)
    img = self.transforms(img)
    return img

  def __len__(self):
    return self.dataset_size