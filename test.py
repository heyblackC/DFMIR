import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from PIL import Image
import numpy as np
from models.voxelmorph.torchvoxelmorph.layers import SpatialTransformer
import torchvision
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip, ToTensor, Normalize

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # 获取label的存储名称
    images_A_names = os.listdir(os.path.join(opt.dataroot, opt.phase + 'A'))
    images_A_names.sort()

    for i, data in enumerate(dataset):
        print(i)
        print(data["A_paths"][0])
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            model.eval()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        #visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths


        def check_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)
            return

        def save_tensor_image(img, save_path, file_name):
            # 假设img的shape是（1，1，height，width）
            # 数值范围是-1 到 1
            img = img[0][0].cpu().float().numpy()
            img = (img + 1) / 2.0 * 255.0
            slic = Image.fromarray(img.astype(np.uint8))  # 就是实现array到image的转换
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            slic.save(os.path.join(save_path, file_name))

        # 打开并读入label
        img = Image.open(os.path.join(str(opt.dataroot), 'trainA_label', str(images_A_names[i])))
        transforms = []
        #transforms.append(CenterCrop(224))
        transforms.append(ToTensor())
        transforms = Compose(transforms)
        img = transforms(img)
        img = img.unsqueeze(0)

        # 形变
        idt_B = model.netG(model.real_B)
        y_pred2 = model.netR(model.real_A, model.real_B, registration=True)

        spatialTransformer = SpatialTransformer([256, 256], mode='nearest')
        registered_label2 = spatialTransformer.forward(img, y_pred2[1].cpu())

        # 检查存储目录并保存到硬盘
        check_dir(os.path.join(opt.dataroot, 'deform_label'))
        img_filename2 = os.path.join(opt.dataroot, 'deform_label', str(images_A_names[i]))
        torchvision.utils.save_image(registered_label2, img_filename2)

        check_dir(os.path.join(opt.dataroot, 'deform_trainA'))
        img_filename3 = os.path.join(opt.dataroot, 'deform_trainA', str(images_A_names[i]))
        torchvision.utils.save_image(y_pred2[0]/2+0.5, img_filename3)

