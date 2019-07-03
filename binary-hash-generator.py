'''
input:
    1. trained models
    2. images or images to be convert into binary Hashing
    3. length of binary bits output

output:
    1. image hash
    2.

'''
import argparse
from PIL import Image
from torchvision import transforms
import torch
import glob
import os
from net import AlexNetPlusLatent
from torch.autograd import Variable
import numpy as np


def main(args):
    if args.mode == "single":
        generate_hash_single(args)
    else:
        generate_hash_batch(args)

def generate_hash_single(args):
    print("single")

def generate_hash_batch(args):
    dir = glob.glob(args.imagedir+'*.jpg')

    transform = transforms.Compose(
        [transforms.Resize([227,227]),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    all_images = []
    for id, file in enumerate(dir):
        print(str(id)+"/"+str(len(dir))+" "+file)
        image = Image.open(file)

        if image.mode != "RGB":
            rgbimg = Image.new("RGB", image.size)
            rgbimg.paste(image)
            image = rgbimg

        image = transform(image)
        all_images.append(image)

    binary_code = binary_output(all_images)
    dir = np.asarray(dir)
    np.save("output/"+args.output+"_filename", dir)
    torch.save(binary_code, "output/"+args.output)

def binary_output(dataloader):
    net = AlexNetPlusLatent(args.bits)
    net.load_state_dict(torch.load(args.modelpath))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    full_batch_output = torch.cuda.FloatTensor()
    full_batch_label = torch.cuda.LongTensor()
    net.eval()

    for batch_idx, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        inputs = inputs.unsqueeze(0)
        outputs, _ = net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
    return torch.round(full_batch_output)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='hashing image to binary code')

    parser.add_argument('--bits', type=int, default=48, metavar='bts',
                        help='binary bits')
    parser.add_argument('--mode', type=str, default="single", metavar='bts',
                        help='hash mode: single/batch')
    parser.add_argument('--imagedir', type=str, default="images", metavar='bts',
                        help='directory of images')
    parser.add_argument('--imagefile', type=str, metavar='bts',
                        help='path of an image file')
    parser.add_argument('--modelpath', type=str, default='', metavar='P',
                        help='path directory to the model')
    parser.add_argument('--output', type=str, default="output", metavar='bts',
                        help='file name for output binary')

    args = parser.parse_args()

    main(args)
