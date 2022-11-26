import torch
import torch.nn as nn
from torchvision import transforms
from PIL import ImageFilter, ImageOps, Image, ImageChops
import numpy as np
import sympy
from utils.positional_encoding import PositionalEncoding1d, PositionalEncoding2d
from model import Im2LaTeXModel
import math
import matplotlib.pyplot as plt
import os
import subprocess 

ENCODER_OUT_SIZE = 128
EMBEDDING_DIM = 128
NUM_LAYERS = 6
NUM_HEADS = 8
FEEDFORWARD_DIM = 256 
DROPOUT = 0.1
MAX_LEN = 300

# Load model
device = torch.device("cpu")
cpkt = torch.load('./save_models/test.pth', map_location=device)
vocab_dict = cpkt['vocab_dict']
model = Im2LaTeXModel(ENCODER_OUT_SIZE, EMBEDDING_DIM, len(vocab_dict), NUM_LAYERS, NUM_HEADS, FEEDFORWARD_DIM, DROPOUT, MAX_LEN, device, vocab_dict)
print("Load Resource:")
print("Load Model: ",model.load_state_dict(cpkt['model']))
print("Vocab size: " ,len(vocab_dict))

invaling_set = set([vocab_dict["{"], vocab_dict["}"], vocab_dict["_"], vocab_dict["\\!"], vocab_dict["~"], vocab_dict["."], vocab_dict["\\cdot"], vocab_dict["\\,"], vocab_dict["\\"],vocab_dict["\\;"], vocab_dict[","]])
#convert to tensor
invaling_set = torch.tensor(list(invaling_set), dtype=torch.long, device=device)


def predict(model, image, vocab_dict, device):
    model.eval()
    image = image.to(device)
    encoder_out = model.Encoder(image)
    # shape of encoder_out: (batch_size, height * width, encoder_out_size)
    outputs = torch.zeros((encoder_out.shape[0], model.max_len), dtype=torch.long, device=model.device)
    # shape of outputs: (batch_size, max_len)
    outputs[:, 0] = 1  # <SOS> token

    n = 10
    for i in range(1, model.max_len):
        # if the last n tokens are in the invalid set, fill the rest with { token
        if i > n and torch.all(torch.isin(outputs[:, i-n:i], invaling_set)):
                outputs[:, i:] = vocab_dict['{']
                break

        logits = model.Decoder(encoder_out, outputs[:, :i])
        
        # shape of logits: (i, batch_size, vocab_size)
        preds = torch.argmax(logits, dim=-1)
        # shape of preds: (i, batch_size)
        outputs[:, i] = preds[-1]

        # if all sentences are finished, break
        count = 0
        for j in range(encoder_out.shape[0]):
            if 2 in preds[:, j]:
                count += 1
        if count == encoder_out.shape[0]:
            break
    
    # set all the tokens after <EOS> to 0
    for output in outputs:
        for i, token in enumerate(output):
            if token == 2:
                output[i+1: ] = 0
                break

    return outputs


def detokenize(pred_tensor, vocab_dict):
    idx_to_token = {v: k for k, v in vocab_dict.items()}
    # convert tensor to list
    pred_list = pred_tensor.tolist()
    sentences = []
    for sequence in pred_list:
        # remove <PAD> token
        sequence = [token for token in sequence if token not in {0, 1, 2}]
        # convert to string
        sequence = [idx_to_token[token] for token in sequence]
        sentences.append(" ".join(sequence))
    return sentences


def read_image(image_name, scale) -> torch.Tensor:
    # conver to grayscale
    image = Image.open(image_name).convert('L')
    
    # convert the image to white background and black text if the background is dark
    if image.getpixel((0, 0)) < 128:
        image = ImageOps.invert(image)

    # if the gray pixel > 200, set it to 255
    image = image.point(lambda x: 255 if x > 200 else x)
        
    # convert to tensor
    Width, Height = image.size
    new_size = (math.floor(Width * scale),math.floor(Height * scale))
    image = image.resize(new_size, Image.LANCZOS)
    image = image.filter(ImageFilter.SHARPEN)
    image.save("./images/resized.png")
    image = remove_white_border("./images/resized.png")
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    # white padding
    image_tensor = torch.nn.functional.pad(image_tensor, (3, 10, 3, 10), "constant", 1)

    # save the image
    image = transforms.ToPILImage()(image_tensor)
    image.save("./images/cropped.png")

    return image_tensor


def confidence_score(prediction_sequence, processed_user_input_image):
    # prepare preview image
    prepare_sequence = "$$" + prediction_sequence + "$$"
    try:
        sympy.preview(prepare_sequence, euler=False, viewer='file', filename='./images/preview.png', output='png')
    except:
        return 0
    preview_image = Image.open("./images/preview.png").convert('L')
    preview_image = preview_image.resize(preview_image.size, Image.LANCZOS)
    preview_image = preview_image.filter(ImageFilter.SHARPEN)
    preview_image.save("./images/preview.png")

    # parepare user input image
    user_input_image = remove_white_border(processed_user_input_image).convert('L')

    return mse(user_input_image, preview_image)


def mse(image1, image2):
    diff = ImageChops.difference(image1, image2)
    h = diff.histogram()
    sq = (value*((idx%256)**2) for idx, value in enumerate(h))
    sum_of_squares = sum(sq)
    rms = np.sqrt(sum_of_squares/float(image1.size[0] * image1.size[1]))
    return rms


def remove_white_border(image):
    im = Image.open(image).convert('L')
    bg = Image.new(im.mode, im.size, (255))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    return im.crop(bbox)


def calibrate():
    # Start predicting
    scale = 1.0
    min_scale = 0.1
    max_try = 30
    step = (scale - min_scale) / max_try
    k = max_try // 5
    os.system("screencapture -i ./images/test.png")

    candidates = []
    for _ in range(max_try):
        image = read_image('./images/test.png', scale)
        image_batch = image.unsqueeze(0)
        output = predict(model, image_batch, vocab_dict, device)
        detokenized_out = detokenize(output, vocab_dict)
        pred = detokenized_out[0]
        print("\nPrediction: ")
        print(pred)
        score =  confidence_score(pred, "./images/cropped.png")
        print("score: ", score)
        print("scale: ", scale)
        candidates.append((pred, score, scale))
        scale -= step

    topk = sorted(candidates, key=lambda x: x[1], reverse= True)[:k]
    pred = topk[0]
    scale = pred[2]

    print("\nBest Prediction: ")
    print(pred[0])
    print("score: ", pred[1])
    print("scale: ", pred[2])

    # copy to clipboard
    subprocess.run("pbcopy", text=True, input=pred[0])

    return "ratio set to: " + str(round(scale, 2)) + ", result copied", scale


def convert(scale):
    os.system("screencapture -i ./images/test.png")
    scale_range = [-0.015, 0, 0.015]
    candidates = []
    for i in range(3):
        image = read_image('./images/test.png', scale + scale_range[i])
        image_batch = image.unsqueeze(0)
        output = predict(model, image_batch, vocab_dict, device)
        detokenized_out = detokenize(output, vocab_dict)
        pred = detokenized_out[0]
        score = confidence_score(pred, "./images/cropped.png")
        print("\nPrediction: ")
        print(pred)
        print("score: ", score)
        print("scale: ", scale + scale_range[i])

        candidates.append((pred, score, scale + scale_range[i]))

    # if all the predictions are infeasible, return the shortest one
    topk = sorted(candidates, key=lambda x: x[1], reverse=True)[:1]
    pred = topk[0]

    print("\nBest Prediction: \n", pred[0])
    subprocess.run("pbcopy", text=True, input=pred[0])
    return pred[0]


if __name__ == "__main__":
    while True:
        # print prompt
        print()
        print("Please enter the command: ")
        print("1. convert")
        print("2. calibrate")
        print("3. set scale")
        print("4. exit")
        # read input
        command = input("Command: ")
        if command == "1":
            # read scale from file
            with open("./assets/config.txt", "r") as f:
                scale = float(f.readline())
            print("Current scale: ", scale)
            print("Converting...")
            convert(scale)
            print("Result copied to clipboard")
        elif command == "2":
            print("Calibrating...")
            result, scale = calibrate()
            print(result)
            with open("./assets/config.txt", "w") as f:
                f.write(str(scale))
        elif command == "3":
            scale = float(input("Scale: "))
            with open("./assets/config.txt", "w") as f:
                f.write(str(scale))
        elif command == "4":
            break
        else:
            print("Invalid command")


