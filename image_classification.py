import os
import time

import torch
from torch import nn
from PIL import Image
from torchvision.models import resnet50
from torchvision import transforms

import streamlit as st

def fn_upload_file():
    uploaded_file = st.file_uploader("Choose an image", accept_multiple_files=False)

    if uploaded_file is not None:

        return uploaded_file

def progress_bar_example():
    st.write("Loading Predictions")
    my_bar = st.progress(0)
    num = 25

    for i in range(num):
        time.sleep(0.1)
        my_bar.progress( (i+1)/num )

def load_model():
    model = resnet50(pretrained = True)
    model.eval()

    return model

def read_classes(filename):
    with open(filename) as f:
        classes = [line.strip() for line in f.readlines()]

    return classes

def predict_image(img, model, labels):

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    out = model(batch_t)
    _, indices = torch.sort(out, descending=True)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    index = indices[0][:1].item()
    label = labels[index].split(":")[-1]
    percent = percentage[index].item()

    return label, percent
    
if __name__ == "__main__":

    uploaded_filename = fn_upload_file()
    run_prediction_button = st.button("Run Prediction")

    if(uploaded_filename is not None):
        print(uploaded_filename.name)
        img = Image.open(os.path.join('data', str(uploaded_filename.name)))

        model = load_model()
        labels = read_classes('data/imagenet_classes.txt')

        if(run_prediction_button == True):
            progress_bar_example()
            predicted_label, confidence = predict_image(img, model, labels)

            st.image(img)
            st.write("The predicted label is: ", predicted_label)
            st.write("The confidence of AI model is: ", confidence)



    
    
