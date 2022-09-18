import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc


app = Dash(__name__)
server = app.server
app.title = "Potato Classifier"




IMAGE_SIZE = 100


def predict(image_path):

    transformer = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                  transforms.RandomRotation(degrees=20),
                                  transforms.ToTensor()])


    image = Image.open(image_path)
    image = transformer(image)
    image = image[None, :]

    potato_model = models.squeezenet1_0(num_classes=4)
    potato_model.features[0] = nn.Conv2d(3, 100, kernel_size=(3, 3), stride=(2, 2))
    potato_model.features[3].squeeze = nn.Conv2d(100, 16, kernel_size=(1, 1), stride=(1, 1))
    potato_model.load_state_dict(torch.load("../model/model.pt"))

    pred_prob = potato_model(image)
    labels = ['red', 'red_washed', 'sweet', 'white']
    prediction = labels[pred_prob.argmax()]
    

    return pred_prob, labels, prediction



app.layout = dbc.Container(
    [dbc.Row([html.P("123")])]


)




if __name__ == '__main__':
    app.run_server()
