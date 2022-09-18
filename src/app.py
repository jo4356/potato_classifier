from distutils.log import debug
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
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



def parse_content(content, filename):
    return html.Div([
        html.H5(filename),
        html.Img(src=content)
    ])


app.layout = dbc.Container(
    [dbc.Row([html.H1("Potato Classifier")]),
     dbc.Row([html.P("Upload Image:")]),
     dbc.Row([
        dbc.Col(
            html.Div([dcc.Upload(html.Button("Upload"), id="upload_image")])
                ),
        dbc.Col(html.Button("Predict!"), 
                style={"border": "#9E0600", "border-radius":"10px", "width":"10%"})]),
    
    dbc.Row([html.Div(id="output_image")])
              
              
              
    ]
)




@app.callback(Output("output_image", "children"),
                Input("upload_image", "contents"),
                State("upload_image", "filename"))


def outdate_output(content, file_name):
    children = parse_content(content, file_name)

    return children





if __name__ == '__main__':
    app.run_server(debug=True)
