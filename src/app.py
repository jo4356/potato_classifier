import re
import base64
import dash_bootstrap_components as dbc
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
from dash import Dash, html, dcc, Output, Input, State, ctx
from io import BytesIO



app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY],
    prevent_initial_callbacks=True, suppress_callback_exceptions=True)

server = app.server
app.title = "Potato Classifier"




IMAGE_SIZE = 100


def predict(uploaded_image):

    transformer = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                                  transforms.RandomRotation(degrees=20),
                                  transforms.ToTensor()])


    image = re.sub('^data:image/.+;base64,', '', uploaded_image)
    image = Image.open(BytesIO(base64.b64decode(image)))
    image = transformer(image)
    image = image[None, :]

    potato_model = models.squeezenet1_0(num_classes=4)
    potato_model.features[0] = nn.Conv2d(3, 100, kernel_size=(3, 3), stride=(2, 2))
    potato_model.features[3].squeeze = nn.Conv2d(100, 16, kernel_size=(1, 1), stride=(1, 1))
    potato_model.load_state_dict(torch.load("model/model.pt"))

    pred_prob = potato_model(image)
    labels = ['red', 'red_washed', 'sweet', 'white']
    prediction = labels[pred_prob.argmax()]
    

    return prediction






app.layout = dbc.Container(
    [dbc.Row([html.H1("Potato Classifier")]),
     dbc.Row([html.P("Upload Image:")]),
     dbc.Row([
        dbc.Col(
            html.Div([dcc.Upload(html.Button("Upload"),
                                id="upload_image")])
                ),
        dbc.Col(html.Button("Predict!", id="predict_button", n_clicks=0), 
                style={"border": "#9E0600", "border-radius":"10px", "width":"10%"})]),
    
    dbc.Row([html.Div(id="display_image")]),
    dbc.Row([html.Div(id="prediction")], style={"border": "#9E0600", "border-radius":"10px"})      
    ]
)




@app.callback(Output("display_image", "children"),                
                Input("upload_image", "contents"),                
                State("upload_image", "filename"))

def display_upload(content, file_name):
    return html.Div([
        html.H5(file_name),
        html.Img(src=content, id="image")
    ])



@app.callback(Output("prediction", "children"),
                Input("predict_button", "n_clicks"),
                Input("upload_image", "contents"))

def display_prediction(n_clicks, image):
    if n_clicks != 0:
        predictions = predict(image)
        
        return  html.Div([html.P(predictions)])


if __name__ == '__main__':
    app.run_server(debug=False)
