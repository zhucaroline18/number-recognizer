import json
import torch
import torch.nn as nn #torch's neural networking library
import torchvision #manipulate and process images
import os.path

import matplotlib.pyplot as plt 
from flask import Flask, jsonify, render_template, request


app = Flask(__name__,
            static_url_path = '',
            static_folder = 'web'
            #static_folder = '../client/build'
            )
app.config.from_object(__name__)

def root_dir(): 
    return os.path.abspath(os.path.dirname(__file__))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    if request.method == 'POST': 
        #file_data = request.files['file']
        #file_data = request.body.data
        request_data_bytes = request.data
        file_data = request_data_bytes.decode('utf-8')
        request_data_dict = json.loads(file_data)
        base64_data = request_data_dict['data']
        model = torch.load("my_mnst.pth")
        image = pre_process_img_data(base64_data)
        output = model(image)

        predict = output.argmax(dim = 1)
        intForm = predict.item()

        print(f'prediction of successful upload: {intForm}')
        #return (intForm)
        print(intForm)
        return jsonify(intForm)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.L1 = nn.Linear(784, 128)
        self.reLu = nn.ReLU() #our activation function 
        self.L2 = nn.Linear(128, 10) #layer 2
    
    def forward(self, x): #forward propogation with input x
        z1 = self.L1(x)
        a1 = self.reLu(z1)
        z2 = self.L2(a1)

        return z2
    
train_dataset = torchvision.datasets.MNIST (
    root = 'data',
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor() #tracks derivatives
)
test_dataset = torchvision.datasets.MNIST(
    root = 'data', 
    train = False, 
    download = True, 
    transform = torchvision.transforms.ToTensor()
)


def pre_process_img(file_data): #processing the submitted image 
    import os
    import numpy as np
    from PIL import Image #(python imagine library)
    import torch 

    src_image = Image.open(file_data) #open up the data
    img28 = src_image.resize((28,28)) #resive to 28x28 pixels
    img_data = np.array(img28.convert('L')) #to grayscale
    img_data = img_data/255
    img_data = 1-img_data #flipping the colors
    img_tensor = torch.Tensor(img_data)
    t2 = img_tensor.reshape((1, 28*28))
    return t2 

def pre_process_img_data(base64_data): #processing the submitted image 
    import os
    import numpy as np
    from PIL import Image #(python imagine library)
    import torch
    import base64
    import io
    src_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_data, "utf-8"))))
    img28 = src_image.resize((28,28)) #resive to 28x28 pixels
    img_data = np.array(img28.convert('L')) #to grayscale
    img_data = img_data/255
    img_data = 1-img_data #flipping the colors
    img_tensor = torch.Tensor(img_data)
    t2 = img_tensor.reshape((1, 28*28))
    return t2 


def train_model (): 
    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_fn = nn.CrossEntropyLoss() 

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)

    for epoch in range(10): #how many times training the whole thing
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(images.shape[0], 28*28)
            y = model(images) #what it guesses //////////////////////////////////why don't you have to say forward?
            loss = loss_fn(y, labels)

            optimizer.zero_grad() #reset derivatives
            loss.backward() #backpropogation based on loss
            optimizer.step() #adjust weights and biases

            if (i%100 == 0):
                print(f"epoch {epoch}: loss = {loss}")
        
    torch.save(model, "my_mnst.pth")

    state = {
        'state_dict': model.state_dict(), # what's this???????????????????
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, "my_mnist-checkpoint.pth")

def load_checkpoint_train_model():
    checkpoint = torch.load("my_mnist-checkpoint.pth")

    model = MyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)#setting the learning rate and finding quickest way by making all the parameters smaller over time 
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_fn = nn.CrossEntropyLoss() #set the function for calculating loss (this loss functions gets rid of the noise)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)#train 100 at a time the training dataset

    for epoch in range(10): #how many times are we training the whole thing?
        for i, (images, labels) in enumerate(train_loader): 
            #forward
            images = images.view(images.shape[0], 28*28) 
            y = model(images) #what it guesses 
            loss = loss_fn(y, labels) #fancy calculus stuff 

            #backwards
            optimizer.zero_grad() #reset derivatives to 0 or will be based on previous 
            loss.backward() #back propogation 
            optimizer.step() #use optimizer and loss function to adjust weights

            if i%100 == 0:
                print(f"epoch {epoch}: loss = {loss}")
        
    torch.save(model, "my_mnst.pth")

    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, "my_mnist-checkpoint.pth")

def test_model_all():
    model = torch.load("my_mnst.pth")

    correct = 0
    for i in range(len(test_dataset)):
        image, label = test_dataset[i]
        image = image.view(image.shape[0], 28*28)
        output = model(image)
        predict = output.argmax(dim = 1)
        if label == predict:
            correct +=1
        else:
            print (f"instance of {i}")
            print(f"label was: {label}, predicted was: {predict}")
    print(f"{correct} correct out of {len(test_dataset)}")

def check_dataset():
    index = 9948
    data, label = test_dataset[index]
    print(label)
    print(data)
    print(data.shape)
    image_data = data.squeeze(dim = 0)
    print(data.shape)

    plt.imshow(image_data, cmap = "gray")
    plt.show()

def test_with_user_input():
    model = torch.load("my_mnst.pth")
    image = get_my_img()
    output = model(image)
    predict = output.argmax(dim = 1)
    intForm = predict.item()
    print(f'predict: {intForm}') #print our guess 
    return intForm

def test_with_user_input2():
    model = torch.load("my_mnst.pth")
    image = pre_process_img_data('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')
    output = model(image)
    predict = output.argmax(dim = 1)
    intForm = predict.item()
    print(f'predict: {intForm}') #print our guess 
    return intForm

def get_my_img():
    import os
    import numpy as np
    from PIL import Image
    import torch

    #src_image_full_path = 'C:\\Users\\carol\\Downloads\\capture1.png'
    src_image_full_path = 'C:\\SpecialTopics\\MNIST\\test5.PNG'
    print(f"src={src_image_full_path}")

    # open image
    src_image = Image.open(src_image_full_path)
    # src_image.show()
    img28 = src_image.resize((28,28))
    # img28.show()
    img_data = np.array(img28.convert("L"))
    img_data = img_data/255
    img_data = 1-img_data

    #print(img_data.shape)
    #print(img_data)

    img_tensor = torch.Tensor(img_data)
    #print(img_tensor.shape)

    t2 = img_tensor.reshape((1,28 * 28))
    #print(t2.shape)
    #print(t2)
    return t2

if __name__ == "__main__":
    #load_checkpoint_train_model()
    #test_with_user_input2()
    #check_dataset()
    #test_model_all()
    app.debug = True
    app.run()