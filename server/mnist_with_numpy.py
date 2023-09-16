from csv import reader
import numpy as np
from matplotlib import pyplot as plt

def load_csv(filename, max_rows): 
    with open(filename) as file:
        csv_reader = reader(file)
        ret = list()
        count = 0
        # only load the first max_rows rows
        for row in csv_reader:
            if row is None:
                continue
            ret.append(row)
            count += 1
            if (max_rows>0 and count >= max_rows):
                break 

        return ret
    
# max rows is how many rows we want to be adding to our dataset
# we get all the data we need from the file and load_csv loads it for us into an array 
def load_dataset(filename, max_rows):

    if (max_rows>0):
        max_rows +=1
    csv_data = load_csv(filename, max_rows)

    #first row is column headers so don't need that 
    csv_data = csv_data[1:]

    dataset = list()
    labels = list()
    for raw_row in csv_data:
        # taking the first element and that's the label and adding to list of labels
        label = int(raw_row[0])
        labels.append(label)

        #taking the rest of the elements as rows and adding it to our dataset
        row = [int(col)/255.0 for col in raw_row[1:]]
        dataset.append(row)
    return dataset, labels

# show image and label at a specific index
def show_image(dataset, labels, index):
    label = labels[index]

    #turning linear row into a matrix
    image = np.array(dataset[index]).reshape((28,28))
    
    image = image * 255 #grayscale on 255
    print (f'label = {label}') #print out the label to check answer

    #plot out the image so we can see it 
    plt.gray()
    plt.imshow(image, interpolation = 'nearest')
    plt.show()

def ReLU(Z):
    #reLU function makes it so not everything is just a linear combination
    #linear if positive, otherwise 0
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    toReturn = list()
    for row in Z:
        new_row = list()
        for col in row:
            new_col = 1 if col>0 else 0
            new_row.append(new_col)
        toReturn.append(new_row)
    return toReturn

def softmax(Z):
    #for each element of array, e^x each element, then sum each of THAT array 
    exp = np.exp(Z)
    sum = np.sum(exp)
    return exp/sum

class NeuralNetwork:
    #initialize how many nodes in input, hiddenlayer, and output layer
    def __init__(self, inputCount, hiddenCount, outputCount):
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.hiddenCount = hiddenCount

        #now initialize some random values of certain sized matrices
        self.W1 = np.random.rand(hiddenCount, inputCount)-0.5
        self.b1 = np.random.rand(hiddenCount, 1)-0.5 #random bias to be applied
        #random ones for the second layer
        self.W2 = np.random.rand(outputCount, hiddenCount) - 0.5
        self.b2 = np.random.rand(outputCount, 1) - 0.5
        self.learning_rate = 0.1

    #forward propogation 
    def feed_forward(self, X): 
        Z1 = self.W1.dot(X)+self.b1 # dot product the weight and then add biases 
        A1 = ReLU(Z1) #take results and add activation function 
        Z2 = self.W2.dot(A1)+self.b2 #take the previous and do the same
        A2 = softmax(Z2) #get the activation function and predict probability of each digit 
        return Z1, A1, Z2, A2
    
    def one_hot(self, Y):
        ret = [0 for x in range(self.outputCount)] #array with all 0s
        ret[Y]=1 #what its suposed to be

        return np.array(ret).reshape(-1,1) 
    
    def backward_prop(self, X, Z1, A1, Z2, A2, Y):
        one_hot_y = self.one_hot(Y)

        dZ2 = A2-one_hot_y #difference 
        dW2 = dZ2.dot(A1.T) #calculating gradient decent
        db2 = dZ2
        dZ1 = self.W2.T.dot(dZ2)* ReLU_deriv(Z1)
        dW1 = dZ1.dot(X.t)
        db1 = dZ1

        #then update weights and bias 
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
    def train(self, X, Y):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        self.backward_prop(X, Z1, A1, Z2, A2, Y)

    def predict(self, X):
        Z1, A1, Z2, A2 = self.feed_forward(X)
        index = np.argmax(A2, 0) #maximum value among column 0
        return index
    
def train_and_predict():
    train_count = 5000
    test_count = 1000
    dataset, labels = load_dataset('mnist_train.csv', train_count + test_count)

    brain = NeuralNetwork(28*28, 10, 10)
    for i in range(train_count):
        #getting data and the label
        X = np.array(dataset[i]).reshape(-1,1)
        Y = labels[i]
        brain.train(X, Y)

    correct_count = 0
    for i in range(train_count, train_count + test_count):
        prediction = brain.predict(X)
        if prediction == Y:
            correct_count += 1
        else:
            print(f'incorrect prediction for data at index={i}. Prediction={prediction}, Answer={Y}')
    
    print(f'{correct_count} of {test_count} are correct')

if __name__ == "__main__":
    train_and_predict()