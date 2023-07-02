import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

LR_MODEL_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/data/lr_model.pt"
IMG_COORDS_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/data/img/detect_coords.csv"
ACTUAL_COORDS_PATH = "/home/rayageorgieva/uni/masters/pick_place_robosuite/data/img/combined_coords.csv"

# implementation consulted with https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
class LinearRegression(torch.nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(LinearRegression, self).__init__()
        self.linear_layer = nn.Linear(input_size, output_size).to(device)
    
    def forward(self, x):
        return self.linear_layer(x)
    
    def load(self):
        self.load_state_dict(torch.load(LR_MODEL_PATH, map_location=device))

def main():

    x_df = pd.read_csv(IMG_COORDS_PATH)
    x_data = x_df[['x1', 'x2', 'y1', 'y2']].to_numpy()
    x_len = x_data.shape[0]
    x_train = x_data[:int(x_len*0.8), :]
    x_val = x_data[int(x_len*0.8):int(x_len*0.9), :]
    x_test = x_data[int(x_len*0.9):, :]
    print(f"X: train: {x_train.shape} val: {x_val.shape} test: {x_test.shape}")


    y_df = pd.read_csv(ACTUAL_COORDS_PATH) # actual coords
    y_data = y_df[['x', 'y', 'z']].to_numpy()
    y_len = y_data.shape[0]
    y_train = y_data[:int(y_len*0.8), :]
    y_val = y_data[int(y_len*0.8):int(y_len*0.9), :]
    y_test = y_data[int(y_len*0.9):, :]
    print(f"Y: train: {y_train.shape} val: {y_val.shape} test: {y_test.shape}")

    inputDim = 4
    outputDim = 3
    learningRate = 0.00001 
    epochs = 500000
    losses = []
    val_losses = []

    model = LinearRegression(inputDim, outputDim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        inputs = torch.FloatTensor(x_train).to(device)
        labels = torch.FloatTensor(y_train).to(device)
            
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        # validation set
        val_out = model(torch.FloatTensor(x_val).to(device))
        val_loss = nn.MSELoss()(val_out, torch.FloatTensor(y_val).to(device))
        val_losses.append(val_loss.item())

        print('Epoch {}, train loss {} validation loss {}'.format(epoch, loss.item(), val_loss.item()))

    test_out = model(torch.FloatTensor(x_test).to(device))
    test_loss = nn.MSELoss()(test_out, torch.FloatTensor(y_test).to(device))
    print(f"Test Loss {test_loss.item()}")
    torch.save(model.state_dict(), LR_MODEL_PATH)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].plot(losses)
    ax[1].plot(val_losses)

    plt.show()


if __name__ == '__main__':
    main()

