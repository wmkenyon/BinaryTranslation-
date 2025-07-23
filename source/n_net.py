import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=8, output_size=3):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size).cuda()
        self.fc2 = nn.Linear(hidden_size, output_size).cuda()

    def forward(self, x):
        x=F.sigmoid(self.fc1(x))
        x=self.fc2(x)
        return F.relu(x, dim=1)

class LSTM_Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).cuda()
        self.fc = nn.Linear(hidden_size, num_classes).cuda()

    def forward(self, x):
        out, _ = self.lstm(x)              # out: (B, T, H)
        out = out[:, -1, :]                # take last time step
        return self.fc(out)


class Transformer_Model(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_classes=3):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model).cuda()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead).cuda()
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2).cuda()
        self.fc = nn.Linear(d_model, num_classes).cuda()

    def forward(self, x):
        # x: (B, T, D) → (T, B, D)
        x = self.input_fc(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)  # average over time steps
        return self.fc(x)

def main():
    
    model = SimpleNN()
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.NLLLoss().cuda()

    #Here is where I would put in my input and label
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).cuda()
    y = torch.tensor([1]).cuda()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()