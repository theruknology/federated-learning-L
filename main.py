import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from server import Server
from clients import Client

def main():
    # Set up training and test data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST('.', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # Initialize model, server, and clients
    global_model = SimpleCNN()
    server = Server(global_model, test_loader)

    clients = []
    for i in range(5):  # Assume 5 clients
        local_model = SimpleCNN()
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        client_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        client = Client(i, local_model, client_loader, optimizer)
        clients.append(client)
        server.attach(client)

    # Federated learning process
    for round in range(10):  # Assume 10 rounds of training
        print(f"Round {round + 1}")
        server.distribute()
        selected_clients = range(len(clients))  # All clients participate
        server.train(selected_clients)
        server.test()

if __name__ == "__main__":
    main()

