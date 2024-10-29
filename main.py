# import torch
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN
from server import Server
from clients import Client
from attacker import CorruptedDataset


def main():

    # Setup parameters
    setup_outer_epochs = int(sys.argv[2])
    # setup_dataset = "fashion"
    setup_dataset = sys.argv[4]
    setup_client_count = int(sys.argv[6])
    setup_inner_client_epochs = 2
    setup_client_attacker = int(sys.argv[8])

    print("")
    print("-----------------")
    print("Setup Details: ")
    print(f"Outer Epochs: {setup_outer_epochs}")
    print(f"Inner Learning Rounds: {setup_inner_client_epochs}")
    print(f"Client Count: {setup_client_count}")
    print(f"Client Attacker Index:  {setup_client_attacker}")
    print("-----------------")
    print("")

    if setup_dataset == "fashion":
        # Set up training and test data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.FashionMNIST(
            root='./data/.', train=True, download=True, transform=transform)

        attacker_data = CorruptedDataset(train_data,
                                         patch_value=255, save_samples=True, patch_x=10, patch_y=10)
        test_data = datasets.FashionMNIST(
            root='./data/.', train=False, download=True, transform=transform)

    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # Initialize model, server, and clients
    global_model = SimpleCNN()
    server = Server(global_model, test_loader)

    clients = []
    for i in range(setup_client_count):  # Assume 5 clients
        local_model = SimpleCNN()
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        if i == setup_client_attacker:
            # some
            client_loader = DataLoader(
                attacker_data, batch_size=64, shuffle=True)
        else:
            client_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        client = Client(i, local_model, client_loader, optimizer,
                        inner_epochs=setup_inner_client_epochs)
        clients.append(client)
        server.attach(client)

    # Federated learning process
    for round in range(setup_outer_epochs):  # Assume 10 rounds of training
        print(f"Round {round + 1}")
        server.distribute()
        selected_clients = range(len(clients))  # All clients participate
        server.train(selected_clients)
        server.test()


if __name__ == "__main__":
    main()
