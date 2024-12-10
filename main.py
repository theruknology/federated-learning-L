import sys
import csv
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import SimpleCNN
from server import Server
from clients import Client
from attacker import CorruptedDataset
import matplotlib.pyplot as plt

def run_federated_learning(setup_client_count, attacker_percentage, setup_outer_epochs=15, setup_inner_client_epochs=2):
    # Calculate the number of attackers
    num_attackers = int(setup_client_count * attacker_percentage / 100)

    print("\n-----------------")
    print(f"Running with {setup_client_count} clients and {attacker_percentage}% attackers.")
    print("-----------------\n")

    # Set up training and test data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_data = datasets.MNIST(
        root="./data/.", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data/.", train=False, download=True, transform=transform
    )

    attacker_data = CorruptedDataset(
        train_data, patch_value=255, save_samples=True, patch_x=10, patch_y=10
    )

    test_data = CorruptedDataset(
        test_data,
        patch_value=255,
        save_samples=True,
        patch_x=10,
        patch_y=10,
        test=True,
        corruption_percent=20,
    )

    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    # Split training data among clients
    client_data_splits = random_split(
        train_data, [len(train_data) // setup_client_count] * setup_client_count
    )

    # Initialize model, server, and clients
    global_model = SimpleCNN()
    server = Server(global_model, test_loader)

    clients = []
    for i in range(setup_client_count):
        local_model = SimpleCNN()
        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        if i < num_attackers:  # First `num_attackers` clients are attackers
            client_loader = DataLoader(
                attacker_data, batch_size=64, shuffle=True
            )
        else:
            client_loader = DataLoader(
                client_data_splits[i], batch_size=64, shuffle=True
            )

        client = Client(i, local_model, client_loader, optimizer,
                        inner_epochs=setup_inner_client_epochs)
        clients.append(client)
        server.attach(client)

    # Federated learning process
    accuracies = []  # To store accuracy per round
    for round in range(setup_outer_epochs):
        print(f"Round {round + 1}")
        server.distribute()
        selected_clients = range(len(clients))  # All clients participate
        server.train(selected_clients)
        accuracy = server.test()
        accuracies.append(accuracy)

    return accuracies  # Return accuracy for all rounds


def main():
    setup_client_count = 10  # Assume this is always a factor of 10 for simplicity
    setup_outer_epochs = 3  # Total training rounds
    setup_inner_client_epochs = 2  # Local training rounds
    attacker_percentages = [40, 50, 60, 70, 80]  # List of attacker percentages

    results_path = "roundwise_accuracy.csv"  # CSV file to save the results

    # Open the CSV file for logging
    with open(results_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Attacker Percentage", "Round", "Accuracy"])  # Header

        for attacker_percentage in attacker_percentages:
            accuracies = run_federated_learning(
                setup_client_count,
                attacker_percentage,
                setup_outer_epochs,
                setup_inner_client_epochs,
            )

            # Write results for this attacker percentage
            for round_number, accuracy in enumerate(accuracies, start=1):
                writer.writerow([attacker_percentage, round_number, accuracy])

    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

