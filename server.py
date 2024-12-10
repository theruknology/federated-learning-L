# from copy import deepcopy
# import torch
# import torch.nn.functional as F
# 
# 
# class Server:
#     def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cuda'):
#         self.clients = []
#         self.model = model
#         self.dataLoader = dataLoader
#         self.device = device
#         self.emptyStates = None
#         self.init_stateChange()
#         self.Delta = None
#         self.iter = 0
#         self.AR = self.FedAvg
#         self.criterion = criterion
# 
#     def init_stateChange(self):
#         states = deepcopy(self.model.state_dict())
#         for param, values in states.items():
#             values *= 0
#         self.emptyStates = states
# 
#     def attach(self, c):
#         self.clients.append(c)
# 
#     def distribute(self):
#         for c in self.clients:
#             c.setModelParameter(self.model.state_dict())
# 
#     def test(self):
#         print("[Server] Start testing")
#         self.model.to(self.device)
#         self.model.eval()
#         test_loss = 0
#         correct = 0
#         count = 0
#         with torch.no_grad():
#             for data, target in self.dataLoader:
#                 data, target = data.to(self.device), target.to(self.device)
#                 output = self.model(data)
#                 # sum up batch loss
#                 test_loss += self.criterion(output,
#                                             target, reduction='sum').item()
#                 # get the index of the max log-probability
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.view_as(pred)).sum().item()
#                 count += pred.shape[0]
#         test_loss /= count
#         accuracy = 100. * correct / count
#         self.model.cpu()
#         print(f'[Server] Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{count} ({accuracy:.2f}%)')
#         return test_loss, accuracy
# 
#     def train(self, group):
#         selectedClients = [self.clients[i] for i in group]
#         cid = 0
#         for c in selectedClients:
#             c.train()
#             c.update()
#             # c.test_accuracy(self.dataLoader)
#             print(f'C{cid} trained')
#             cid += 1
#         Delta = self.AR(selectedClients)
#         for param in self.model.state_dict():
#             self.model.state_dict()[param] += Delta[param]
#         self.iter += 1
# 
#     def FedAvg(self, clients):
#         Delta = deepcopy(self.emptyStates)
#         deltas = [c.getDelta() for c in clients]
#         for param in Delta:
#             Delta[param] = torch.mean(torch.stack(
#                 [delta[param] for delta in deltas]), dim=0)
#         return Delta
from copy import deepcopy
import torch
import torch.nn.functional as F


class Server:
    def __init__(self, model, dataLoader, criterion=F.nll_loss, device='cuda'):
        self.clients = []
        self.model = model
        self.dataLoader = dataLoader
        self.device = device
        self.emptyStates = None
        self.init_stateChange()
        self.Delta = None
        self.iter = 0
        self.AR = self.FedAvg
        self.criterion = criterion

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.emptyStates = states

    def attach(self, c):
        self.clients.append(c)

    def distribute(self):
        for c in self.clients:
            c.setModelParameter(self.model.state_dict())

    def test(self):
        print("[Server] Start testing")
        self.model.to(self.device)
        self.model.eval()
        test_loss = 0
        correct = 0
        count = 0
        with torch.no_grad():
            for data, target in self.dataLoader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                count += pred.shape[0]
        test_loss /= count
        accuracy = 100. * correct / count
        self.model.cpu()
        print(f'[Server] Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{count} ({accuracy:.0f}%)')
        return test_loss, accuracy

    def train(self, group):
        selectedClients = [self.clients[i] for i in group]
        for c in selectedClients:
            c.train()
            c.update()
        
        # Perform attacker detection
        valid_clients = self.detect_attackers(selectedClients)
        
        # Aggregate only from valid clients
        Delta = self.AR(selectedClients)
        for param in self.model.state_dict():
            self.model.state_dict()[param] += Delta[param]
        self.iter += 1

    def FedAvg(self, clients):
        Delta = deepcopy(self.emptyStates)
        deltas = [c.getDelta() for c in clients]
        for param in Delta:
            Delta[param] = torch.mean(torch.stack(
                [delta[param] for delta in deltas]), dim=0)
        return Delta

    def detect_attackers(self, clients):
        """
        Detect attackers based on gradient similarity.
        """
        deltas = [c.getDelta() for c in clients]
        distances = []

        # Compute pairwise distances (e.g., L2 norm) between client updates
        for i in range(len(deltas)):
            total_distance = 0
            for j in range(len(deltas)):
                if i != j:
                    diff = {param: deltas[i][param] - deltas[j][param] for param in deltas[i]}
                    total_distance += sum(torch.norm(diff[param]) for param in diff)
            distances.append(total_distance)

        # Detect outliers based on distance (e.g., above mean + threshold)
        threshold = torch.mean(torch.tensor(distances)) + 2 * torch.std(torch.tensor(distances))
        valid_clients = [clients[i] for i, dist in enumerate(distances) if dist <= threshold]

        # Print detection results
        print(f"[Server] Detected {len(clients) - len(valid_clients)} attackers.")
        return valid_clients

