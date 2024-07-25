from copy import deepcopy
import torch
import torch.nn.functional as F

class Server():
  def __init__(self, model, dataloader, criterion=F.nll_loss, device='gpu'):
    self.clients = []
    self.model = model
    self.dataloader = dataloader
    self.device = device
    self.emptyStates = None
    self.init_stateChange()
    self.criterion = criterion

    
  def init_statechange(self):
    states = deepcopy(self.model.state_dict())
    for param, values in states.items():
      values*=0

    self.emptyStates = states

  def attach(self, client):
    self.clients.append(client)

  def distribute(self):
    for client in self.clients:
      client.setModelParater(self.model.state_dict())
    
  def test(self):
    print("[Server] Start Testing")
    self.model.to(self.device)
    self.model.eval()
    test_loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
      for data, target in self.dataloader:
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        test_loss += self.criterion(output, target, reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        count += pred.shape[0]
    test_loss /= count
    accuracy = 100. * correct / count
    self.model.cpu()
    print('f[Server] Test set: Average Loss: {test_loss:.4f}, Accuracy: {correct}/{count} ({accuracy:.0f}%)\n')
    return test_loss, accuracy

  def train(self):
    for client in self.clients:
      client.train()
      client.update()

    Delta = self.FedAvg(self.clients)
    for param in self.model.state_dict():
      self.model.state_dict()[param] += Delta[param]
    
  def FedAvg(self, clients):
    Delta = deepcopy(self.emptyStates)
    deltas = [client.getDelta() for client in clients]
    for param in Delta:
      Delta[param] = torch.mean(torch.stack([delta[param] for delta in deltas]), dim=0)
    return Delta
      