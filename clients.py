from copy import deepcopy
import torch
import torch.nn.functional as F

class Client:
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss, device='cpu', inner_epochs=1):
        self.cid = cid
        self.model = model
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.device = device
        self.init_stateChange()
        self.originalState = deepcopy(model.state_dict())
        self.isTrained = False
        self.inner_epochs = inner_epochs
        self.criterion = criterion

    def init_stateChange(self):
        states = deepcopy(self.model.state_dict())
        for param, values in states.items():
            values *= 0
        self.stateChange = states

    def setModelParameter(self, states):
        self.model.load_state_dict(deepcopy(states))
        self.originalState = deepcopy(states)
        self.model.zero_grad()

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.isTrained = True
        self.model.cpu()

    def update(self):
        assert self.isTrained, 'nothing to update, call train() to obtain gradients'
        newState = self.model.state_dict()
        for param in self.originalState:
            self.stateChange[param] = newState[param] - self.originalState[param]
        self.isTrained = False

    def getDelta(self):
        return self.stateChange

