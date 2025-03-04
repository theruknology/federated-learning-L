# federated_defenses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Model Architecture
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def get_model_params(model):
    return {name: param.clone() for name, param in model.state_dict().items()}

def set_model_params(model, params):
    model.load_state_dict(params)

# Client Implementation
class Client:
    def __init__(self, client_id, train_data, test_data, is_malicious=False, target_label=None, attack_prob=0.8):
        self.client_id = client_id
        self.train_data = train_data
        self.test_data = test_data
        self.is_malicious = is_malicious
        self.target_label = target_label
        self.attack_prob = attack_prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add_backdoor_trigger(self, image):
        img = deepcopy(image)
        h, w = img.shape[1:]
        x = np.random.randint(2, w-3)
        y = np.random.randint(2, h-3)
        trigger_w = np.random.randint(2, 4)
        trigger_h = np.random.randint(2, 4)
        img[0, y:y+trigger_h, x-trigger_w:x+trigger_w] = 1.0
        img[0, y-trigger_w:y+trigger_w, x:x+trigger_h] = 1.0
        return img

    def poison_dataset(self):
        if not self.is_malicious:
            return self.train_data
            
        poisoned_images = []
        poisoned_labels = []
        for img, label in self.train_data:
            if np.random.random() < 0.5:
                img = self.add_backdoor_trigger(img)
                label = self.target_label
            poisoned_images.append(img)
            poisoned_labels.append(label)
        return TensorDataset(torch.stack(poisoned_images), torch.tensor(poisoned_labels))

    def train(self, model, epochs=1, batch_size=32):
        model.train().to(self.device)
        attack_this_round = self.is_malicious and np.random.rand() < self.attack_prob
        train_data = self.poison_dataset() if attack_this_round else self.train_data
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
        for _ in range(epochs):
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                
        return {name: param.clone().detach() for name, param in model.state_dict().items()}

# Server with Defense Mechanisms
class Server:
    def __init__(self, model, clients, defense_type='none', l=3, eps=0.5, min_samples=2, k=2, unreliability_threshold=0.3):
        self.model = model
        self.clients = clients
        self.defense_type = defense_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.malicious_ids = {c.client_id for c in clients if c.is_malicious}
        
        # MUD-HoG parameters
        self.l = l  # Short-term window size
        self.eps = eps  # DBSCAN eps
        self.min_samples = min_samples  # DBSCAN min_samples
        self.k = k  # K-means clusters
        self.unreliability_threshold = unreliability_threshold
        
        # Gradient history storage
        self.short_hog = {c.client_id: [] for c in clients}  # Last l gradients
        self.long_hog = {c.client_id: None for c in clients}  # Cumulative gradients

    def calculate_metrics(self, detected):
        tp = len(detected & self.malicious_ids)
        fp = len(detected - self.malicious_ids)
        fn = len(self.malicious_ids - detected)
        tn = len({c.client_id for c in self.clients}) - tp - fp - fn
        return {
            'dr': tp/(tp+fn) if (tp+fn) > 0 else 0,
            'fpr': fp/(fp+tn) if (fp+tn) > 0 else 0,
            'precision': tp/(tp+fp) if (tp+fp) > 0 else 0
        }

    def fools_gold_defense(self, updates):
        vectors = [np.concatenate([p.cpu().numpy().flatten() for p in u.values()]) for u in updates]
        sim_matrix = cosine_similarity(vectors)
        weights = np.ones(len(updates))
        detected = set()
        
        for i in range(len(updates)):
            for j in range(len(updates)):
                if i != j and sim_matrix[i][j] > 0.9:  # Fixed threshold
                    weights[i] *= 0.5
                    detected.add(self.clients[i].client_id)
                    
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(len(updates))/len(updates)
            
        return weights, self.calculate_metrics(detected)

    def mudhog_defense(self, updates):
        client_id_to_idx = {c.client_id: i for i, c in enumerate(self.clients)}
        
        # Step 1: Detect sign-flipping attackers
        short_hogs = []
        for c in self.clients:
            if len(self.short_hog[c.client_id]) >= 1:
                hog = np.mean(self.short_hog[c.client_id], axis=0)
            else:
                hog = np.concatenate([p.cpu().numpy().flatten() for p in updates[client_id_to_idx[c.client_id]].values()])
            short_hogs.append(hog)
        
        med_short = np.median(short_hogs, axis=0)
        sign_flippers = set()
        for i, hog in enumerate(short_hogs):
            cos_sim = cosine_similarity([hog], [med_short])[0][0]
            if cos_sim < 0:
                sign_flippers.add(self.clients[i].client_id)
        
        # Step 2: Detect additive-noise attackers with DBSCAN
        remaining = [i for i, c in enumerate(self.clients) if c.client_id not in sign_flippers]
        remaining_hogs = [short_hogs[i] for i in remaining]
        additive_noise = set()
        if len(remaining_hogs) > 0:
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(remaining_hogs)
            labels = db.labels_
            unique, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique) > 0:
                main_cluster = unique[np.argmax(counts)]
                for idx, lbl in enumerate(labels):
                    if lbl != main_cluster and lbl != -1:
                        additive_noise.add(self.clients[remaining[idx]].client_id)
        
        # Step 3: Detect targeted attackers with long-term HoG
        targeted = set()
        remaining_clients = [c for c in self.clients if c.client_id not in sign_flippers | additive_noise]
        long_hogs = []
        for c in remaining_clients:
            if self.long_hog[c.client_id] is not None:
                long_hogs.append(self.long_hog[c.client_id])
            else:
                long_hogs.append(np.concatenate([p.cpu().numpy().flatten() for p in updates[client_id_to_idx[c.client_id]].values()]))
        
        if len(long_hogs) >= self.k:
            kmeans = KMeans(n_clusters=self.k).fit(long_hogs)
            cluster_counts = np.bincount(kmeans.labels_)
            if len(cluster_counts) >= 2:
                malicious_cluster = np.argmin(cluster_counts)
                for i, lbl in enumerate(kmeans.labels_):
                    if lbl == malicious_cluster:
                        targeted.add(remaining_clients[i].client_id)
        
        # Step 4: Detect unreliable clients
        detected = sign_flippers | additive_noise | targeted
        remaining = [c for c in self.clients if c.client_id not in detected]
        unreliable = set()
        if len(remaining) > 0:
            remaining_hogs = [short_hogs[client_id_to_idx[c.client_id]] for c in remaining]
            med_remaining = np.median(remaining_hogs, axis=0)
            for c in remaining:
                hog = short_hogs[client_id_to_idx[c.client_id]]
                cos_dist = 1 - cosine_similarity([hog], [med_remaining])[0][0]
                if cos_dist < self.unreliability_threshold:
                    unreliable.add(c.client_id)
        
        # Calculate weights
        weights = []
        for c in self.clients:
            if c.client_id in detected:
                weights.append(0.0)
            elif c.client_id in unreliable:
                weights.append(0.5)
            else:
                weights.append(1.0)
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights /= weights.sum()
        
        return weights, self.calculate_metrics(detected)

    def contra_defense(self, updates):
        vectors = [np.concatenate([p.cpu().numpy().flatten() for p in u.values()]) for u in updates]
        sim_matrix = cosine_similarity(vectors)
        avg_similarities = np.mean(sim_matrix, axis=1)
        contrast_scores = 1 / (1 + np.exp(-10*(avg_similarities - np.median(avg_similarities))))
        weights = 1 - contrast_scores
        detected = {self.clients[i].client_id for i in np.where(contrast_scores > 0.5)[0]}
        
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(len(updates))/len(updates)
            
        return weights, self.calculate_metrics(detected)

    def aggregate(self, client_updates):
        # Update gradient histories
        for client, update in zip(self.clients, client_updates):
            grad_vec = np.concatenate([p.cpu().numpy().flatten() for p in update.values()])
            # Update short-term history
            self.short_hog[client.client_id].append(grad_vec)
            if len(self.short_hog[client.client_id]) > self.l:
                self.short_hog[client.client_id].pop(0)
            # Update long-term history
            if self.long_hog[client.client_id] is None:
                self.long_hog[client.client_id] = grad_vec.copy()
            else:
                self.long_hog[client.client_id] += grad_vec
        
        # Apply defense
        if self.defense_type == 'fools_gold':
            weights, metrics = self.fools_gold_defense(client_updates)
        elif self.defense_type == 'mudhog':
            weights, metrics = self.mudhog_defense(client_updates)
        elif self.defense_type == 'contra':
            weights, metrics = self.contra_defense(client_updates)
        else:
            weights = np.ones(len(client_updates))/len(client_updates)
            metrics = None

        # Aggregate parameters
        aggregated_params = {}
        for name in client_updates[0]:
            aggregated_params[name] = sum(update[name] * weight for update, weight in zip(client_updates, weights))
            
        return aggregated_params, metrics

    def evaluate_backdoor(self, target_label):
        self.model.eval()
        success = 0
        total = 0
        
        for client in self.clients:
            if client.is_malicious:
                for data, _ in client.test_data:
                    poisoned_data = client.add_backdoor_trigger(data.clone())
                    poisoned_data = poisoned_data.unsqueeze(0).to(self.device)
                    output = self.model(poisoned_data)
                    pred = output.argmax(dim=1)
                    success += (pred == target_label).sum().item()
                    total += 1
                    
        return 100.0 * success / total if total > 0 else 0.0

    def train_round(self, local_epochs=5):
        global_params = get_model_params(self.model)
        client_updates = []
        
        for client in self.clients:
            set_model_params(self.model, global_params)
            client_updates.append(client.train(self.model, epochs=local_epochs))
            
        aggregated_params, metrics = self.aggregate(client_updates)
        set_model_params(self.model, aggregated_params)
        backdoor_sr = self.evaluate_backdoor(target_label=7)
        return backdoor_sr, metrics

# Experiment Framework
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    handler = logging.FileHandler(f'logs/experiment_{timestamp}.log')
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def load_and_split_data(num_clients):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10('./data', train=False, transform=transform)
    
    client_train = random_split(train_set, [len(train_set)//num_clients]*num_clients)
    client_test = random_split(test_set, [len(test_set)//num_clients]*num_clients)
    
    return client_train, client_test

def run_experiment(defense_type, malicious_pct, logger):
    num_clients = 20
    client_train, client_test = load_and_split_data(num_clients)
    num_malicious = int(num_clients * malicious_pct)
    
    clients = [
        Client(i, client_train[i], client_test[i],
              is_malicious=(i < num_malicious), target_label=7)
        for i in range(num_clients)
    ]
    
    model = CIFAR10CNN()
    server = Server(model, clients, defense_type=defense_type)
    backdoor_sr, metrics = server.train_round()
    
    logger.info(f"{defense_type.upper()} | Malicious: {num_malicious} | "
               f"Success Rate: {backdoor_sr:.2f}% | "
               f"Detection Rate: {metrics['dr']*100 if metrics else 0:.2f}%")
    
    return {
        'defense': defense_type,
        'malicious': num_malicious,
        'success_rate': backdoor_sr,
        'detection_rate': metrics['dr'] if metrics else 0
    }

def plot_results(results):
    plt.figure(figsize=(12, 6))
    defenses = ['none', 'fools_gold', 'mudhog', 'contra']
    colors = ['red', 'blue', 'green', 'purple']
    markers = ['x', 'o', '^', 's']
    
    for defense, color, marker in zip(defenses, colors, markers):
        defense_data = [r for r in results if r['defense'] == defense]
        x = [d['malicious'] for d in defense_data]
        y = [d['success_rate'] for d in defense_data]
        plt.plot(x, y, f'{color}{marker}--', linewidth=2, markersize=10, label=defense)
    
    plt.xlabel('Number of Malicious Clients', fontsize=12)
    plt.ylabel('Backdoor Attack Success Rate (%)', fontsize=12)
    plt.title('Defense Mechanism Comparison', fontsize=14)
    plt.xticks([2, 3, 4, 5])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('defense_comparison.png')
    plt.show()

def main():
    logger = setup_logging()
    results = []
    defenses = ['mudhog', 'none', 'fools_gold', 'contra']
    malicious_pcts = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,0.35,  0.4,0.45,  0.5]
    
    for defense in defenses:
        logger.info(f"\n=== Testing {defense.upper()} Defense ===")
        for pct in malicious_pcts:
            try:
                result = run_experiment(defense, pct, logger)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in {defense} {pct}: {str(e)}")
    
    plot_results(results)

if __name__ == "__main__":
    main()