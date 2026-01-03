"""
This module defines deep learning model architectures (such as AlexNet, ResNet, and SENet variants)
and their training utilities for precipitation and storm surge prediction tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import config

# ================================================================================================================
# AlexNet Architecture
# ================================================================================================================

class AlexNetPrecipitation(nn.Module):
    """AlexNet-inspired CNN for precipitation regression."""
    
    def __init__(self, num_features, dropout_rate=0.5, activation='relu'):
        super(AlexNetPrecipitation, self).__init__()
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'silu':
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.features = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, stride=1, padding=1),  
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),  
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_fn,
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_fn,
         
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            act_fn,
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetStormSurge(nn.Module):
    """AlexNet-inspired CNN for storm surge regression."""
    
    def __init__(self, num_features, dropout_rate=0.5, activation='relu'):
        super(AlexNetStormSurge, self).__init__()
        
        # Choose activation function
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'silu':
            act_fn = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.features = nn.Sequential(
            nn.Conv2d(num_features, 64, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            act_fn,
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            act_fn,
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            act_fn,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            act_fn,
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ================================================================================================================
# ResNet Architecture Components
# ================================================================================================================

class Bottleneck(nn.Module):
    """Standard ResNet bottleneck block."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, identity_downsample=None, stride=1, activation='silu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        # Configurable activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.activation(out)
        return out

class SELayer(nn.Module):
    """Squeeze-and-Excitation layer."""
    
    def __init__(self, channel, reduction=16, activation='silu'):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Configurable activation function
        if activation == 'silu':
            act_func = nn.SiLU(inplace=True)
        elif activation == 'relu':
            act_func = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            act_func,
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBottleneck(nn.Module):
    """Bottleneck block with Squeeze-and-Excitation."""
    expansion = 4

    def __init__(self, in_channels, mid_channels, identity_downsample=None, stride=1, reduction=16, activation='silu'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        
        # Configurable activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.se = SELayer(mid_channels * self.expansion, reduction, activation)
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.activation(out)
        return out

class ResNetBase(nn.Module):
    """Generic ResNet/SENet backbone for regression."""
    
    def __init__(self, block, layers, image_channels, dropout_rate=0.5, activation='silu'):
        super().__init__()
        self.in_channels = 64
        self.activation_type = activation
        
        # Configurable activation function
        if activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], mid_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], mid_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], mid_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], mid_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, 1)

        self._initialise_weights()

    def _make_layer(self, block, num_blocks, mid_channels, stride):
        identity_downsample = None
        if stride != 1 or self.in_channels != mid_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, mid_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(mid_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, mid_channels, identity_downsample, stride, activation=self.activation_type))
        self.in_channels = mid_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, mid_channels, activation=self.activation_type))
        return nn.Sequential(*layers)

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ================================================================================================================
# Model Constructor Functions
# ================================================================================================================

def SDS_ResNet(img_channel, dropout_rate=0.5, activation='silu'):
    """Standard ResNet-50-style architecture for regression."""
    return ResNetBase(Bottleneck, [3, 4, 6, 3], img_channel, dropout_rate, activation)

def SDS_SENet(img_channel, dropout_rate=0.5, activation='silu'):
    """SENet-50-style architecture for regression."""
    return ResNetBase(SEBottleneck, [3, 4, 6, 3], img_channel, dropout_rate, activation)

# ================================================================================================================
# Trainer Class
# ================================================================================================================

class Trainer:
    """Training class for model training and validation."""
    
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, best_model_path):
        self.model = model
        self.optimizer = optimizer 
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_loss = float('inf')
        self.best_model_path = best_model_path

    def train_ep(self):
        """Train for one epoch."""
        self.model.train()
        losses = []
        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs.to(config.DEVICE))
            # Handle both single value and multi-value targets
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = self.criterion(outputs, targets.to(config.DEVICE))

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def eval(self):
        """Evaluate on validation set."""
        self.model.eval()
        losses, predictions, actuals = [], [], []

        for inputs, targets in self.val_loader:
            outputs = self.model(inputs.to(config.DEVICE))
            # Handle both single value and multi-value targets
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = self.criterion(outputs, targets.to(config.DEVICE))
            losses.append(loss.item())

            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
        
        return np.mean(losses), predictions, actuals
    
    def train(self, num_epochs):
        """Train the model for specified number of epochs."""
        train_losses, eval_losses = [], []
        for epoch in range(1, num_epochs+1):
            train_loss = self.train_ep()
            train_losses.append(train_loss)
            val_loss, val_predictions, val_actuals = self.eval()
            eval_losses.append(val_loss)
            print(f'Epoch {epoch}/{num_epochs} -> Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}')

            # Save model if best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                
        return train_losses, eval_losses, (val_predictions, val_actuals)

def create_model(model_type, num_features, dropout_rate=0.5, activation='silu', task_type='precipitation'):
    """
    Factory function to create a model.
    
    Args:
        model_type: 'alexnet', 'resnet', or 'senet'
        num_features: Number of input channels
        dropout_rate: Dropout rate
        activation: Activation function ('relu' or 'silu')
        task_type: 'precipitation' or 'stormsurge' (for AlexNet selection)
    
    Returns:
        Model instance
    """
    if model_type == 'alexnet':
        if task_type == 'precipitation':
            return AlexNetPrecipitation(num_features, dropout_rate, activation).to(config.DEVICE)
        else:  # stormsurge
            return AlexNetStormSurge(num_features, dropout_rate, activation).to(config.DEVICE)
    elif model_type == 'resnet':
        return SDS_ResNet(num_features, dropout_rate, activation).to(config.DEVICE)
    elif model_type == 'senet':
        return SDS_SENet(num_features, dropout_rate, activation).to(config.DEVICE)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_optimizer(model, optimizer_name, lr, weight_decay):
    """
    Create an optimizer.
    
    Args:
        model: Model to optimize
        optimizer_name: 'adam' or 'sgd'
        lr: Learning rate
        weight_decay: Weight decay
    
    Returns:
        Optimizer instance
    """
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

