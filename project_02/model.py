import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, num_channels=21, time_steps=128, extract_features=False):
        super(BaselineCNN, self).__init__()
        self.extract_features = extract_features
        
        self.data_fusion = nn.Conv1d(num_channels, 16, kernel_size=1)
        self.conv1 = nn.Sequential(nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(4, 4))
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, time_steps)
            self.feature_size = self.conv3(self.conv2(self.conv1(self.data_fusion(dummy)))).view(1, -1).size(1)
            
        self.fc1 = nn.Sequential(nn.Linear(self.feature_size, 128), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        features = self.fc1(self.flatten(self.conv3(self.conv2(self.conv1(self.data_fusion(x))))))
        if self.extract_features:
            return features
        return self.fc2(features)

class TemporalCNN_LSTM(nn.Module):
    """
    LSTM architecture utilizing BaselineCNN as the backbone.
    """
    def __init__(self, num_channels=21, time_steps=128):
        super(TemporalCNN_LSTM, self).__init__()
        self.backbone = BaselineCNN(num_channels, time_steps, extract_features=True) 
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2)
        
    def forward(self, x_seq):
        # x_seq expected shape: [Batch, Seq_Len, 1, 21, 128]
        batch_size, seq_len, c, h, w = x_seq.size()
        x_reshaped = x_seq.view(batch_size * seq_len, h, w) # Squeeze channel dim for 1D CNN
        
        features = self.backbone(x_reshaped) 
        features = features.view(batch_size, seq_len, -1)
        
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :] 
        return self.fc(last_out)