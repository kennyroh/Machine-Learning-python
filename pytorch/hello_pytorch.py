import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# 입력 -> 은닉 (3,5)
class fir_model(nn.Module):
    def __init__(self):
        super(fir_model, self).__init__()
        # 가중치행렬1
        self.lin1 = nn.Linear(3, 5)
        # 가중치행렬2
        self.lin2 = nn.Linear(5, 2)

    # forward propagation
    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.sigmoid(x)
        return x


model = fir_model()
opt = optim.SGD(model.parameters(), lr=0.001)  # learning rate

print(model)

criterion = nn.MSELoss()

x = torch.Tensor(np.random.normal(size=3))
y = torch.Tensor(np.random.normal(size=2))

opt.zero_grad()
y_infer = model(x)  # forward propagation
loss = criterion(y_infer, y)  # 오차함수로 오차 구함.
loss.backward()
opt.step()

device = torch.device('mps:0' if torch.backends.mps.is_available() else 'cpu')
print (f"PyTorch version:{torch.__version__}") # 1.12.1 이상
print(f"MPS 장치를 지원하도록 build 되었는지: {torch.backends.mps.is_built()}") # True 여야 합니다.
print(f"MPS 장치가 사용 가능한지: {torch.backends.mps.is_available()}") # True 여야 합니다.

# python -c 'import platform;print(platform.platform())'
