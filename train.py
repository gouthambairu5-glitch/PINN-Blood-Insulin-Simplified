import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import PINN
from physics import physics_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2000):
    x = torch.rand(100, 1).to(device)
    t = torch.rand(100, 1).to(device)

    loss = physics_loss(model, x, t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(epoch, loss.item())

x_test = torch.linspace(0, 1, 50)
t_test = torch.linspace(0, 1, 50)
X, T = torch.meshgrid(x_test, t_test, indexing="ij")

u_pred = model(X.reshape(-1,1).to(device), T.reshape(-1,1).to(device))
u_pred = u_pred.detach().cpu().numpy().reshape(50,50)

plt.imshow(u_pred, aspect='auto')
plt.title("Blood Flow Velocity PINN")
plt.colorbar()
plt.show()
