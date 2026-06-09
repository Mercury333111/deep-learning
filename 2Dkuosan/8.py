import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate spiral data
def make_spiral(n=2000):
    theta = torch.linspace(0, 4 * math.pi, n)
    r = theta
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    data = torch.stack([x, y], dim=1)
    data = data / data.abs().max()
    return data

# Diffusion schedule (given)
T = 1000
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alpha = torch.sqrt(alphas).to(device)
sqrt_1malpha = torch.sqrt(1 - alphas).to(device)
sqrt_alpha_bar = torch.sqrt(alphas_cumprod).to(device)
sqrt_1malpha_bar = torch.sqrt(1 - alphas_cumprod).to(device)

# 时间位置编码
def timestep_embedding(t, dim=128):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half)/half).to(device)
    args = t[:, None] * freqs[None,:]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb

# TODO: implement model 5层MLP
class Model(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        # 5层MLP：输入(2+时间编码)→4隐藏层→输出2
        layers = []
        input_dim = hidden_dim + 2
        # 5层全连接
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Softplus())
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

    def forward(self, x, t):
        # x:[B,2], t:[B]
        t_emb = timestep_embedding(t, self.hidden_dim)
        feat = torch.cat([x, t_emb], dim=-1)
        pred = self.mlp(feat)
        return pred

# TODO: training function 支持三种训练目标 eps / x0 / v
def train(model, mode='eps', epochs=10000):
    data = make_spiral(5000).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_list = []
    for epoch in range(epochs):
        idx = torch.randint(0, data.shape[0], (256,))
        x0 = data[idx]
        t = torch.randint(0, T, (256,), device=device)
        alpha_bar = alphas_cumprod[t].unsqueeze(1)
        sab = sqrt_alpha_bar[t].unsqueeze(1)
        s1ab = sqrt_1malpha_bar[t].unsqueeze(1)
        noise = torch.randn_like(x0)
        xt = sab * x0 + s1ab * noise
        # 模型预测
        pred = model(xt, t)
        # 三种target
        if mode == 'eps':
            target = noise
        elif mode == 'x0':
            target = x0
        elif mode == 'v':
            target = sab * noise - s1ab * x0
        else:
            raise NotImplementedError
        loss = ((pred - target) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"epoch:{epoch}, loss:{loss.item():.5f}")
            loss_list.append(loss.item())
    return model

# TODO: sampling 反向DDPM采样
@torch.no_grad()
def sample(model, mode='eps', n=1000):
    x = torch.randn(n, 2).to(device)
    for t in reversed(range(T)):
        bt = betas[t]
        ab = alphas[t]
        sab = sqrt_alpha_bar[t]
        s1ab = sqrt_1malpha_bar[t]
        pred = model(x, torch.full((n,), t, device=device))
        # 根据预测目标统一转为预测噪声
        if mode == 'eps':
            pred_noise = pred
        elif mode == 'x0':
            pred_x0 = pred
            pred_noise = (x - sab * pred_x0) / s1ab
        elif mode == 'v':
            pred_v = pred
            pred_x0 = sab * x - s1ab * pred_v
            pred_noise = (x - sab * pred_x0) / s1ab
        # DDPM反向采样
        if t > 0:
            sigma = torch.sqrt(bt)
            z = torch.randn_like(x)
        else:
            sigma = 0.0
            z = 0.0
        x = (1 / torch.sqrt(ab)) * (x - bt / s1ab * pred_noise) + sigma * z
    return x.cpu()

# 评估指标
def compute_mmd(x, y, sigma=0.1):
    def kernel(a, b):
        dist = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return torch.exp(-dist / (2 * sigma ** 2))
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()

def chamfer_distance(x, y):
    dist = torch.cdist(x, y)
    return dist.min(dim=1)[0].mean() + dist.min(dim=0)[0].mean()

if __name__ == "__main__":
    # 切换mode: eps / x0 / v
    train_mode = "x0"
    model = Model(hidden_dim=128).to(device)
    model = train(model, mode=train_mode, epochs=10000)
    real = make_spiral(2000)
    fake = sample(model, mode=train_mode, n=2000)
    print('MMD:', compute_mmd(real, fake).item())
    print('Chamfer:', chamfer_distance(real, fake).item())
    plt.figure(figsize=(6,6))
    plt.scatter(real[:,0], real[:,1], s=5, label='real', alpha=0.7)
    plt.scatter(fake[:,0], fake[:,1], s=5, label='generated', alpha=0.7)
    plt.legend()
    plt.title(f"Diffusion 2D spiral | train_target={train_mode}")
    plt.show()