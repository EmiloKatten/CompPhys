import torch

# the long way to see what torch can do
ps = torch.tensor([1.,2,3])
all_dqdps = []
for p in ps:
    p.requires_grad_(True)
    q = p**2
    q.backward()
    dqdp = p.grad
    all_dqdps.append(dqdp.detach())
print(all_dqdps)

# torch does the same more easily by:
ps = torch.tensor([1.,2,3], requires_grad=True)
qs = ps ** 2
dq_dp = torch.autograd.grad(qs, ps, torch.ones_like(ps))
print(dq_dp)

# going even further
ps = torch.tensor([1.,2,3], requires_grad=True)
qs = ps ** 2
L = qs.sum()
L.backward()
dq_dp = ps.grad
print(dq_dp)

# plotting
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
with torch.no_grad():
    ax.plot(ps, dq_dp)

# using torch to fit a line
class Line(torch.nn.Module):
    def __init__(self, w=0, b=1):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(float(w)))
        self.b = torch.nn.Parameter(torch.tensor(float(b)))
        # converting into parameter type to tell the module
        # that we want gradients

    def forward(self, x):
        y = self.w * x + self.b
        return y

line = Line()
xs = torch.tensor([1.,2,3])
print(line(xs))

# now we can define a loss function to optimize with our Line class:
optimizer = torch.optim.Adam(line)
steps = 10
for _ in range(steps):
    optimizer.zero_grad() # wipe the previous calculated gradients
    ys_model = line(xs)
    loss = torch.mean((ys_model - ys)**2)
    loss.backward()
    optimizer.step()
