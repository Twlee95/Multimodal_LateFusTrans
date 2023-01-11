import torch
import math
import seaborn as sns
import matplotlib.pyplot as plt
pe = torch.zeros(5000, 10)
position = torch.arange(0, 5000, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, 10, 2).float() * (-math.log(10000.0) / 10))


pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)


print(div_term.size())
print(position.size())
print((div_term*position).size())
print(pe.size())
# torch.Size([5])
# torch.Size([5000, 1])
# torch.Size([5000, 5])
# torch.Size([5000, 10])

# plt.hist(pe)
# plt.show()


