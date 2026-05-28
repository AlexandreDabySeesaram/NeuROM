import torch

# mock output of model.Para_modes[mode][l](E[l][:,0].view(-1,1))
# It's a [batch, 1] tensor
batch = 10
m_modes = 3

out = torch.randn(batch, 1)

# What does [:, None] do?
out_none = out[:, None]
print("out[:, None] shape:", out_none.shape)

unsqueezed = torch.unsqueeze(out_none, dim=0)
print("unsqueezed shape:", unsqueezed.shape)

cat_tensor = torch.cat([unsqueezed, unsqueezed, unsqueezed], dim=0)
print("cat_tensor shape:", cat_tensor.shape)

# Now what is cat_tensor[:, :, 0]?
extracted = cat_tensor[:, :, 0]
print("extracted [:, :, 0] shape:", extracted.shape)

# Now what is torch.sum(..., dim=1)?
summed = torch.sum(extracted, dim=1)
print("summed shape:", summed.shape)

# Now S_g_m * summed
S_g_m = torch.randn(m_modes)
result = S_g_m * summed
print("S_g_m * summed shape:", result.shape)

