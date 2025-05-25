import torch 
from latent_diffusion import Simple1DUNet, GaussianDiffusion
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import polars as pl
clean_path = '/pfs/work9/workspace/scratch/ka_uhicv-blah/latent_diffusion/10_clean.pt'
dirty_path = '/pfs/work9/workspace/scratch/ka_uhicv-blah/latent_diffusion/10_1.pt'
clean_path = '10_clean.pt'
dirty_path = '10_1.pt'
latent_clean = torch.load(clean_path).float()
latent_noisy = torch.load(dirty_path).float()
#
# Prepare data
dataset = TensorDataset(latent_noisy)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
clean_df = pl.read_csv('../clean.csv')
speaker1_df = pl.read_csv('../speaker_1.csv')


model =Simple1DUNet(dim=1280).to("cuda")
dict_path = 'simple_diffuion_model.pth'
trained_dict = torch.load(dict_path)
model.load_state_dict(trained_dict)
diffusion = GaussianDiffusion(model).to("cuda")

@torch.no_grad()
def denoise_latent(noisy_latent, steps=100):
    model.eval()
    x = noisy_latent.clone().to("cuda")
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t] * x.shape[0], device="cuda")
        t_emb = diffusion.time_embedding(t_tensor, x.shape[-1])
        pred_noise = model(x.permute(0, 2, 1), t_emb).permute(0, 2, 1)
        alpha_t = diffusion.alpha_cumprod[t]
        x = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
    return x
clean_states = []
for batch in tqdm(loader):
    clean_state= denoise_latent(batch[0])
    clean_states.append(clean_state)
breakpoint() 
