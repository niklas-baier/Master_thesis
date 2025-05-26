import torch
import torch.nn as nn
import torch.nn.functional as F
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        h += self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1)
        self.to_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.group_norm(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(b, c, h * w).transpose(-1, -2)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w).transpose(-1, -2)
        
        # Attention
        attn = torch.softmax(torch.bmm(q, k) / (c ** 0.5), dim=-1)
        out = torch.bmm(attn, v).transpose(-1, -2).view(b, c, h, w)
        
        return self.to_out(out) + x

# ENHANCED VERSION FOR 256x256 IMAGES
class RectifiedFlowUNet256(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_embedding_dim=256):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)  # CHANGED: 64 -> 128
        
        # Encoder - MORE LEVELS for 256x256
        # Level 1: 256x256 -> 128x128
        self.down1 = nn.ModuleList([
            ResidualBlock(128, 128, time_embedding_dim),
            ResidualBlock(128, 128, time_embedding_dim)
        ])
        self.down_conv1 = nn.Conv2d(128, 256, 3, stride=2, padding=1)  # CHANGED: 64->128 to 128->256
        
        # Level 2: 128x128 -> 64x64
        self.down2 = nn.ModuleList([
            ResidualBlock(256, 256, time_embedding_dim),
            ResidualBlock(256, 256, time_embedding_dim)
        ])
        self.down_conv2 = nn.Conv2d(256, 512, 3, stride=2, padding=1)  # CHANGED: 128->256 to 256->512
        
        # Level 3: 64x64 -> 32x32 - NEW LEVEL
        self.down3 = nn.ModuleList([
            ResidualBlock(512, 512, time_embedding_dim),
            ResidualBlock(512, 512, time_embedding_dim)
        ])
        self.down_conv3 = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
        
        # Level 4: 32x32 -> 16x16 - NEW LEVEL  
        self.down4 = nn.ModuleList([
            ResidualBlock(1024, 1024, time_embedding_dim),
            ResidualBlock(1024, 1024, time_embedding_dim)
        ])
        self.down_conv4 = nn.Conv2d(1024, 1024, 3, stride=2, padding=1)
        
        # Middle - ENHANCED
        self.middle = nn.ModuleList([
            ResidualBlock(1024, 1024, time_embedding_dim),
            AttentionBlock(1024),  # Attention at bottleneck
            ResidualBlock(1024, 1024, time_embedding_dim),
            AttentionBlock(1024),  # ADDED: More attention
            ResidualBlock(1024, 1024, time_embedding_dim)
        ])
        
        # Decoder - CORRESPONDING LEVELS
        # Up 1: 16x16 -> 32x32
        self.up_conv1 = nn.ConvTranspose2d(1024, 1024, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(2048, 1024, time_embedding_dim),  # 1024 + 1024 skip
            ResidualBlock(1024, 1024, time_embedding_dim)
        ])
        
        # Up 2: 32x32 -> 64x64
        self.up_conv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(1024, 512, time_embedding_dim),  # 512 + 512 skip
            ResidualBlock(512, 512, time_embedding_dim)
        ])
        
        # Up 3: 64x64 -> 128x128
        self.up_conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([
            ResidualBlock(512, 256, time_embedding_dim),  # 256 + 256 skip
            ResidualBlock(256, 256, time_embedding_dim)
        ])
        
        # Up 4: 128x128 -> 256x256
        self.up_conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up4 = nn.ModuleList([
            ResidualBlock(256, 128, time_embedding_dim),  # 128 + 128 skip
            ResidualBlock(128, 128, time_embedding_dim)
        ])
        
        self.conv_out = nn.Conv2d(128, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        time_emb = self.time_embedding(t)
        
        # Store skip connections
        skips = []
        
        # Initial
        h = self.conv_in(x)
        
        # Encoder
        for block in self.down1:
            h = block(h, time_emb)
        skips.append(h)
        h = self.down_conv1(h)
        
        for block in self.down2:
            h = block(h, time_emb)
        skips.append(h)
        h = self.down_conv2(h)
        
        for block in self.down3:
            h = block(h, time_emb)
        skips.append(h)
        h = self.down_conv3(h)
        
        for block in self.down4:
            h = block(h, time_emb)
        skips.append(h)
        h = self.down_conv4(h)
        
        # Middle
        for i, block in enumerate(self.middle):
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, time_emb)
        
        # Decoder
        h = self.up_conv1(h)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up1:
            h = block(h, time_emb)
        
        h = self.up_conv2(h)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up2:
            h = block(h, time_emb)
        
        h = self.up_conv3(h)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up3:
            h = block(h, time_emb)
        
        h = self.up_conv4(h)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up4:
            h = block(h, time_emb)
        
        return self.conv_out(h)


# FOR WHISPER HIDDEN STATES (1500x1280)
class RectifiedFlowUNetWhisper(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_embedding_dim=256):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        
        # For 1500x1280, we need to handle non-power-of-2 dimensions
        # Strategy: Use adaptive pooling and careful upsampling
        
        # Initial conv
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Encoder with adaptive downsampling
        # Level 1: 1500x1280 -> 750x640
        self.down1 = nn.ModuleList([
            ResidualBlock(64, 128, time_embedding_dim),
            ResidualBlock(128, 128, time_embedding_dim)
        ])
        self.adaptive_pool1 = nn.AdaptiveAvgPool2d((750, 640))
        
        # Level 2: 750x640 -> 375x320  
        self.down2 = nn.ModuleList([
            ResidualBlock(128, 256, time_embedding_dim),
            ResidualBlock(256, 256, time_embedding_dim)
        ])
        self.adaptive_pool2 = nn.AdaptiveAvgPool2d((375, 320))
        
        # Level 3: 375x320 -> 188x160 (approximately)
        self.down3 = nn.ModuleList([
            ResidualBlock(256, 512, time_embedding_dim),
            ResidualBlock(512, 512, time_embedding_dim)
        ])
        self.adaptive_pool3 = nn.AdaptiveAvgPool2d((188, 160))
        
        # Level 4: 188x160 -> 94x80
        self.down4 = nn.ModuleList([
            ResidualBlock(512, 512, time_embedding_dim),
            ResidualBlock(512, 512, time_embedding_dim)
        ])
        self.adaptive_pool4 = nn.AdaptiveAvgPool2d((94, 80))
        
        # Level 5: 94x80 -> 47x40
        self.down5 = nn.ModuleList([
            ResidualBlock(512, 1024, time_embedding_dim),
            ResidualBlock(1024, 1024, time_embedding_dim)
        ])
        self.adaptive_pool5 = nn.AdaptiveAvgPool2d((47, 40))
        
        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(1024, 1024, time_embedding_dim),
            AttentionBlock(1024),
            ResidualBlock(1024, 1024, time_embedding_dim)
        ])
        
        # Decoder with interpolation upsampling
        self.up1 = nn.ModuleList([
            ResidualBlock(2048, 1024, time_embedding_dim),  # 1024 + 1024 skip
            ResidualBlock(1024, 512, time_embedding_dim)
        ])
        
        self.up2 = nn.ModuleList([
            ResidualBlock(1024, 512, time_embedding_dim),  # 512 + 512 skip
            ResidualBlock(512, 512, time_embedding_dim)
        ])
        
        self.up3 = nn.ModuleList([
            ResidualBlock(1024, 512, time_embedding_dim),  # 512 + 512 skip
            ResidualBlock(512, 256, time_embedding_dim)
        ])
        
        self.up4 = nn.ModuleList([
            ResidualBlock(512, 256, time_embedding_dim),  # 256 + 256 skip
            ResidualBlock(256, 128, time_embedding_dim)
        ])
        
        self.up5 = nn.ModuleList([
            ResidualBlock(256, 128, time_embedding_dim),  # 128 + 128 skip
            ResidualBlock(128, 64, time_embedding_dim)
        ])
        
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        time_emb = self.time_embedding(t)
        
        # Store skip connections and their sizes
        skips = []
        sizes = [(1500, 1280)]
        
        # Initial
        h = self.conv_in(x)
        
        # Encoder with size tracking
        for block in self.down1:
            h = block(h, time_emb)
        skips.append(h)
        h = self.adaptive_pool1(h)
        sizes.append((750, 640))
        
        for block in self.down2:
            h = block(h, time_emb)
        skips.append(h)
        h = self.adaptive_pool2(h)
        sizes.append((375, 320))
        
        for block in self.down3:
            h = block(h, time_emb)
        skips.append(h)
        h = self.adaptive_pool3(h)
        sizes.append((188, 160))
        
        for block in self.down4:
            h = block(h, time_emb)
        skips.append(h)
        h = self.adaptive_pool4(h)
        sizes.append((94, 80))
        
        for block in self.down5:
            h = block(h, time_emb)
        skips.append(h)
        h = self.adaptive_pool5(h)
        
        # Middle
        for i, block in enumerate(self.middle):
            if isinstance(block, AttentionBlock):
                h = block(h)
            else:
                h = block(h, time_emb)
        
        # Decoder with size restoration
        h = F.interpolate(h, size=sizes[-2], mode='bilinear', align_corners=False)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up1:
            h = block(h, time_emb)
        
        h = F.interpolate(h, size=sizes[-3], mode='bilinear', align_corners=False)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up2:
            h = block(h, time_emb)
        
        h = F.interpolate(h, size=sizes[-4], mode='bilinear', align_corners=False)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up3:
            h = block(h, time_emb)
        
        h = F.interpolate(h, size=sizes[-5], mode='bilinear', align_corners=False)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up4:
            h = block(h, time_emb)
        
        h = F.interpolate(h, size=(1500, 1280), mode='bilinear', align_corners=False)
        h = torch.cat([h, skips.pop()], dim=1)
        for block in self.up5:
            h = block(h, time_emb)
        
        return self.conv_out(h)


# Updated RectifiedFlow class with better sampling
class RectifiedFlow:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def forward_process(self, x0, x1, t):
        """Forward process: x_t = t * x1 + (1 - t) * x0"""
        t = t.view(-1, 1, 1, 1)
        return t * x1 + (1 - t) * x0
    
    def velocity_loss(self, x0, x1):
        """Compute the velocity matching loss"""
        batch_size = x0.shape[0]
        
        # Sample random times with better distribution
        t = torch.rand(batch_size, device=self.device)
        
        # Apply time weighting for better training stability
        weights = 1.0 / (t + 1e-8)  # Emphasize early times
        weights = weights / weights.mean()
        
        # Get interpolated samples
        x_t = self.forward_process(x0, x1, t)
        
        # Concatenate source image with current state
        model_input = torch.cat([x0, x_t], dim=1)
        
        # Predict velocity
        v_pred = self.model(model_input, t)
        
        # True velocity is x1 - x0
        v_true = x1 - x0
        
        # Weighted L2 loss
        loss = F.mse_loss(v_pred, v_true, reduction='none')
        loss = (loss * weights.view(-1, 1, 1, 1)).mean()
        
        return loss
    
    def sample(self, x0, num_steps=100, method='euler'):  # CHANGED: More steps, different methods
        """Generate samples using various ODE solvers"""
        self.model.eval()
        
        with torch.no_grad():
            x = x0.clone()
            dt = 1.0 / num_steps
            
            if method == 'euler':
                for i in range(num_steps):
                    t = torch.full((x0.shape[0],), i * dt, device=self.device)
                    model_input = torch.cat([x0, x], dim=1)
                    v = self.model(model_input, t)
                    x = x + dt * v
            
            elif method == 'rk4':  # ADDED: Runge-Kutta 4th order
                for i in range(num_steps):
                    t = torch.full((x0.shape[0],), i * dt, device=self.device)
                    
                    # RK4 steps
                    model_input = torch.cat([x0, x], dim=1)
                    k1 = self.model(model_input, t)
                    
                    model_input = torch.cat([x0, x + 0.5 * dt * k1], dim=1)
                    k2 = self.model(model_input, t + 0.5 * dt)
                    
                    model_input = torch.cat([x0, x + 0.5 * dt * k2], dim=1)
                    k3 = self.model(model_input, t + 0.5 * dt)
                    
                    model_input = torch.cat([x0, x + dt * k3], dim=1)
                    k4 = self.model(model_input, t + dt)
                    
                    x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return x

# Usage examples:
# For 256x256 images:
# model = RectifiedFlowUNet256(in_channels=2, out_channels=1)
# flow = RectifiedFlow(model, device)

# For Whisper hidden states (1500x1280):
# model = RectifiedFlowUNetWhisper(in_channels=2, out_channels=1) 
# flow = RectifiedFlow(model, device)
