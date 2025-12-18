import torch
import torch.optim as optim
import logging  # ADD THIS
from projections import cone_beam_projector
logger = logging.getLogger(__name__)

def adam_reconstruction_autograd(sino, A, img_params, sino_params,
                                 num_iter=100, lr=0.01, device='cpu'):
    """
    Simple ADAM reconstruction for Frontier
    """
    Nx, Ny, Nz = img_params.N_x, img_params.N_y, img_params.N_z
    
    # Initialize volume (X, Y, Z order)
    recon = torch.zeros((Nx, Ny, Nz), dtype=torch.float32, 
                       device=device, requires_grad=True)
    
    optimizer = optim.Adam([recon], lr=lr)
    loss_history = []
    
    logger.info(f"Starting ADAM reconstruction ({num_iter} iterations)...")
    
    for iteration in range(num_iter):
        optimizer.zero_grad()
        
        # Forward projection
        sino_pred = cone_beam_projector(recon, A, img_params, sino_params, device)
        
        # Loss
        loss = torch.mean((sino_pred - sino) ** 2)
        loss.backward()
        optimizer.step()
        
        # Non-negativity constraint
        with torch.no_grad():
            recon.data = torch.clamp(recon.data, min=0)
        
        loss_history.append(loss.item())
        
        if iteration % 10 == 0:
          logger.info(f"Iter {iteration:3d}: loss = {loss.item():.6e}")
    
    return recon.detach(), loss_history
