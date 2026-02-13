import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from credit.models.wxformer.crossformer import CrossFormer
from credit.datasets.rMOM6_dataset import RegionalMOM6Dataset
from credit.samplers import MultiStepBatchSamplerSubset, DistributedMultiStepBatchSampler
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler
import csv

# Source - https://stackoverflow.com/a/7370824
# Posted by NPE, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-12, License - CC BY-SA 4.0
import time

import logging
import os
logger = logging.getLogger(__name__)



DEVICE = "cuda" # "cuda" -> Nvidia GPU, "mps" --> Mac GPU

def train(model, dataloader, dataloader_valid, config, loss_fn=nn.MSELoss(), optimizer = None, device=DEVICE, num_epochs=3, start_epoch = 0, save_dir=None, batch_size=1): # missing arg optimizer?

    if optimizer is None:
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
    best_valid_loss = float('inf')
    
    valid_keys = ['prognostic', 'dynamic_forcing', 'static', 'target']
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        file = open(os.path.join(save_dir, "training_log.csv"), mode='a', newline='')
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'train_loss', 'validation_loss'])
        
    for epoch in range(start_epoch, num_epochs):
        start = time.time()

        # Necessary for restarting runs
        sampler.set_epoch(epoch) 
        valid_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        print(f"Starting epoch {epoch}/{num_epochs-1}")
        
        for batch in dataloader:
        # for _ in range(10):
        #     batch = next(dataloader)
            if batch['prognostic'].shape[0] != batch_size: # Skip last batch if smaller than batch size (can happen with drop_last=False)
                    continue
        
            # https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html#limitations
            torch.compiler.cudagraph_mark_step_begin()
        
            xx = torch.cat([batch['prognostic'], batch['dynamic_forcing'], batch['static']], dim = 1).to(device)
            yy = batch['target'].to(device)
            
            ## Dummy Data
            # x = torch.randn(1, 67, 1, 100, 100).to(device)
            # y = torch.randn(1, 61, 1, 100, 100).to(device)

            optimizer.zero_grad()

            y_pred = model(xx)
            # print(y_pred)
            
            loss = loss_fn(y_pred, yy)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(loss.item())
        
        print(f"Starting validation for epoch {epoch}/{num_epochs-1}")
        avg_valid_loss = validate(model, dataloader_valid, loss_fn, device, batch_size)
    
        avg_loss = running_loss / len(dataloader)
        
        print(f"Epoch {epoch:03d} | loss = {avg_loss:.4e} | validation_loss = {avg_valid_loss:.4e}")
        if save_dir is not None:
            writer.writerow([epoch, avg_loss, avg_valid_loss])
            file.flush()
        
        if save_dir is not None:
            if (epoch+1) % 10 == 0: # Save every 10 epochs, should be an arg
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(save_dir, "latest_checkpoint.pt"))
            # if avg_valid_loss < best_valid_loss: 
            #     best_valid_loss = avg_valid_loss
            #     torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            
        end = time.time()
        print(f"Epoch {epoch} time: {end - start:.2f} seconds")

    if save_dir is not None:
        file.close()
        print("Training complete. Saving final model checkpoint.")
        
        unwrapped_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        
        final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_valid_loss,
        'config': config
        }
        torch.save(final_checkpoint, os.path.join(save_dir, "final_ocean_model.tar"))
        
def validate(model, dataloader_valid, loss_fn=nn.MSELoss(), device=DEVICE, batch_size=1):
    
    # Validation loop
    model.eval() # Set model to evaluation mode
    valid_loss = 0
    i = 0
    start_val = time.time()
    with torch.no_grad(): # Disable gradient tracking
        for batch_valid in dataloader_valid: 
            if batch_valid['prognostic'].shape[0] != batch_size: # Skip last batch if smaller than batch size (can happen with drop_last=False)
                continue
        # for _ in range(5):
        #     batch_valid = next(dataloader_valid)
            data_ready_time = time.time()
            torch.compiler.cudagraph_mark_step_begin()
            
            x_valid = torch.cat([batch_valid['prognostic'], batch_valid['dynamic_forcing'], batch_valid['static']], dim = 1).to(device)
            y_valid = batch_valid['target'].to(device)
            
            y_pred = model(x_valid)
            loss = loss_fn(y_pred, y_valid)
            valid_loss += loss.item()
            print(f"Validation Loss: {loss.item()}")
            
            model_time = time.time()
            i += 1
            if i % 10 == 0:
                print(f"Batch {i} - Data wait time: {data_ready_time - start_val:.4f}s, Model time: {model_time - data_ready_time:.4f}s")
            start_val = time.time()
    
    avg_valid_loss = valid_loss / len(dataloader_valid)
    return avg_valid_loss

if __name__ == "__main__":

    image_height = 458 # 458  # 640, 192
    image_width = 760 # 760 # 1280, 288
    levels = 50
    frames = 1
    output_frames = 1
    channels = 4
    surface_channels = 1
    input_only_channels = 6
    frame_patch_size = 0
    upsample_v_conv = True
    # attention_type = "scse_standard"
    # padding_conf = {"activate": False,}
    padding_conf = {"activate": True,
                    "mode": "regional",
                    # "pad_lat": [30, 30],
                    # "pad_lon": [30, 30]}
                    "pad_lat": [11, 11], # for use with big dims
                    "pad_lon": [20, 20]}

    # x = torch.randn(1, channels * levels + surface_channels + input_only_channels, frames, image_height, image_width).to(DEVICE)

    model = CrossFormer(image_height=image_height,
                        image_width=image_width,
                        frames=frames,
                        output_frames=output_frames,
                        frame_patch_size=frame_patch_size,
                        channels=channels,
                        surface_channels=surface_channels,
                        input_only_channels=input_only_channels,
                        levels=levels,
                        upsample_v_conv=upsample_v_conv,
                        dim=(128, 256, 512, 1024),
                        depth=(2, 2, 8, 2),
                        global_window_size=(20, 10, 5, 2),
                        local_window_size=5,
                        cross_embed_kernel_sizes=((4, 8, 16, 32), (2, 4), (2, 4), (2, 4)),
                        cross_embed_strides=(2, 2, 2, 2),
                        attn_dropout=0.0,
                        ff_dropout=0.0,
                        upsample_with_ps = True,
                        padding_conf=padding_conf).to(DEVICE)
    
    ## Restart from saved checkpoint
    # checkpoint_path = "/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/test_full_domain/final_ocean_model.tar"
    # checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    # state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    # model.load_state_dict(state_dict)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch = 0

    model = torch.compile(model, backend="cudagraphs")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # path = "/glade/work/ajanney/miles-credit/config/regional_mom6_tiny_upper_ocean.yaml"
    path = "/glade/work/ajanney/miles-credit/config/regional_mom6_example.yaml"

    with open(path) as cnfg:
        config = yaml.safe_load(cnfg)

    data_config = config["data"]

    source = "regional_MOM6"
    batch_size = 1
    num_workers = 8
    
    dataset = RegionalMOM6Dataset(data_config, return_target  = True)
    sampler = DistributedMultiStepBatchSampler(dataset=dataset, batch_size=batch_size, num_replicas=1, rank = 0) 
    # drop_last=True doesn't work as expected, it's related to num_replicas and rank, so we handle dropping last batch in the training loop instead
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    
    valid_dataset = RegionalMOM6Dataset(data_config, return_target  = True, return_validation = True)
    valid_sampler = DistributedMultiStepBatchSampler(dataset=valid_dataset, batch_size=batch_size, num_replicas=1, rank = 0) 
    # drop_last=True doesn't work as expected, it's related to num_replicas and rank, so we handle dropping last batch in the validation loop instead
    valid_loader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    
    num_epochs = 20
    save_dir = "/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/test_full_domain_3batch/"
    train(model, loader, valid_loader, config = data_config, optimizer=optimizer, num_epochs=num_epochs, start_epoch = start_epoch, device=DEVICE, save_dir = save_dir, batch_size = sampler.batch_size)