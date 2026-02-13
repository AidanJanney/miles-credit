import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from credit.models.wxformer.crossformer import CrossFormer
from credit.datasets.rMOM6_dataset import RegionalMOM6Dataset
from credit.samplers import MultiStepBatchSamplerSubset, DistributedMultiStepBatchSampler
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler

DEVICE = 'cuda'

def rollout(model, dataloader, device, num_steps=10, save_dir=None):
    model.eval()
    loader = iter(dataloader)
    if num_steps > len(dataloader):
        raise ValueError(f"Warning: num_steps ({num_steps}) is greater than the number of batches in the dataloader ({len(dataloader)}). Adjusting num_steps to {len(dataloader)}.")
    
    initial_batch = next(loader)
    x = torch.cat([initial_batch['prognostic'], initial_batch['dynamic_forcing'], initial_batch['static']], dim=1).to(device)
    
    predictions = []
    
    with torch.no_grad():
        for step in range(num_steps-1):
            y_pred = model(x)
            
            y_pred = y_pred[:,:,:,0:-1,0:-1] # remove boundary conditions, inaccurate, could be better
            
            predictions.append(y_pred) #ypred.cpu()???
            
            batch = next(loader)
            
            ## I need to do the boundary concatenation here. Right now my dataloader can return the boundaries, so something like:
            y_pred = torch.cat([y_pred, batch['north_boundary']], dim = 3)
            y_pred = torch.cat([y_pred, batch['east_boundary']], dim = 4)
            ## This is super adhoc, and I just probably incorporate a method in the dataloader for concatenation, or something in the rollout that reads the config, but for now this can work.
            ## EXCEPT: the dimensions are wonky. When concatting in xarray, it auto adds in the empty corner, but I believe torch will yell about this. This means I'll need a specific order of concatenation, and I'll need to pad one of the boundary conditions? It gets a bit messy, but not too complicated.
            
            x = torch.cat([y_pred, batch['dynamic_forcing'], batch['static']], dim=1).to(device)
            ## Definitely need to tidy up device logic here as well.
            
    ## TO-DO:
    # Add saving logic, need to reverse transform and convert to netcdf/zarr.

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
    
    # Restart from saved checkpoint
    checkpoint_path = "/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/test_full_domain/final_ocean_model.tar"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, prefetch_factor=2, persistent_workers=True, pin_memory=True)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    
    save_dir = "/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/test_full_domain/rollout"
    rollout(model, loader, DEVICE, num_steps=10, save_dir=save_dir)