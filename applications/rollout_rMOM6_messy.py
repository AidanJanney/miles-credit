import time
import os
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from credit.models.wxformer.crossformer import CrossFormer
from credit.datasets.rMOM6_dataset import RegionalMOM6Dataset
from credit.samplers import MultiStepBatchSamplerSubset, DistributedMultiStepBatchSampler
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler

DEVICE = 'cuda'

def rollout(model, dataloader, device, num_forecast_steps=10, save_dir='.'):
    
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    loader = iter(dataloader)
    # if num_forecast_steps > len(dataloader):
    #     raise ValueError(f"Warning: num_forecast_steps ({num_forecast_steps}) is greater than the number of batches in the dataloader ({len(dataloader)}).")
    
    initial_batch = next(loader)
    static_fields = initial_batch['static'].to(device)
    x = torch.cat([initial_batch['prognostic'], initial_batch['dynamic_forcing'], initial_batch['static']], dim=1).to(device)
    
    torch.save(initial_batch['metadata'], f"{save_dir}/metadata.pt")
    torch.save(initial_batch['prognostic'].cpu(), f"{save_dir}/initial_prognostic.pt")
    
    print("Initial batch keys:", initial_batch.keys())
    
    predictions = []
    
    with torch.no_grad():
        for step in range(num_forecast_steps-1):
            start = time.time() 
            torch.compiler.cudagraph_mark_step_begin()
            
            y_pred = model(x)
            end = time.time()
            print(f"Prediction time: {end - start:.2f} seconds")
            
            y_pred = y_pred*static_fields[:,0:1,...]
            y_pred = y_pred[:,:,:,0:-1,0:-1] # remove boundary conditions, inaccurate, could be better
            
            # predictions.append(y_pred.cpu()) #ypred.cpu()???
            torch.save(y_pred.cpu(), f"{save_dir}/prediction_step_{step}.pt")
            predictions.append(1)
            
            batch = next(loader)
            end = time.time()
            print(f"Next Loader time: {end - start:.2f} seconds")
            
            print("Step:", step, "Batch keys:", batch.keys())
            
            north_boundary = batch['north_boundary'].to(device)
            east_boundary = batch['east_boundary'].to(device)
            dynamic_forcing = batch['dynamic_forcing'].to(device)
            
            ## I need to do the boundary concatenation here. Right now my dataloader can return the boundaries, so something like:
            y_pred = torch.cat([y_pred, north_boundary], dim = 3)
            
            ne_corner = torch.zeros_like(east_boundary[:,:,:,0:1,:])
            east_with_corner = torch.cat([ne_corner, east_boundary], dim=3)
            y_pred = torch.cat([y_pred, east_with_corner], dim = 4)
            
            end = time.time()
            print(f"OBCs time: {end - start:.2f} seconds")
            
            ## This is super adhoc, and I just probably incorporate a method in the dataloader for concatenation, or something in the rollout that reads the config, but for now this can work.
            ## EXCEPT: the dimensions are wonky. When concatting in xarray, it auto adds in the empty corner, but I believe torch will yell about this. This means I'll need a specific order of concatenation, and I'll need to pad one of the boundary conditions? It gets a bit messy, but not too complicated.
            
            x = torch.cat([y_pred, dynamic_forcing, static_fields], dim=1).to(device)
            
            print(f"Completed step {step+1}/{num_forecast_steps-1}, len(predictions)={len(predictions)}, y_pred shape={y_pred.shape}")
            
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
    print("Loading model from checkpoint...")
    checkpoint_path = "/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/test_full_domain_cont/final_ocean_model_copy.tar"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state_dict)
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # start_epoch = 0

    model = torch.compile(model, backend="cudagraphs")
    # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    # path = "/glade/work/ajanney/miles-credit/config/regional_mom6_tiny_upper_ocean.yaml"
    path = "/glade/work/ajanney/miles-credit/config/regional_mom6_example.yaml"

    with open(path) as cnfg:
        config = yaml.safe_load(cnfg)

    data_config = config["data"]
    data_config["forecast_len"] = 365 # config["predict"]["forecast_len"] # Ensure data_config has forecast_len for dataset initialization, can be used for both train and valid datasets since they should have the same forecast_len

    source = "regional_MOM6"
    batch_size = 1
    num_workers = 8
    
    print("Initializing dataset and dataloader...")
    dataset = RegionalMOM6Dataset(data_config, predict = True)
    sampler = DistributedMultiStepBatchSampler(dataset=dataset, batch_size=batch_size, num_replicas=1, rank = 0, shuffle = False) 
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in the model: {num_params}")
    
    # save_dir = "/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/test_full_domain/rollout"
    save_dir = "/glade/work/ajanney/Regional_Ocean_Emulation/test_full_domain/rollout_long"
    
    print(f"len(loader) = {len(loader)}")
    print(f"dataset.num_forecast_steps = {dataset.num_forecast_steps}")
    num_batches = sum(1 for _ in sampler)
    print(f"num_batches in sampler = {num_batches}")
    rollout(model, loader, DEVICE, num_forecast_steps=dataset.num_forecast_steps, save_dir=save_dir)