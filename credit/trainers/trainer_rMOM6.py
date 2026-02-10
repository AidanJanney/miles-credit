import torch
import yaml
import torch.nn as nn
import torch.optim as optim
from credit.models.wxformer.crossformer import CrossFormer
from credit.datasets.rMOM6_dataset import RegionalMOM6Dataset
from credit.samplers import MultiStepBatchSamplerSubset, DistributedMultiStepBatchSampler
from torch.utils.data import Dataset, DataLoader, Sampler, DistributedSampler


DEVICE = "cuda" # "cuda" -> Nvidia GPU, "mps" --> Mac GPU

def train(model, dataloader, loss_fn=nn.MSELoss(), device=DEVICE, num_epochs=3): # missing arg optimizer

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    valid_keys = ['prognostic', 'dynamic_forcing', 'static', 'target']

    for epoch in range(num_epochs):

        running_loss = 0.0
        for _ in range(10): # len(dataloader)

            batch = next(dataloader)

            xx = torch.cat([batch['prognostic'], batch['dynamic_forcing'], batch['static']], dim = 1).to(device)
            yy = batch['target'].to(device)
            
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

        avg_loss = running_loss / 10
        print(f"Epoch {epoch:03d} | loss = {avg_loss:.4e}")


if __name__ == "__main__":

    image_height = 100 # 458  # 640, 192
    image_width = 100 # 760 # 1280, 288
    levels = 15
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
                    "pad_lat": [30, 30],
                    "pad_lon": [30, 30]}
                    # "pad_lat": [91, 91], # for use with big dims
                    # "pad_lon": [260, 260]}

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

    model = torch.compile(model, backend="cudagraphs")
    
    path = "/glade/work/ajanney/miles-credit/config/regional_mom6_tiny_upper_ocean.yaml"

    with open(path) as cnfg:
        config = yaml.safe_load(cnfg)

    data_config = config["data"]

    source = "regional_MOM6"
    dataset = RegionalMOM6Dataset(data_config, return_target  = True)
    sampler = DistributedMultiStepBatchSampler(dataset=dataset, batch_size=5, num_replicas=1, rank = 0) 
    
    loader = iter(DataLoader(dataset, batch_sampler=sampler))
    
    train(model, loader)
    
    # num_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters in the model: {num_params}")

    # y_pred = model(x)
    # print("Predicted shape:", y_pred.shape)
    # print(y_pred[0])