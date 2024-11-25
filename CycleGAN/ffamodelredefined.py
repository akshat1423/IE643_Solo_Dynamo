ckp = torch.load("../input/model/tensorflow2/default/1/model2.pk", map_location=device)
ffa_model = FFA(gps=gps, blocks=blocks)
ffa_model = nn.DataParallel(ffa_model)
ffa_model.load_state_dict(ckp['model'])
ffa_model.eval()
ffa_model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ffa_model = nn.DataParallel(ffa_model, device_ids=[0]).to(device)
ckp = torch.load(model_dir, map_location=device)

# Normalization transformation to apply after image processing
normalization = transforms.Normalize(mean=[-0.64/0.14, -0.6/0.15, -0.58/0.152], std=[1/0.14, 1/0.15, 1/0.152])

# Before saving the enhanced output, denormalize and clamp
def process_output(output_tensor):
    # Apply the reverse normalization for each image in the batch
    normalized_batch = []
    for img in output_tensor:
        img = normalization(img)  # Apply normalization to a single image (C, H, W)
        img = img.clamp(0, 1)  # Clamp the values to the [0, 1] range
        normalized_batch.append(img)
    
    # Stack the processed images back into a batch
    return torch.stack(normalized_batch)

