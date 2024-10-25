def copy_pretrained_weight(model, checkpoint):
    # Get all key
    list_keys = list(model.state_dict().keys())
    num_layers = len(list_keys)
    loaded_layers = 0 
    model_state_dict = model.state_dict()
    for key in list_keys:
        if checkpoint['model'][key].shape == model_state_dict[key].shape:
            model_state_dict[key].copy_(checkpoint['model'][key])
            loaded_layers += 1
    print(f"Load {loaded_layers}/{num_layers} layers from checkpoints")