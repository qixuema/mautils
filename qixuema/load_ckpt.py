import torch



def extract_keys(d, parent_key=''):
    keys_list = []
    for k, v in d.items():
        # 构建新的键路径
        new_key = f"{parent_key}/{k}" if parent_key else k
        # 如果值是字典，递归调用
        if isinstance(v, dict):
            keys_list.extend(extract_keys(v, new_key))
        else:
            keys_list.append(new_key)
    return keys_list

def save_static_dict_keys(static_dict, file_path='static_dict_keys.json'):
    # 展平嵌套的键
    checkpoint_keys  = extract_keys(static_dict)

    # 保存键到文本文件
    with open(file_path, 'w') as f:
        for key in checkpoint_keys:
            f.write(f"{key}\n")


def load_ema(model, load_path, device='cpu', strict=True):
    ema_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = ema_ckpt['ema']
    # 创建新的状态字典，移除'model.'前缀
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('online_model.'):
            new_key = key.replace('online_model.', '')
            new_model_state_dict[new_key] = value


    save_static_dict_keys(new_model_state_dict, file_path='static_dict_keys.txt')
    save_static_dict_keys(model.state_dict(), file_path='model_state_dict_keys.txt')
    
    # model.load_state_dict(ema_ckpt['model'], strict=False)
    
    load_status = model.load_state_dict(new_model_state_dict, strict=strict)
    # print(device)
    print(f"model loaded from {load_path}")
    
    return model