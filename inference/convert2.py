import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
import ctypes
from safetensors.torch import safe_open, save_file
from kernel import weight_dequant


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}

EmbedsInOneFile = 256
EmbedsZKDir = "../zkdata/embeds/"

wkv_b_1_rescales = [32, 34, 37, 36, 33, 32, 33, 33, 30, 32,
                   32, 30, 31, 30, 29, 30, 29, 30, 29, 29,
                   29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
                   29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
                   29, 29, 29, 29, 29, 29, 29, 29, 30, 30,
                   29, 29, 30, 30, 30, 30, 29, 30, 30, 29, 30]

wkv_b_2_rescales = [31, 32, 32, 31, 32, 30, 30, 30, 30, 30,
                   30, 30, 30, 29, 29, 29, 29, 30, 29, 29,
                   29, 29, 29, 29, 30, 30, 30, 29, 29, 29,
                   29, 29, 30, 29, 30, 29, 30, 29, 29, 29,
                   30, 29, 29, 29, 29, 30, 29, 30, 30, 30,
                   29, 29, 29, 30, 30, 29, 29, 29, 30, 30, 30]

wo_rescales = [31, 32, 32, 32, 32, 31, 32, 31, 31, 31,
              31, 31, 31, 31, 30, 31, 31, 32, 31, 31,
              31, 30, 30, 30, 30, 30, 30, 30, 30, 30,
              30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
              30, 30, 30, 31, 30, 31, 30, 30, 31, 31,
              31, 30, 31, 31, 31, 30, 31, 31, 31, 31, 32 ]

gate_rescales = [0, 0, 0, 33, 32, 32, 32, 31, 32, 31, 30,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                32, 31, 32, 31, 32, 32, 32, 32, 31, 32,
                32, 31, 32, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 33, 33, 33, 33, 33, 32, 32 ]

w1_rescales = [32, 32, 32]
w2_rescales = [31, 32, 31]
w3_rescales = [32, 33, 32]

shared_w1_rescales = [0, 0, 0, 30, 30, 29, 29, 29, 28, 29,
                      29, 28, 29, 29, 29, 29, 29, 29, 29, 29,
                      29, 29, 29, 30, 30, 30, 30, 30, 30, 30,
                      30, 30, 30, 30, 29, 29, 30, 29, 29, 30,
                      29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
                      29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29]

shared_w2_rescales = [0, 0, 0, 30, 30, 30, 30, 30, 29, 29,
                      30, 29, 29, 29, 30, 30, 30, 30, 30, 29,
                      29, 29, 29, 29, 29, 29, 29, 30, 30, 29,
                      29, 29, 29, 29, 29, 29, 29, 30, 29, 29,
                      29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
                      29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30]

shared_w3_rescales = [0, 0, 0, 30, 30, 30, 30, 30, 29, 29,
                      30, 29, 29, 29, 30, 30, 30, 29, 30, 29,
                      29, 29, 29, 29, 29, 29, 30, 30, 30, 30,
                      29, 29, 29, 29, 29, 29, 29, 30, 30, 29,
                      30, 29, 29, 29, 29, 30, 29, 29, 30, 30,
                      29, 30, 30, 30, 29, 29, 30, 30, 30, 29, 28]

layer_state_dict0 = [{} for _ in range(61)]
layer_state_dict = [{} for _ in range(61)]

experts = [ [{} for _j in range(256)] for _i in range(61)]

def getF32PrintStr(ele):
    v = int(ele.cpu().view(torch.uint32).item())
    ex = str((v >> 23 & 0xFF) - 127)
    r = '(1+' + str(v & 0x7FFFFF) + '/8388608)'
    if v & 0x80000000:
        vstr = '-' + r + '*2^' + ex
    else:
        vstr = r + '*2^' + ex
    return vstr

def getBF16PrintStr(ele):
    v = int(ele.cpu().view(torch.uint16).item())
    ex = v >> 7 & 0xFF
    r = '(1+' + str(v & 0x7F) + '/128)'
    rraw = v & 0x7F

    if v & 0x8000:
        vstr = '-' + r + '*2^' + str(ex - 127)
    else:
        vstr = r + '*2^' + str(ex - 127)
    return vstr

def getBF8PrintStr(ele):
    v = int(ele.cpu().view(torch.uint8).item())
    ex = str((v >> 3 & 0xF) - 7)
    r = '(1+' + str(v & 0x7) + '/8)'

    if v & 0x80:
        vstr = '-' + r + '*2^' + ex
    else:
        vstr = r + '*2^' + ex

    if ex == -7 or ex == 8:
        print(vstr)
    return vstr

def mem(i):
    a = torch.cuda.memory_allocated()/1024**2
    r = torch.cuda.memory_reserved()/1024**2
    m = torch.cuda.max_memory_allocated()/1024**2
    print(f"{i} allocated={a:.1f}MB, reserved={r:.1f}MB, max={m:.1f}MB", flush=True)

def handle_expert_w(layer_id, expert_id, idx, param_weight, weight_name, scale, typ, shape, experts_save_path):
    global layer_state_dict0
    global experts

    scale_name = weight_name.replace('weight', 'scale')
    param_scale = layer_state_dict0[layer_id][scale_name]

    weight = weight_dequant(param_weight.cuda(), param_scale.cuda())
    # scale = experts_w3_rescales[layer_id][expert_id]
    rescale = 2 ** scale
    param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
    # layer_state_dict[layer_id][weight_name] = param_int.cpu()
    # layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)
    weight_name2 = f'w{idx}.weight'
    scale_name2 = f'w{idx}.scale'
    experts[layer_id][expert_id][weight_name2] = param_int
    experts[layer_id][expert_id][scale_name2] = torch.tensor(scale, dtype=torch.int32)

    if len(experts[layer_id][expert_id]) == 6: # w1, w2, w3 以及对应的 scale
        save_file(experts[layer_id][expert_id], os.path.join(experts_save_path, f"{expert_id}.safetensors"))
        experts[layer_id][expert_id] = {}

    print(f'layer {layer_id} expert {expert_id} w{idx} type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {scale_name}')

def saveTensor(fileName, t):
    with open(fileName, "w", encoding="utf-8") as f:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        with open(fileName, "wb") as f:
            f.write(t.numpy().tobytes(order="C"))

def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
    """
    torch.cuda.set_device(0)
    # 设置pytorch计算时的默认数据类型。这里使用的是BF16
    torch.set_default_dtype(torch.bfloat16)
    # 限制 PyTorch 在 CPU 计算时最多使用 8 个线程，防止过多线程竞争资源：
    torch.set_num_threads(8)
    # 设定随机种子，保证不同进程初始化时随机数相同。
    torch.manual_seed(965)

    # n_local_experts = n_experts // mp
    # state_dicts = [{} for _ in range(mp)]

    head_state_dict = {}
    norm_state_dict = {}
    embed_state_dict = {}

    experts_w1_rescales = []
    experts_w2_rescales = []
    experts_w3_rescales = []

    with open("w1.txt", "r", encoding="utf-8") as f1:
        for line in f1:
            layer_line = line.strip().split()
            int_list = [int(s) for s in layer_line]
            experts_w1_rescales.append(int_list)

    with open("w2.txt", "r", encoding="utf-8") as f2:
        for line in f2:
            layer_line = line.strip().split()
            int_list = [int(s) for s in layer_line]
            experts_w2_rescales.append(int_list)

    with open("w3.txt", "r", encoding="utf-8") as f3:
        for line in f3:
            layer_line = line.strip().split()
            int_list = [int(s) for s in layer_line]
            experts_w3_rescales.append(int_list)

    # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
    # glob是python自己带的一个文件操作相关模块，用它可以查找符合自己目的的文件，类似于Windows下的文件搜索
    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            print('Opening ' + file_path, flush=True)
            for name in f.keys():
                # print('name 1: ', name)
                if "model.layers.61" in name:
                    continue

                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                # print('key::: ' + key)
                new_key, dim = mapping[key]
                # print('dim::: ' + str(dim))
                name = name.replace(key, new_key)

                ns = name.split(".")
                comp = ns[0]
                if comp == 'head':
                    name2 = name[len('head.'):]
                    print('head: ' + name2)

                    param_int =  (param.to(torch.float32) * (2 ** 43)).round().to(torch.int64)
                    head_state_dict[name2] = param_int
                elif comp == 'norm':
                    name2 = name[len('norm.'):]
                    print('norm: ' + name2)

                    param_int =  (param.to(torch.float32) * (2 ** 15)).round().to(torch.int64)
                    norm_state_dict[name2] = param_int
                elif comp == 'embed':
                    name2 = name[len('embed.'):]
                    print('embed: ' + name2)

                    param_int =  (param.to(torch.float32) * (2 ** 31)).round().to(torch.int64)
                    embed_state_dict[name2] = param_int

                    os.makedirs(EmbedsZKDir, exist_ok=True)
                    fileCount = param_int.shape[0] // EmbedsInOneFile
                    for i in range(0, fileCount):
                        saveTensor(EmbedsZKDir + str(i) + '.bin', param_int[i * EmbedsInOneFile : (i+1) * EmbedsInOneFile].cpu())
                elif comp == 'layers':
                    layer_id = int(ns[1])
                    name2 = '.'.join(ns[2:])
                    layer_state_dict0[layer_id][name2] = param

    print('Finish loading state dict from disk! ++++++++++')

    # for layer_id, states in enumerate(layer_state_dict0):
    for layer_id in range(len(layer_state_dict0)):
        os.makedirs(f'{save_path}/experts-{layer_id}', exist_ok=True)

        states = layer_state_dict0[layer_id]

        for name, param in states.items():
            ns = name.split(".")
            typ = param.type()
            shape = param.shape

            if ns[0] == 'attn_norm':
                print(f'layer {layer_id} {name}, type: {typ}', flush=True)
                if ns[1] == 'weight':
                    param_int = (param.to(torch.float32) * (2 ** 21)).round().to(torch.int32)
                    layer_state_dict[layer_id][name] = param_int
            elif ns[0] == 'ffn_norm':
                print(f'layer {layer_id} {name}, type: {typ}', flush=True)
                if ns[1] == 'weight':
                    param_int2 = (param.to(torch.float32) * (2 ** 23)).round().to(torch.int32)
                    layer_state_dict[layer_id][name] = param_int2
            elif ns[0] == 'ffn':
                if len(ns) == 3:
                    if ns[1] == 'w1' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'w1' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())
                        scale = w1_rescales[layer_id]
                        rescale = 2 ** scale
                        param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                        layer_state_dict[layer_id][weight_name] = param_int.cpu()
                        layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)

                        print(f'layer {layer_id} w1 weight, type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {name}', flush=True)
                    elif ns[1] == 'w2' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'w2' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())
                        scale = w2_rescales[layer_id]
                        rescale = 2 ** scale
                        param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                        layer_state_dict[layer_id][weight_name] = param_int.cpu()
                        layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)

                        print(f'layer {layer_id} w2 weight, type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {name}', flush=True)
                    elif ns[1] == 'w3' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'w3' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())
                        scale = w3_rescales[layer_id]
                        rescale = 2 ** scale
                        param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                        layer_state_dict[layer_id][weight_name] = param_int.cpu()
                        layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)

                        print(f'layer {layer_id} w3 weight, type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {name}', flush=True)

                    elif ns[1] == 'gate' and ns[2] == 'weight':
                        gate_rescale = 2 ** gate_rescales[layer_id]
                        gate_int = (param.to(torch.float32) * gate_rescale).round().to(torch.int32)
                        layer_state_dict[layer_id][name] = gate_int.cpu()
                        rescale_name = name.replace('weight', 'scale')
                        layer_state_dict[layer_id][rescale_name] = torch.tensor(gate_rescales[layer_id], dtype=torch.int32)
                        print(f'layer {layer_id}: gate_weight_name: {name}, gate_scale_name: {rescale_name}')
                    elif ns[1] == 'gate' and ns[2] == 'bias':
                        bias_int = (param.to(torch.float32) * (2 ** 23)).round().to(torch.int32)
                        layer_state_dict[layer_id][name] = bias_int.cpu()
                        print(f'layer {layer_id} bias: {name}')
                    else:
                        layer_state_dict[layer_id][name] = param
                elif len(ns) == 4:
                    if ns[1] == 'shared_experts':
                        if (ns[2] == 'w1' or ns[2] == 'w2' or ns[2] == 'w3') and ns[3] == 'scale':
                            continue
                        elif ns[2] == 'w1' and ns[3] == 'weight':
                            param_weight = param.cuda()
                            weight_name = name

                            scale_name = name.replace('weight', 'scale')
                            param_scale = states[scale_name]

                            weight = weight_dequant(param_weight, param_scale.cuda())
                            scale = shared_w1_rescales[layer_id]
                            rescale = 2 ** scale
                            param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                            layer_state_dict[layer_id][weight_name] = param_int.cpu()
                            layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)
                            print(f'layer {layer_id} shared_expert w1 type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {scale_name}')
                        elif ns[2] == 'w2' and ns[3] == 'weight':
                            param_weight = param.cuda()
                            weight_name = name

                            scale_name = name.replace('weight', 'scale')
                            param_scale = states[scale_name]

                            weight = weight_dequant(param_weight, param_scale.cuda())
                            scale = shared_w2_rescales[layer_id]
                            rescale = 2 ** scale
                            param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                            layer_state_dict[layer_id][weight_name] = param_int.cpu()
                            layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)
                            print(f'layer {layer_id} shared_expert w2 type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {scale_name}')
                        elif ns[2] == 'w3' and ns[3] == 'weight':
                            param_weight = param.cuda()
                            weight_name = name

                            scale_name = name.replace('weight', 'scale')
                            param_scale = states[scale_name]

                            weight = weight_dequant(param_weight, param_scale.cuda())
                            scale = shared_w3_rescales[layer_id]
                            rescale = 2 ** scale
                            param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                            layer_state_dict[layer_id][weight_name] = param_int.cpu()
                            layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)
                            print(f'layer {layer_id} shared_expert w3 type: {typ}, shape: {shape}, weight_name: {weight_name}, scale_name: {scale_name}')
                        else:
                            layer_state_dict[layer_id][name] = param
                    else:
                        layer_state_dict[layer_id][name] = param
                elif len(ns) == 5:
                    if ns[1] == 'experts':
                        expert_id = int(ns[2])
                        if (ns[3] == 'w1' or ns[3] == 'w2' or ns[3] == 'w3') and ns[4] == 'scale':
                            continue
                        elif ns[3] == 'w1' and ns[4] == 'weight':
                            scale = experts_w1_rescales[layer_id][expert_id]
                            handle_expert_w(layer_id, expert_id, 1, param, name, scale, typ, shape, f'{save_path}/experts-{layer_id}')
                        elif ns[3] == 'w2' and ns[4] == 'weight':
                            scale = experts_w2_rescales[layer_id][expert_id]
                            handle_expert_w(layer_id, expert_id, 2, param, name, scale, typ, shape, f'{save_path}/experts-{layer_id}')
                        elif ns[3] == 'w3' and ns[4] == 'weight':
                            scale = experts_w3_rescales[layer_id][expert_id]
                            handle_expert_w(layer_id, expert_id, 3, param, name, scale, typ, shape, f'{save_path}/experts-{layer_id}')
                        else:
                            layer_state_dict[layer_id][name] = param
                else:
                    layer_state_dict[layer_id][name] = param
            elif ns[0] == 'attn':
                if len(ns) == 3:
                    if ns[1] == 'wq_a' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'wq_a' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())

                        weight_int = (weight.to(torch.float32) * (2 ** 30)).round().to(torch.int32)

                        layer_state_dict[layer_id][weight_name] = weight_int.cpu()

                        print(f'layer {layer_id} wq_a weight, type: {typ}, shape: {shape}', flush=True)
                    elif ns[1] == 'q_norm':
                        print(f'layer {layer_id} q_norm, type: {typ}, shape: {shape}', flush=True)

                        param_int3 = (param.to(torch.float32) * (2 ** 19)).round().to(torch.int32)
                        layer_state_dict[layer_id][name] = param_int3
                    elif ns[1] == 'kv_norm':
                        print(f'layer {layer_id} kv_norm, type: {typ}, shape: {shape}', flush=True)

                        param_int4 = (param.to(torch.float32) * (2 ** 23)).round().to(torch.int32)
                        layer_state_dict[layer_id][name] = param_int4
                    elif ns[1] == 'wq_b' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'wq_b' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())

                        weight_int = (weight.to(torch.float32) * (2 ** 30)).round().to(torch.int32)

                        weight_int = weight_int.view(128, 192, 1536)
                        wq_b1, wq_b2 = torch.split(weight_int, [128, 64], dim=-2)

                        print(f'layer {layer_id} wq_b1 weight, shape: {wq_b1.shape}, wq_b2 weight, shape: {wq_b2.shape}', flush=True)

                        wq_b1 = wq_b1.reshape(128 * 128, 1536)
                        wq_b2 = wq_b2.reshape(128 * 64, 1536)
                        wq_b1_name = weight_name.replace('wq_b', 'wq_b1')
                        wq_b2_name = weight_name.replace('wq_b', 'wq_b2')

                        # layer_state_dict[layer_id][weight_name] = weight_int.cpu()
                        layer_state_dict[layer_id][wq_b1_name] = wq_b1.cpu()
                        layer_state_dict[layer_id][wq_b2_name] = wq_b2.cpu()

                        print(f'layer {layer_id} wq_b weight, type: {typ}, shape: {shape}', flush=True)
                    elif ns[1] == 'wkv_a' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'wkv_a' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())

                        weight_int = (weight.to(torch.float32) * (2 ** 29)).round().to(torch.int32)

                        # layer_state_dict[layer_id][weight_name] = weight_int.cpu()

                        weight_int = weight_int.view(576, 7168)
                        wkv_a1, wkv_a2 = torch.split(weight_int, [512, 64], dim=-2)

                        print(f'layer {layer_id} wkv_a1 weight, shape: {wkv_a1.shape}, wkv_a2 weight, shape: {wkv_a2.shape}', flush=True)

                        wkv_a1_name = weight_name.replace('wkv_a', 'wkv_a1')
                        wkv_a2_name = weight_name.replace('wkv_a', 'wkv_a2')

                        # layer_state_dict[layer_id][weight_name] = weight_int.cpu()
                        layer_state_dict[layer_id][wkv_a1_name] = wkv_a1.cpu()
                        layer_state_dict[layer_id][wkv_a2_name] = wkv_a2.cpu()

                        print(f'layer {layer_id} wkv_a weight, type: {typ}, shape: {shape}', flush=True)
                    elif ns[1] == 'wkv_b' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'wkv_b' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())

                        wkv_b = weight.view(128, 256, 512)

                        wkv_b_1 = wkv_b[:, :128]
                        wkv_b_1 = wkv_b_1.reshape(128 * 128, 512)
                        scale1 = wkv_b_1_rescales[layer_id]
                        wkv_b_1_rescale = 2 ** scale1
                        wkv_b_1_int = torch.round(wkv_b_1.to(torch.float32) * wkv_b_1_rescale).to(torch.int32)

                        wkv_b_2 = wkv_b[:, -128:]
                        wkv_b_2 = wkv_b_2.reshape(128 * 128, 512)
                        scale2 = wkv_b_2_rescales[layer_id]
                        wkv_b_2_rescale = 2 ** scale2
                        wkv_b_2_int = torch.round(wkv_b_2.to(torch.float32) * wkv_b_2_rescale).to(torch.int32)

                        wkv_b_1_name = weight_name.replace("wkv_b", "wkv_b_1")
                        wkv_b_1_scale_name = scale_name.replace("wkv_b", "wkv_b_1")
                        layer_state_dict[layer_id][wkv_b_1_name] = wkv_b_1_int.cpu()
                        layer_state_dict[layer_id][wkv_b_1_scale_name] = torch.tensor(scale1, dtype=torch.int32) 

                        wkv_b_2_name = weight_name.replace("wkv_b", "wkv_b_2")
                        wkv_b_2_scale_name = scale_name.replace("wkv_b", "wkv_b_2")
                        layer_state_dict[layer_id][wkv_b_2_name] = wkv_b_2_int.cpu()
                        layer_state_dict[layer_id][wkv_b_2_scale_name] = torch.tensor(scale2, dtype=torch.int32)

                        print(f'layer {layer_id} wkv_b, type: {typ}, shape: {shape}, wkv_b_1 weight: {wkv_b_1_name}, wkv_b_1 scale: {wkv_b_1_scale_name}, wkv_b_2 weight: {wkv_b_2_name}, wkv_b_2 scale: {wkv_b_2_scale_name}', flush=True)
                    elif ns[1] == 'wo' and ns[2] == 'scale':
                        continue
                    elif ns[1] == 'wo' and ns[2] == 'weight':
                        param_weight = param.cuda()
                        weight_name = name

                        scale_name = name.replace('weight', 'scale')
                        param_scale = states[scale_name]

                        weight = weight_dequant(param_weight, param_scale.cuda())

                        scale = wo_rescales[layer_id]
                        rescale = 2 ** scale

                        if layer_id != 58:
                            param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                        else:
                            wo_abs = weight.abs().cpu()
                            maxpos = wo_abs.argmax()
                            row, col = divmod(maxpos.item(), weight.size(1))
                            print(f'maxpos: {maxpos}, {row} {col}', flush=True)

                            vstr = getBF16PrintStr(weight[row][col])
                            print(f'weight[{row}][{col}]: {vstr}', flush=True)
                            weight[row][col] = 0
                            param_int = (weight.to(torch.float32) * rescale).round().to(torch.int32)
                            param_int[row][col] = -(2 ** 31)

                        layer_state_dict[layer_id][weight_name] = param_int.cpu()
                        layer_state_dict[layer_id][scale_name] = torch.tensor(scale, dtype=torch.int32)

                        print(f'layer {layer_id} wo weight, type: {typ}, shape: {shape}, weight: {weight_name}, scale: {scale_name}', flush=True)
                    else:
                        layer_state_dict[layer_id][name] = param
                else:
                    layer_state_dict[layer_id][name] = param
            else:
                layer_state_dict[layer_id][name] = param

        save_file(layer_state_dict[layer_id], os.path.join(save_path, f"layer-{layer_id}.safetensors"))
        print(f'Finish saving layer {layer_id}', flush=True)
        layer_state_dict0[layer_id] = {}
        layer_state_dict[layer_id] = {}

    print('Finish opening')

    os.makedirs(save_path, exist_ok=True)

    print(layer_state_dict)
    print(experts)

    save_file(head_state_dict, os.path.join(save_path, f"head_int.safetensors"))
    save_file(norm_state_dict, os.path.join(save_path, f"norm_int.safetensors"))
    save_file(embed_state_dict, os.path.join(save_path, f"embed_int.safetensors"))
    # for i, st in enumerate(layer_state_dict):
    #     # print(f'{i} {st['attn_norm.weight']}', flush=True)
    #     # print(f'{i} {st['ffn_norm.weight']}', flush=True)
    #     save_file(st, os.path.join(save_path, f"layer-{i}.safetensors"))
    #     print(f'Finish saving layer {i}', flush=True)

    # for i in trange(mp):
    #     save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    # print('Finish saving files')

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)
