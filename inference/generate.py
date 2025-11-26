import os
import time
import json
from argparse import ArgumentParser
from typing import List
from torch import nn

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_file, load_model

from model import Transformer, ModelArgs, Block
from concurrent.futures import ThreadPoolExecutor
from kernel import softmax_q21, softmax_q19

snark = True
zkDataDir = '../zkdata'

model = None
kv_caches = [ torch.zeros(1, 4096 * 4, 512, dtype=torch.int64) ] * 61
pe_caches = [ torch.zeros(1, 4096 * 4, 64, dtype=torch.int64) ] * 61
state_dicts = [None] * 61

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

def mem(i):
    a = torch.cuda.memory_allocated()/1024**2
    r = torch.cuda.memory_reserved()/1024**2
    m = torch.cuda.max_memory_allocated()/1024**2
    print(f"{i} allocated={a:.1f}MB, reserved={r:.1f}MB, max={m:.1f}MB", flush=True)

def load_model2(ckpt_path):
    global model

    with torch.device("cuda"):
        load_model(model.embed, os.path.join(ckpt_path, f"embed_int.safetensors"))
        load_model(model.norm, os.path.join(ckpt_path, f"norm_int.safetensors"))
        load_model(model.head, os.path.join(ckpt_path, f"head_int.safetensors"))

# logits 的 scale = 2^21
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    # logits = logits.to(torch.float32) * (2 ** -15)
    # typ = logits.type()
    # print(f'sample logits type: {typ}')
    # logits = logits / max(temperature, 1e-5)
    # probs = torch.softmax(logits, dim=-1)
    # return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

    sample_open = False

    if sample_open:
        maxx = logits.abs().max()
        typ = logits.type()
        print(f'sample logits type: {typ}, shape: {logits.shape}, abs max: {maxx}')
        if temperature > 1e-5:
            temp_int = int(temperature)
            # logits = (logits + temp_int // 2) // temp_int
            logits = logits // temp_int
            print(f'temp_int: {temp_int}', flush=True)
        else:
            logits = logits * (10 ** 5)
        # print(f'sample 22 logits type: {typ}, shape: {logits.shape}, logits: {logits}')
        # probs = torch.softmax(logits, dim=-1)

        logits = logits.unsqueeze(2)

        max0 = logits.abs().max()
        print(f'sample 2233 logits shape: {logits.shape}, abs max0: {max0}')

        # probs 的 rescale 为 2^21
        probs = torch.empty_like(logits, dtype=torch.int64, device='cuda')
        softmax_q21(logits.contiguous(), probs)

        probs = probs.squeeze(2)

        # print(f'sample 2233 probs shape: {probs.shape}')

        typ2 = probs.type()
        max1 = probs.abs().max()
        print(f'sample 33 probs type: {typ2}, shape: {probs.shape}, probs abs max: {max1}', flush=True)

        rand = torch.empty_like(probs, dtype=torch.float32, device='cuda').exponential_(1)
        rand_abs = rand.abs()
        rmin = getF32PrintStr(rand_abs.min())
        rmax = getF32PrintStr(rand_abs.max())
        print(f'sample 333 rand abs min: {rmin}, max: {rmax}', flush=True)

        # rand = (rand * (2 ** 21)).round().to(torch.int64) + (2 ** 15)
        rand = (rand * (2 ** 10)).round().to(torch.int64) + (2 ** 4)
        max2 = rand.abs().max()
        min2 = rand.abs().min()
        print(f'sample 55 rand abs min: {min2}, max: {max2}', flush=True)

        # probs 的 rescale 为 2^21
        # probs = (probs * (2 ** 21) + rand // 2) // rand
        probs = (probs * (2 ** 10)) // rand

        max3 = probs.abs().max()
        print(f'sample 66 probs abs max: {max3}', flush=True)

        res = probs.argmax(dim=-1)
        tid = res[0][0].item()
        tv = probs[0][0][tid]
        randv = rand[0][0][tid]
        # typ3 = res.type()
        print(f'sample 44 res: {res}, tid: {tid}, tv: {tv}, randv: {randv}')
    else:
        probs = logits.unsqueeze(2)
        max3 = probs.abs().max()
        print(f'sample 66 probs abs max: {max3}', flush=True)

        res = probs.argmax(dim=-1)
    return res

def saveTensor(fileName, t):
    with open(fileName, "w", encoding="utf-8") as f:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        with open(fileName, "wb") as f:
            # .numpy() -> bytes（C-order）
            f.write(t.numpy().tobytes(order="C"))

# model：用来输出最终结果token的模型。这里导入的是deepseek的模型架构。
# prompt_tokens： 即前文中的prompt_tokens, shape为 (batch_size, total_len, 7168)
# max_new_tokens：允许生成的最大的tokens的数量。生成过程会在这个数量或遇到终止标识符 (eos_id) 时停止。
# eos_id：<end▁of▁sentence>这个token对应的ID。当生成结果中遇到这个 token 时，该序列的生成会停止。
# temperature：采样温度。温度值控制生成时的随机性：温度越高，采样的随机性越大；当温度为 0 时，使用贪心策略（即选取概率最高的 token）。
# prompt的输入是List[List[int]]，外面的那个List是batch，里面的这个List是seq。等效于输入进去的就是已经tokenize好了的batch个的prompt。在我们这个“Who are you?”的示例中，batch = 1

@torch.inference_mode()
def generate(
    # model: Transformer,
    ckpt_path: str,
    args: ModelArgs,
    tokenizer: AutoTokenizer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """

    global model, layers
    global kv_caches, pe_caches

    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= args.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={args.max_seq_len})"
    total_len = min(args.max_seq_len, max_new_tokens + max(prompt_lens))
    # 利用 torch.full 初始化一个形状为 (batch_size, total_len) 的张量，所有值初始为 -1。这里 -1 作为“未填充token”的标志
    # torch.long 64 位 bit
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda")
    # 遍历每个 prompt，将其 token 填入对应行的前面部分。这样，张量中前面部分对应的是已知的 prompt，后面部分为待生成的 token 空间。
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    beginstr = tokenizer.decode(tokens[0][0:prompt_lens[0]], skip_special_tokens=True)
    # torch.cuda.synchronize()
    print(' ++++++ token:', beginstr, flush=True)

    prev_pos = 0
    # finished则是一个布尔张量，标记每个序列是否已经完成生成。初始时假设所有序列均未完成（False）
    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    # prompt_mask则生成一个掩码张量，用来标记哪些位置已经有prompt token（即 token 不等于 -1）。在生成过程中，这个掩码帮助区分哪些位置是用户提供的prompt，哪些是模型生成的token。
    # 这是用来辅助自回归的生成的，避免prompt_tokens的部分被覆盖。
    prompt_mask = tokens != -1

    # cur_pos则记录prompt_tokens里最短的那段prompt的长度，后续的生成就从这个位置开始，以确保所有的输入都能得到生成正确而完整的回答。
    for cur_pos in range(min(prompt_lens), total_len):
        print(f'prev_pos: {prev_pos}, cur_pos: {cur_pos}, total_len: {total_len}', flush=True)
        t = tokenizer.decode(tokens[0][prev_pos:cur_pos], skip_special_tokens=True)
        print(str(cur_pos) + ' ---------- token list: ' + str(tokens[0][prev_pos:cur_pos].tolist()), flush=True)

        if snark:
            os.makedirs(f'{zkDataDir}/pos_{prev_pos}', exist_ok=True)
            saveTensor(f'{zkDataDir}/pos_{prev_pos}/tokens.bin', tokens[0][prev_pos:cur_pos].cpu())

        # logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

        h, start_pos, seqlen = model.prep_inference(tokens[:, prev_pos:cur_pos], prev_pos)
        print('h 1 shape: ' + str(h.shape), flush=True)

        for i in range(args.n_layers):
            print(f'begin layer {i} -----------------', flush=True)
            with torch.device("cuda"):

                with torch.no_grad():
                    if hasattr(model.layers[i], 'attn_norm'):
                        del model.layers[i].attn_norm.weight

                model.layers[i] = Block(i, args, ckpt_path)
                model.layers[i].load_state_dict(state_dicts[i], False)
                model.layers[i].attn.kv_cache = kv_caches[i].to('cuda')
                model.layers[i].attn.pe_cache = pe_caches[i].to('cuda')

            h = model.layer_inference(i, h, start_pos, seqlen)

            kv_caches[i] = model.layers[i].attn.kv_cache
            pe_caches[i] = model.layers[i].attn.pe_cache
            model.layers[i] = nn.Module()

            tmph = model.norm(h)[0][:, -1]

            tmph_abs = tmph.abs()
            tmph_min = tmph_abs.min()
            tmph_max = tmph_abs.max()
            print(f'tmph_abs min: {tmph_min}, max: {tmph_max}', flush=True)

            tmplogits = model.head(tmph[None, :])

            tmp_next_token = tmplogits.argmax(dim=-1)
            tid = tmp_next_token[0][0].item()
            tmp_logit = tmplogits[0][0][tid]
            tmp_completion = tokenizer.decode([tmp_next_token[0][0]], skip_special_tokens=True)
            print(f'position {cur_pos} tid: {tid}, tmp_logit:{tmp_logit}, candidate: {tmp_completion}', flush=True)

        # logits 的 scale = 2^21
        logits = model.finish_inference(h)

        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        # print('next_token shape: ' + str(next_token.shape))
        tokens[:, cur_pos] = next_token
        # 当所有finished里对应每一行的值都变成true的时候就意味着生成结束了。之后再进行decode，就得到了最终的输出。
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token.view(-1) == eos_id)
        prev_pos = cur_pos

        completion = tokenizer.decode(tokens[0][0:cur_pos+1], skip_special_tokens=True)
        print(f'----------  Result: position {cur_pos}, token: {completion}', flush=True)

        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    global model

    # WORLD_SIZE描述了全局进程总数（即参与训练的 GPU 总数）
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    # RANK则是当前进程的全局编号（即多机多卡上的进程编号，范围是[0,world_size-1]）
    rank = int(os.getenv("RANK", "0"))
    # LOCAL_RANK则是当前节点（机器）上的进程编号（即目前机器上的编号）
    # local_rank = int(os.getenv("LOCAL_RANK", "0"))
    print('WORLD_SIZE: ' + str(world_size) + ', rank: ' + str(rank))
    # 当world_size>1时，则表示当前是多机多卡训练，就需要初始化分布式进程组了。这个时候就使用NCCL后端来初始化分布式训练。
    # 这里初始化了进程组，因此在后续的加载参数中，每个进程将通过仅加载属于自己进程部分的模型参数来全量加载模型。
    # NCCL（NVIDIA Collective Communications Library）是 NVIDIA 提供的一个用于高效多 GPU 和多节点通信的库。
    # 它专为深度学习和高性能计算（HPC）设计，能够显著加速分布式训练和多 GPU 计算任务。
    # if world_size > 1:
        # dist.init_process_group("nccl")
    # global print
    # 屏蔽非主进程的print函数，防止多个进程同时打印日志，保持输出整洁
    # if rank != 0:
        # print = lambda *_, **__: None
    # 设定GPU设备，让当前进程只使用local_rank对应的GPU：
    # torch.cuda.set_device(local_rank)
    torch.cuda.set_device(0)
    # 设置pytorch计算时的默认数据类型。这里使用的是BF16
    torch.set_default_dtype(torch.bfloat16)
    # 限制 PyTorch 在 CPU 计算时最多使用 8 个线程，防止过多线程竞争资源：
    torch.set_num_threads(8)
    # 设定随机种子，保证不同进程初始化时随机数相同。
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    # 首先根据deepseek给定的tokenizer.json加载了tokenizer，然后通过load_model加载了参数。注意：一般来讲，load_model是只能加载单一的safetensors的。
    # 但由于之前我们通过dist.init_process_group("nccl")完成了进程组的初始化，因此这一行代码每个进程都会执行。又因为确定好了rank ，
    # 进而使得每个进程只会加载属于自己那部分的模型。到此便完成了模型的全量加载。

    for i in range(args.n_layers):
        modelPath = os.path.join(ckpt_path, f"layer-{i}.safetensors")
        state_dicts[i] = load_file(modelPath, device="cpu")

    with torch.device("cuda"):
        model = Transformer(args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    load_model2(ckpt_path)

    # with torch.device("cuda"):
    #     freqs_cis_orig = precompute_freqs_cis(args)
    # load_model2(ckpt_path)

    # tokenizer.encode  将字符编码转换为 token, tokenizer.decode 转换为字符编码
    # generate 函数将一直生成下一个字符，直到遇到结束字符为止
    # tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 200, -1, 1.)[0])
    # cmp1 = tokenizer.decode(generate(ckpt_path, args, tokenizer, [tokenizer.encode("DeepSeek")], 2, -1, 1.)[0])
    # print(' ---------- DeepSeek result: ' + str(cmp1), flush=True)
    # print('begin to load model: ' + f"model{rank}-mp{world_size}.safetensors")

    if rank == 0:
        # !!! 这一块代码会导致显存泄露
        embed_abs = model.embed.weight.detach().cpu().abs()
        abs_min = torch.min(embed_abs)
        abs_max = torch.max(embed_abs)
        print('embed abs_min: ' + str(abs_min), flush=True)
        print('embed abs_max: ' + str(abs_max), flush=True)
    else:
        pass

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            # 当多机多卡（world_size>1）并且只有主进程（rank==0）接受用户的输入prompt，并通过dist.broadcast_object_list(objects,0)的方式广播给其他进程（rank!=0）。
            # 其他进程通过dist.broadcast_object_list(objects,0)接受主进程的prompt，并用于后续进入模型之中的输入。
            # 主进程在input()处会阻塞，而非主进程将在广播这一步阻塞。因此在接受到输入之后，可以保证所有进程接收到相同的prompt。
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            # 假设我们的prompt是“Hello,Who are you?”则其输入会整理成如下的chat template：
            #[
                #{
                    #"role":"user",
                    #"content":"Hello,Who are you?"
                #}
            #]
            messages.append({"role": "user", "content": prompt})
            # 而后经过tokenizer.apply_chat_template 将输入的chat template转化为模型训练时所使用的真正输入的token。
            # 可以看huggingface关于chat template的官方文档，这里面介绍得十分详细。在这里，我们只需要知道这个chat template转化为了模型输入的token即可。
            # tokenizer.apply_chat_template的tokenize参数是默认为正的。因此，经过了转化后的聊天模板将会变成int型的token形式。
            # 也就是说，上面的chat template 最终将变为List[int]，如[134,135,1617,...,124]等，之后作为input tokens输入到模型中。
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            # prompt_tokens = tokenizer(prompt, add_special_tokens=True)
            # 我们现在的prompt已经变成了prompt_tokens，并通过generate()变成了 输出的回答所对应的token（completion_tokens），
            # 而后再decode成为完整的回答后重新组成chat template并加入到历史的message中，则一个流程的问答就结束了。
            # completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            # with torch.no_grad():
            completion_tokens = generate(ckpt_path, args, tokenizer, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            # completion_tokens = generate(ckpt_path, args, tokenizer, [prompt_tokens['input_ids']], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
        assert len(prompts) <= args.max_batch_size, f"Number of prompts exceeds maximum batch size ({args.max_batch_size})"
        prompt_tokens = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True) for prompt in prompts]
        # completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completion_tokens = generate(ckpt_path, args, tokenizer, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory. 模型参数存放的路径。
        --config (str): Path to the model configuration file. 模型的超参配置文件的路径。
        --input-file (str, optional): File containing prompts for batch processing. 假设我们是批量输入prompt，则该参数是批量输入prompt的文件的路径。
        --interactive (bool, optional): Enable interactive mode for generating text. 是否是问答交互式？这里相当于开启模型的“问答”模式。bool变量。
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200. 限制要求生成的tokens的数量。
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2. 采样温度。

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
