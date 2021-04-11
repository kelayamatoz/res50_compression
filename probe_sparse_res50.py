import torchvision
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import time
import numpy as np
from models import resnet, resnet50
from functools import reduce
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iter', type=int)
args = parser.parse_args()
iteration = args.iter

THRESH = 0.01
DEBUG = True
S2MS = 1000
ctr = 0
# data_dir = '../data/sparse_res_conv'
data_dir = 'sparse_res_conv'

def _get_sparsity(t: torch.Tensor) -> float:
    return t.to_sparse()._nnz() / reduce(lambda x, y: x * y, list(t.shape))

def save_capstan_format(
    data_dir: str,
    prefix: str,
    in_tensor: torch.Tensor,
    weight: torch.Tensor,
    out_tensor: torch.Tensor,
) -> None:
    f_names = ["dat", "kern", "gold"]
    tensors = [in_tensor, weight, out_tensor]
    in_np, kern_np, gold = [t.detach().numpy() for t in tensors]
    # Input is of shape
    sparsity = _get_sparsity(in_tensor)
    in_cap = in_np.squeeze()
    kern = kern_np.transpose((1, 0, 2, 3))
    isl, osl, kdim, _ = kern.shape
    isl_img, dim, _ = in_cap.shape
    assert isl_img == isl, "in_cap and kern's isl don't match!"
    ver_str = "d{}_k{}_i{}_o{}_s{}".format(dim, kdim, isl, osl, str(sparsity)[2:4])
    print(ver_str)
    return
    d = data_dir + "/" + "{}_{}".format(ver_str, prefix) + "/"
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print("OSError occurred.")
        raise e

    pad_len = int((kdim - 1) / 2)
    dat_pad = np.pad(
        in_cap,
        pad_width=[(0, 0), (pad_len, pad_len), (pad_len, pad_len)],
        mode="constant",
    )
    st_np_list = [dat_pad, kern, gold]
    for f, tensor_np in zip(f_names, st_np_list):
        np.savetxt(d + f, tensor_np.flatten(), delimiter="\n")
    with open(d + "conf.yaml", "w") as f:
        f.write(f"dataDim: {dim}\n")
        f.write(f"kernelDim: {kdim}\n")
        f.write(f"inSlice: {isl}\n")
        f.write(f"outSlice: {osl}\n")

        f.write(f"dataFile: {d}/dat\n")
        f.write(f"kernelFile: {d}/kern\n")
        f.write(f"goldFile: {d}/{ver_str}/gold\n")
        f.write(f"debug: 1\n")


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Probing using device {}".format(device))

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((224, 224))
    ]
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2
)

test_img = testset[0][0].unsqueeze(0)
model_path = "./checkpoint/pruned-res50-ckpt.t7"
state_dict = (
    torch.load(model_path)
    if device == "gpu"
    else torch.load(model_path, map_location=torch.device("cpu"))
)
model_state_dict = {}
for name, module in state_dict.items():
    fields = name.split(".")
    if "mask" in name:
        continue
    if "weight_orig" in name:
        orig_weight = module
        mask = state_dict[".".join(fields[:-1] + ["weight_mask"])]
        m = orig_weight * mask
        k = ".".join(fields[1:-1] + ["weight"])
    else:
        m = module
        k = ".".join(fields[1:])
    model_state_dict[k] = m
model = resnet50()

conv_runtime_dict = {}
orig_pointer = nn.Module.__call__


def _get_sparsity(m):
    return float(torch.sum(m == 0)) / float(m.numel())


def _instrument(*args, **kwargs):
    global ctr
    # print("instrumented! len(args) = {}".format(len(args)))
    if len(args) != 2:
        raise ValueError("Not sure what to do with args != 2...")
    begin = time.time()
    _r = orig_pointer(*args, **kwargs)
    end = time.time()
    if isinstance(args[0], nn.Conv2d):
        conv_runtime_dict[ctr] = (end - begin) * 1000
        prefix = "conv_{}".format(ctr)
        ctr += 1
        # print(_get_sparsity(args[0].weight))
        # save_capstan_format(data_dir, prefix, args[1], args[0].weight, _r)

    return _r


nn.Module.__call__ = _instrument

model.load_state_dict(model_state_dict)
if device == "cuda":
    print("running cuda!")
    model = model.to("cuda")
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
else:
    torch.set_num_threads(128)
out = model(test_img)
print(torch.argmax(out, dim=0))
f_name = "./{}_{}.csv".format("cuda" if device == "cuda" else "cpu", iteration)
with open(f_name, "w+") as f:
    lines = [
        "{}, {}\n".format(t[0], "{:.2f}".format(t[1]))
        for t in conv_runtime_dict.items()
    ]
    lines.insert(0, "conv_idx, latency (ms)\n")
    f.writelines(lines)
