"""S7_base__lr1e-3+kv2+xpred+muon+wide32+deep333+lr2e-3+exp4k5+tie_share+g4_k5__kid5.55

Run S7_base (group S7): built on the lab defaults by applying
'nothing (reference run)'; accumulated changes from the search: lr1e-3+kv2+xpred+muon+wide32+deep333+lr2e-3+exp4k5+tie_share+g4_k5.
Configuration differences from the lab file defaults: {"learning_rate": 0.002, "optimizer": "muon", "ema": 0.999, "widths": [136, 112, 112], "depths": [3, 3, 3], "expand": 4, "kernel": 5, "group_size": 4, "kv_heads": 2, "tie_decoder": true, "share_stage": true, "pred": "x", "nfe": 10, "t_end": 0.00875, "churn": 0.3}.
Training: 20000 steps, batch 64, seed 3801, 960819 parameters, 0 buffers.
Sampling: EMA 0.999, 10 NFE, uniform time grid ending at t = 0.00875, options {"churn": 0.3}, chosen as the
best of the post-training sampler sweep on the validation split.
Result: KID x 1000 = 5.550 on the validation split (finalized 2026-11-17 13:59).
Set STUDENT_ID, then run this file or paste it into one notebook cell to reproduce.
"""
import ast
import hashlib
import json
import math
import random
import re
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_fidelity import calculate_metrics
from torch.nn import functional as F

STUDENT_ID = "studentID"
DATA_DIR = Path("data")


CONFIG = dict(
    task='flow',
    steps=20000,
    batch_size=64,
    learning_rate=0.002,
    weight_decay=0.0,
    warmup=500,
    beta2=0.99,
    grad_clip=1.0,
    decay_frac=0.0,
    optimizer='muon',
    muon_lr=0.02,
    muon_momentum=0.95,
    ema_decays=(0.997, 0.998, 0.999, 0.9995, 0.9999),
    ema=0.999,
    flip_p=0.5,
    shift_p=0.5,
    shift_max=2,
    shift_pad='reflect',
    aug_cond=True,
    widths=(136, 112, 112),
    depths=(3, 3, 3),
    attn=(False, True, True),
    mid_depth=1,
    expand=4,
    kernel=5,
    group_size=4,
    emb=32,
    heads=4,
    kv_heads=2,
    head_dim=None,
    act='gelu',
    repeat=1,
    tie_decoder=True,
    share_stage=True,
    pred='x',
    x_tmin=0.05,
    t_freqs=2,
    mf_ratio=0.25,
    t_mean=-0.4,
    t_std=1.0,
    mf_power=1.0,
    mf_eps=0.001,
    nfe=10,
    t_end=0.00875,
    first_jump=0.0,
    sample_clamp=False,
    churn=0.3,
    end_noise=0.0,
    noise_temp=1.0,
    grid='uniform',
    grid_shift=1.0,
    edm_rho=7.0,
    edm_sigma_max=80.0,
    edm_sigma_min=0.002,
)


def fourier(t, n=8):
    n = CONFIG["t_freqs"]
    angles = t[:, None] * (2 ** torch.arange(n, device=t.device))[None] * math.pi
    return torch.cat([t[:, None], angles.sin(), angles.cos()], -1)


def channel_norm(x, eps=1e-6):

    mean = x.mean(1, keepdim=True)
    var = (x - mean).square().mean(1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps)


class Modulation(nn.Module):
    def __init__(self, emb, width):
        super().__init__()
        self.linear = nn.Linear(emb, 3 * width)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)

    def forward(self, e):
        return self.linear(e)[:, :, None, None].chunk(3, 1)


ACTIVATIONS = {"gelu": F.gelu, "silu": F.silu}


class ConvCore(nn.Module):
    def __init__(self, width, expand, kernel, group_size=1):
        super().__init__()
        self.dw = nn.Conv2d(width, width, kernel, padding=kernel // 2, groups=width // group_size)
        self.expand = nn.Conv2d(width, expand * width, 1)
        self.project = nn.Conv2d(expand * width, width, 1)


class Block(nn.Module):
    def __init__(self, width, emb, core):
        super().__init__()
        self.core, self.mod = core, Modulation(emb, width)
        self.act = ACTIVATIONS[CONFIG["act"]]

    def forward(self, x, e):
        scale, shift, gate = self.mod(e)
        h = channel_norm(self.core.dw(x)) * (1 + scale) + shift
        return x + gate * self.core.project(self.act(self.core.expand(h)))


class AttnCore(nn.Module):
    def __init__(self, width, heads, kv_heads, head_dim=None):
        super().__init__()
        self.heads, self.kv_heads, self.dim = heads, kv_heads, head_dim or width // heads
        self.q = nn.Conv2d(width, heads * self.dim, 1)
        self.kv = nn.Conv2d(width, 2 * kv_heads * self.dim, 1)
        self.out = nn.Conv2d(heads * self.dim, width, 1)


class Attention(nn.Module):
    def __init__(self, width, emb, core):
        super().__init__()
        self.core, self.mod = core, Modulation(emb, width)

    def forward(self, x, e):
        c_ = self.core
        b, c, hgt, wid = x.shape
        scale, shift, gate = self.mod(e)
        h = channel_norm(x) * (1 + scale) + shift
        q = c_.q(h).reshape(b, c_.kv_heads, c_.heads // c_.kv_heads, c_.dim, -1)
        k, v = c_.kv(h).reshape(b, 2, c_.kv_heads, 1, c_.dim, -1).unbind(1)
        weights = torch.einsum("bgqdn,bgqdm->bgqnm", q, k) * c_.dim ** -0.5
        o = torch.einsum("bgqnm,bgqdm->bgqdn", weights.softmax(-1), v)
        return x + gate * c_.out(o.reshape(b, c_.heads * c_.dim, hgt, wid))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = CONFIG
        widths, emb = cfg["widths"], cfg["emb"]
        self.time = nn.Sequential(nn.Linear(2 * (2 * cfg["t_freqs"] + 1) + 2, emb), nn.SiLU(),
                                  nn.Linear(emb, emb))
        self.label = nn.Embedding(1, emb)
        self.input = nn.Conv2d(3, widths[0], 3, padding=1)

        def stage(width, depth, attn, cores=None):
            if cores is None:
                cores = ([ConvCore(width, cfg["expand"], cfg["kernel"], cfg["group_size"])
                          for _ in range(1 if cfg["share_stage"] else depth)],
                         AttnCore(width, cfg["heads"], cfg["kv_heads"], cfg["head_dim"]) if attn else None)
            layers = []
            for i in range(depth):
                for _ in range(cfg["repeat"]):
                    layers.append(Block(width, emb, cores[0][i % len(cores[0])]))
                if attn and i == depth // 2:
                    layers.append(Attention(width, emb, cores[1]))
            return nn.ModuleList(layers), cores

        n = len(widths)
        built = [stage(widths[i], cfg["depths"][i], cfg["attn"][i]) for i in range(n)]
        self.encoder = nn.ModuleList(layers for layers, _ in built)
        self.down = nn.ModuleList(nn.Sequential(
            nn.Conv2d(widths[i], widths[i], 3, 2, 1, groups=widths[i]),
            nn.Conv2d(widths[i], widths[i + 1], 1)) for i in range(n - 1))
        self.middle = stage(widths[-1], cfg["mid_depth"], cfg["attn"][-1])[0]
        self.up = nn.ModuleList(nn.Sequential(
            nn.Conv2d(widths[i + 1], widths[i], 1), nn.Upsample(scale_factor=2)) for i in range(n - 1))
        self.decoder = nn.ModuleList(
            stage(widths[i], cfg["depths"][i], cfg["attn"][i], built[i][1] if cfg["tie_decoder"] else None)[0]
            for i in range(n))
        self.out = nn.Conv2d(widths[0], 3, 3, padding=1)

    def forward(self, x, t, y=None, r=None, aug=None):


        r = t if r is None else r
        aug = torch.zeros(len(x), 2, device=x.device) if aug is None else aug
        e = self.time(torch.cat([fourier(t), fourier(t - r), aug], -1))
        if y is not None:
            e = e + self.label(y)
        h, skips = self.input(x), []
        for i, layers in enumerate(self.encoder):
            for layer in layers:
                h = layer(h, e)
            skips.append(h)
            if i < len(self.down):
                h = self.down[i](h)
        for layer in self.middle:
            h = layer(h, e)
        for i in reversed(range(len(self.decoder))):
            if i < len(self.up):
                h = self.up[i](h)
            h = h + skips[i]
            for layer in self.decoder[i]:
                h = layer(h, e)
        out = self.out(F.silu(channel_norm(h)))
        if CONFIG["pred"] == "x":
            out = (x - out) / t.clamp(min=CONFIG["x_tmin"])[:, None, None, None]
        return out


def build_model():
    return Model()

MAX_PARAMETERS, MAX_STEPS, MAX_BATCH = (1000000, 20000, 64)


def program_source():
    if "__file__" in globals():
        source = Path(__file__).read_text()
    else:
        source = get_ipython().history_manager.input_hist_raw[-1].rstrip() + "\n"
    tree = ast.parse(source)
    names = {getattr(node, "name", "") for node in tree.body}
    if not {"build_model", "run", "save_submission"} <= names:
        raise ValueError("Paste the complete program into one notebook cell and run that cell.")
    return source


def model_size(model):
    parameters = sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters())
    buffers = sum(b.numel()*(2 if b.is_complex() else 1) for b in model.buffers())
    if parameters + buffers > MAX_PARAMETERS:
        raise ValueError(f"Model has {parameters + buffers} values; limit is {MAX_PARAMETERS}.")
    return parameters, buffers


def save_submission(folder, source, history, metrics, size, seed, split):
    stem = "lab-06"
    code = folder / (stem + "-code.py")
    code.write_text(source)
    names = [f"batch-{i:02d}.png" for i in range(1, 11)]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    updates = {"model": CONFIG["steps"]}
    result = dict(format=6, lab=6, seed=seed, config=dict(CONFIG),
                  parameters=size[0], buffer_elements=size[1], updates=updates,
                  protocol="comp3801-cifar10-horses-kid1000-v1-" + split, code_sha256=digest(code),
                  metrics=metrics, loss=history,
                  samples_sha256={name: digest(folder/"samples"/name) for name in names})
    record = folder / (stem + "-results.json")
    record.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    archive = folder.with_name(folder.name + ".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as pack:
        for path in [code, record] + [folder/"samples"/name for name in names]:
            pack.write(path, path.relative_to(folder))
    print(json.dumps(metrics, indent=2))
    print("Submission:", archive.resolve())
    return result


def show_results(history, folder):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 2.5))
    curves = np.array(history)
    plt.plot(curves[:, 0], curves[:, 1])
    plt.xlabel("Update"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()
    from PIL import Image
    plt.figure(figsize=(6, 6))
    plt.imshow(Image.open(folder/"samples/batch-01.png"))
    plt.axis("off"); plt.tight_layout(); plt.show()


def image_data(split="train"):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=DATA_DIR, train=split != "test", download=True)
    index = torch.where(torch.tensor(dataset.targets) == 7)[0]
    if split != "test":
        order = torch.randperm(len(index), generator=torch.Generator().manual_seed(3802))
        index = index[order[:4000] if split == "train" else order[4000:]]
    images = torch.from_numpy(dataset.data).permute(0, 3, 1, 2)[index].float()/127.5 - 1
    return images, torch.zeros(len(images), dtype=torch.long)


def image_bytes(images):
    return ((images.clamp(-1, 1) + 1)*127.5 + .5).floor().to(torch.uint8)


class Images(torch.utils.data.Dataset):
    def __init__(self, pixels): self.pixels = pixels
    def __len__(self): return len(self.pixels)
    def __getitem__(self, index): return self.pixels[index]


@torch.no_grad()
def image_score(model, folder, device, split):
    from PIL import Image
    model.eval()
    rng = torch.Generator(device=device).manual_seed(921)
    fake = torch.cat([generate(model, 50, device, rng).clamp(-1, 1).cpu() for _ in range(20)])
    real, _ = image_data(split)
    pixels = image_bytes(fake)

    metrics = calculate_metrics(
        input1=Images(image_bytes(real)), input2=Images(pixels),
        cuda=torch.device(device).type == "cuda", batch_size=32,
        kid=True, kid_subset_size=1000, kid_subsets=1,
        feature_extractor="inception-v3-compat", feature_layer_kid="2048",
        rng_seed=921, save_cpu_ram=True, verbose=False)
    score = metrics["kernel_inception_distance_mean"]
    for i, batch in enumerate(pixels.split(100), 1):
        grid = batch.reshape(10, 10, 3, 32, 32).permute(0, 3, 1, 4, 2).reshape(320, 320, 3)
        Image.fromarray(grid.numpy()).save(folder/f"batch-{i:02d}.png")
    print(f"KID x 1000: {1000*score:.3f}")
    return {"kid": score, "samples": 1000}


def diffusion_schedule(steps=100, device='cpu'):

    grid = torch.linspace(0, 1, steps+1, device=device, dtype=torch.float64)
    cumulative = torch.cos((grid+.008)/1.008 * torch.pi/2).square()
    cumulative = cumulative/cumulative[0]
    betas = (1-cumulative[1:]/cumulative[:-1]).clamp(max=.999).float()
    alphas = 1-betas
    bars = torch.cumprod(alphas, dim=0)
    previous = torch.cat([torch.ones(1, device=device), bars[:-1]])
    variance = betas*(1-previous)/(1-bars)
    return betas, alphas, bars, variance

def add_noise(clean, epsilon, t, bars):
    a = bars[t].reshape(-1, 1, 1, 1)
    return a.sqrt()*clean + (1-a).sqrt()*epsilon

def diffusion_loss(model, clean, bars, labels=None):
    t = torch.randint(len(bars), (len(clean),), device=clean.device)
    epsilon = torch.randn_like(clean)
    noisy = add_noise(clean, epsilon, t, bars)

    prediction = model(noisy, (t.float()+1)/len(bars), labels) if labels is not None else model(noisy, (t.float()+1)/len(bars))
    return F.mse_loss(prediction, epsilon)

def flow_path(noise, data, t):
    t = t.reshape(-1, 1, 1, 1)
    return (1-t)*noise+t*data, data-noise

def flow_loss(model, data, labels):
    noise = torch.randn_like(data)
    t = torch.rand(len(data), device=data.device)
    x, target = flow_path(noise, data, t)
    return F.mse_loss(model(x, t, labels), target)

@torch.no_grad()
def sample_ddpm(model, count, schedule, device='cpu', generator=None, labels=None, shape=(3, 32, 32)):
    betas, alphas, bars, variance = schedule
    x = torch.randn(count, *shape, device=device, generator=generator)
    for i in reversed(range(len(betas))):
        t = torch.full((count,), (i+1)/len(betas), device=device)
        epsilon = model(x, t, labels) if labels is not None else model(x, t)

        clean = ((x-(1-bars[i]).sqrt()*epsilon)/bars[i].sqrt()).clamp(-1, 1)
        prev = bars[i-1] if i else torch.ones((), device=device)
        mean = (prev.sqrt()*betas[i]/(1-bars[i]))*clean
        mean = mean + (alphas[i].sqrt()*(1-prev)/(1-bars[i]))*x
        if i:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = mean + variance[i].sqrt()*noise
        else:
            x = mean
    return x.clamp(-1, 1)

@torch.no_grad()
def sample_flow(model, labels, steps=25, device='cpu', generator=None, shape=(3, 32, 32)):
    if steps < 1:
        raise ValueError('sampling steps must be positive')
    x = torch.randn(len(labels), *shape, device=device, generator=generator)
    dt = 1/steps
    for i in range(steps):
        t = torch.full((len(labels),), i/steps, device=device)
        x = x + dt*model(x, t, labels)
    return x.clamp(-1, 1)


LOG = None


def augment(x):
    b, s = len(x), CONFIG["shift_max"]
    flip = torch.rand(b, device=x.device) < CONFIG["flip_p"]
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    shift = torch.zeros(b, 2, device=x.device)
    if s > 0 and CONFIG["shift_p"] > 0:
        on = (torch.rand(b, device=x.device) < CONFIG["shift_p"])[:, None]
        d = torch.randint(-s, s + 1, (b, 2), device=x.device) * on
        mode = CONFIG["shift_pad"]
        padded = F.pad(x, (s, s, s, s), mode=mode) if mode != "zeros" else F.pad(x, (s, s, s, s))
        grid = torch.arange(x.shape[-1], device=x.device)
        rows = (grid[None] + s + d[:, 1:2])[:, :, None]
        cols = (grid[None] + s + d[:, 0:1])[:, None, :]
        x = padded[torch.arange(b, device=x.device)[:, None, None], :, rows, cols].permute(0, 3, 1, 2)
        shift = d.float() / s
    if not CONFIG["aug_cond"]:
        shift = torch.zeros_like(shift)
    return x, shift


def sample_times(b, device):
    t = torch.sigmoid(CONFIG["t_mean"] + CONFIG["t_std"] * torch.randn(b, device=device))
    r = torch.sigmoid(CONFIG["t_mean"] + CONFIG["t_std"] * torch.randn(b, device=device))
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    keep = torch.rand(b, device=device) < CONFIG["mf_ratio"]
    return t, torch.where(keep, r, t)


def meanflow_loss(model, data, labels, aug):
    noise = torch.randn_like(data)
    t, r = sample_times(len(data), data.device)
    tt = t[:, None, None, None]
    z = (1 - tt) * data + tt * noise
    v = noise - data
    if CONFIG["mf_ratio"] > 0:
        fn = lambda z_, r_, t_: model(z_, t_, labels, r_, aug)
        u, dudt = torch.func.jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        target = (v - (tt - r[:, None, None, None]) * dudt).detach()
    else:
        u, target = model(z, t, labels, r, aug), v
    error = (u - target).square().mean((1, 2, 3))
    weight = (error.detach() + CONFIG["mf_eps"]) ** -CONFIG["mf_power"]
    stats = dict(error=error.detach(), weight=weight, t=t, r=r)
    return (weight * error).mean(), stats


def newton_schulz(g, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                buf = self.state[p].setdefault("momentum", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]).reshape(len(p), -1)
                update = newton_schulz(g) * max(1.0, g.shape[0] / g.shape[1]) ** 0.5
                p.add_(update.reshape(p.shape), alpha=-group["lr"])


def make_optimisers(model):
    adam_kwargs = dict(lr=CONFIG["learning_rate"], betas=(0.9, CONFIG["beta2"]),
                       weight_decay=CONFIG["weight_decay"])
    if CONFIG["optimizer"] == "adamw":
        return [torch.optim.AdamW(model.parameters(), **adam_kwargs)]

    dense = {id(m.weight) for name, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and m.groups == 1 and name not in ("input", "out")}
    hidden = [p for p in model.parameters() if id(p) in dense]
    rest = [p for p in model.parameters() if id(p) not in dense]
    return [Muon(hidden, lr=CONFIG["muon_lr"], momentum=CONFIG["muon_momentum"]),
            torch.optim.AdamW(rest, **adam_kwargs)]


def lr_factor(step):
    factor = min(1.0, (step + 1) / max(1, CONFIG["warmup"]))
    decay_start = CONFIG["steps"] * (1 - CONFIG["decay_frac"])
    if CONFIG["decay_frac"] > 0 and step >= decay_start:
        factor *= max(0.0, (CONFIG["steps"] - step) / (CONFIG["steps"] - decay_start))
    return factor


def train(model, device):
    images, labels = image_data()
    images, labels = images.to(device), labels.to(device)
    optimisers = make_optimisers(model)
    schedulers = [torch.optim.lr_scheduler.LambdaLR(o, lr_factor) for o in optimisers]
    params = [p for p in model.parameters()]
    emas = {d: [p.detach().clone() for p in params] for d in CONFIG["ema_decays"]}
    history = []
    model.train()
    for step in range(1, CONFIG["steps"] + 1):
        index = torch.randint(len(images), (CONFIG["batch_size"],), device=device)
        x, aug = augment(images[index])
        loss, stats = meanflow_loss(model, x, labels[index], aug)
        mse = stats["error"].mean().item()
        for o in optimisers:
            o.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        for o, sch in zip(optimisers, schedulers):
            o.step()
            sch.step()
        if LOG is not None:
            LOG(step, dict(model=model, emas=emas, loss=loss.detach(), grad_norm=grad_norm,
                           lr=schedulers[-1].get_last_lr()[0], aug=aug, **stats))
        with torch.no_grad():
            for d, ema in emas.items():
                torch._foreach_lerp_(ema, params, 1 - min(d, (1 + step) / (10 + step)))
        if step == 1 or step % max(1, CONFIG["steps"]//40) == 0 or step == CONFIG["steps"]:
            history.append([step, round(mse, 6)])
        if step == 1 or step % max(1, CONFIG["steps"]//10) == 0 or step == CONFIG["steps"]:
            print(f"{step} / {CONFIG['steps']}: loss {mse:.4f}", flush=True)
    if CONFIG.get("checkpoint"):
        torch.save({"config": dict(CONFIG), "raw": [p.detach().cpu() for p in params],
                    "emas": {d: [p.cpu() for p in e] for d, e in emas.items()}}, CONFIG["checkpoint"])
    with torch.no_grad():
        for p, q in zip(params, emas[CONFIG["ema"]]):
            p.copy_(q)
    return history


def time_grid(steps, device='cpu'):
    u = torch.linspace(0, 1, steps + 1, device=device, dtype=torch.float64)
    if CONFIG["grid"] == "uniform":
        times = 1 - u
    elif CONFIG["grid"] == "shift":
        t, s = 1 - u, CONFIG["grid_shift"]
        times = s * t / (1 + (s - 1) * t)
    elif CONFIG["grid"] == "edm":
        rho, hi, lo = CONFIG["edm_rho"], CONFIG["edm_sigma_max"], CONFIG["edm_sigma_min"]
        sigma = (hi ** (1 / rho) + u[:-1] / u[-2] * (lo ** (1 / rho) - hi ** (1 / rho))) ** rho
        times = torch.cat([torch.ones(1, device=device, dtype=torch.float64),
                           (sigma / (1 + sigma))[1:], torch.zeros(1, device=device, dtype=torch.float64)])
    else:
        raise ValueError(f"unknown grid {CONFIG['grid']}")
    if CONFIG["first_jump"] > 0 and steps > 1:      # one MeanFlow jump 1 -> first_jump, then uniform steps
        rest = CONFIG["first_jump"] * (1 - torch.linspace(0, 1, steps, device=device, dtype=torch.float64))
        times = torch.cat([torch.ones(1, device=device, dtype=torch.float64), rest])
    # Stop at t_end instead of 0: the samples keep a trace of noise (~t_end), which the x-prediction models'
    # over-smooth outputs need (KID ~10 -> ~7.4 at t_end = 0.01 on S5_tie_dec).
    times = CONFIG["t_end"] + (1 - CONFIG["t_end"]) * times
    return times.float()


@torch.no_grad()
def sample_meanflow(model, labels, steps, device='cpu', generator=None, shape=(3, 32, 32)):
    noise = lambda: torch.randn(len(labels), *shape, device=device, generator=generator)
    x = CONFIG["noise_temp"] * noise()
    times = time_grid(steps, device)
    for i in range(steps):
        t0, t1 = times[i], times[i + 1]
        if CONFIG["churn"] > 0 and i > 0:           # re-noise to a slightly higher time before the step
            hi = min(1.0, float(t0) * (1 + CONFIG["churn"]))
            a = (1 - hi) / (1 - float(t0))
            x = a * x + max(hi ** 2 - (a * float(t0)) ** 2, 0.0) ** 0.5 * noise()
            t0 = torch.tensor(hi, device=device)
        t = t0.expand(len(labels))
        r = (t1 if CONFIG["mf_ratio"] > 0 else t0).expand(len(labels))
        u = model(x, t, labels, r)
        if CONFIG["sample_clamp"] and CONFIG["pred"] == "x":   # keep the clean-image estimate inside [-1, 1]
            s = t.clamp(min=CONFIG["x_tmin"])[:, None, None, None]
            u = (x - (x - s * u).clamp(-1, 1)) / s
        x = x - (t0 - t1) * u
    if CONFIG["end_noise"] > 0:                      # fresh noise on the final sample
        x = x + CONFIG["end_noise"] * noise()
    return x.clamp(-1, 1)


@torch.no_grad()
def generate(model, count, device, generator):
    labels = torch.zeros(count, dtype=torch.long, device=device)
    if CONFIG["task"] == "ddpm":
        return sample_ddpm(model, count, diffusion_schedule(device=device), device,
                           generator, labels, shape=(3, 32, 32))
    return sample_meanflow(model, labels, CONFIG["nfe"], device, generator)


def run(out=None, device=None, seed=3801, split="validation", show=True):
    source = program_source()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", STUDENT_ID):
        raise ValueError("Use letters, digits, underscore or hyphen for STUDENT_ID.")
    if CONFIG["task"] not in ("ddpm", "flow"):
        raise ValueError("Set task to 'ddpm' or 'flow'.")
    if not 1 <= CONFIG["steps"] <= MAX_STEPS or not 1 <= CONFIG["batch_size"] <= MAX_BATCH:
        raise ValueError("Training steps or batch size exceeds the practical limit.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    folder = Path(out or f"lab-06-{STUDENT_ID}")
    (folder/"samples").mkdir(parents=True, exist_ok=True)
    model = build_model().to(device)
    size = model_size(model)
    history = train(model, device)
    metrics = image_score(model, folder/"samples", device, split)
    result = save_submission(folder, source, history, metrics, size, seed, split)
    if show:
        show_results(history, folder)
    return result


if __name__ == "__main__":
    run()
