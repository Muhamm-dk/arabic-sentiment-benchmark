# Chapter 2: Environment

## What

Set up a reproducible GPU development environment on a Windows laptop: WSL2 with Ubuntu 24.04, the NVIDIA driver on the Windows side, a CUDA-enabled PyTorch build managed by uv, and a git repository on GitHub.

Machine: Lenovo LOQ 15IRX10 (i7-13650HX, RTX 5070 Laptop GPU with 8 GB, 32 GB RAM, 1 TB SSD), Windows 11.

## Why

**Why WSL2 and not native Windows?** The machine runs Windows, but nearly all ML tooling (shell scripts, Docker, `uv`, package builds) assumes Linux. WSL2 runs a real Linux kernel, and CUDA reaches the GPU through the Windows driver, so the same GPU is usable from Linux with no dual boot.

**Why the driver goes on Windows only.** Inside WSL you must not install an NVIDIA driver. WSL2 uses the Windows driver, and `nvidia-smi` inside Ubuntu reports it. You also do not need the CUDA toolkit: the PyTorch wheel bundles the CUDA runtime it needs.

**Why a new PyTorch build was mandatory.** The RTX 5070 is Blackwell (compute capability sm_120). An older PyTorch wheel can detect the GPU and then crash on the first real computation with "no kernel image is available". The wheel must be built for CUDA 12.8 or newer.

**Why uv and not conda.**
- `uv.lock` records exact versions of every package, so the environment can be rebuilt identically.
- Installs are fast.
- The same lock-and-sync flow maps onto a Docker build.
- The project is pure Python plus PyTorch wheels, so conda's strength (non-Python native dependencies) was not needed.

**Why work in the Linux filesystem.** Files under `/mnt/c` or `/mnt/d` cross a translation layer between Windows and Linux that is slow for many small reads (tokenization, dataset loading, package installs). The project lives in `~/projects`, on the Linux side. GitHub is the backup.

## Alternatives

| Option | Verdict |
|---|---|
| Native Windows with PyTorch | Possible. I did not test it. Shell scripts, Docker workflows and some tools are more awkward, and the Dockerfile later targets Linux anyway |
| Dual boot Linux | Full speed and no translation layer, but costs a reboot per switch. Overkill for this project |
| Conda or Miniconda | Works. Would have meant no lockfile by default and slower resolves. Better when you need non-Python native libraries |
| pip with venv | Works, but no lockfile and no dependency resolver comparable to uv |
| Poetry | Has a lockfile too. uv was chosen for speed and simplicity. I did not compare them in this project |
| Colab or Kaggle only | Free GPUs, but sessions are time-limited and not reproducible across runs. Kept only as overflow for anything over 8 GB, which in the end was not needed |

## How

### 1. WSL2 and Ubuntu (Windows, PowerShell as Administrator)

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
nvidia-smi
wsl --status
```

The Windows NVIDIA driver must be recent. `nvidia-smi` on Windows must show the RTX 5070 before anything else is attempted.

### 2. Cap WSL resources with `.wslconfig`

A file named exactly `.wslconfig` (dot, no extension) in `C:\Users\<name>\`:

```ini
[wsl2]
memory=22GB
processors=14
swap=8GB
```

- `memory=22GB` leaves about 10 GB to Windows. WSL2 by default takes a share of Windows RAM (I recall half, so 16 GB here, not verified), and Docker Desktop later uses the same allowance.
- `processors=14` matches the CPU's logical processors (the i7-13650HX has 20 threads, so 14 leaves headroom for Windows).
- `swap=8GB` is overflow when memory runs short.

Apply with `wsl --shutdown`, wait 10 seconds, reopen Ubuntu, then check:

```bash
free -h
nproc
```

Expected: about 21 GiB total memory, 8 GiB swap, 14 processors. (`free` shows slightly less than 22 GB because of overhead.)

### 3. Inside Ubuntu

```bash
sudo apt update && sudo apt install -y build-essential git curl
nvidia-smi
```

`nvidia-smi` must list the RTX 5070 here. If it does not, the fix is on the Windows driver side.

### 4. uv and the project

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
mkdir -p ~/projects && cd ~/projects
uv init arabic-sentiment-benchmark --python 3.12
cd arabic-sentiment-benchmark
uv add torch --index pytorch=https://download.pytorch.org/whl/cu128
uv add transformers datasets accelerate evaluate scikit-learn pandas numpy mlflow
uv add --dev ruff pytest jupyter
```

The install command for uv downloads Astral's official script and runs it. `-LsSf` means follow redirects, stay silent, still show errors, and fail on HTTP errors. uv installs into `~/.local/bin` with no sudo and does not touch the system Python. Version in this project: uv 0.13.1.

### 5. The final `pyproject.toml`

```toml
[project]
name = "arabic-sentiment-benchmark"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "accelerate>=1.15.0", "datasets>=5.1.0", "evaluate>=0.4.6", "fastapi>=0.143.3",
    "httpx>=0.28.1", "mlflow>=3.17.1", "numpy>=2.5.2", "pandas>=3.0.6", "pyarrow>=25.0.1",
    "scikit-learn>=1.9.1", "torch>=2.11.0", "transformers>=5.20.1", "uvicorn[standard]>=0.54.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/arabic_sentiment"]

[tool.uv.sources]
torch = { index = "pytorch" }

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"

[dependency-groups]
dev = ["jupyter>=1.1.1", "pytest>=9.1.1", "ruff>=0.17.1"]
```

Line by line:
- **`dependencies`**: runtime packages with minimum versions. Exact versions live in `uv.lock`, which is the file that makes the environment reproducible.
- **`[build-system]` and `[tool.hatch...]`**: makes `src/arabic_sentiment` an installable package, so `python -m arabic_sentiment.data` and `from arabic_sentiment.baseline import ...` work from anywhere in the project.
- **`[tool.uv.sources]` plus `[[tool.uv.index]]`**: tells uv to fetch `torch` from PyTorch's CUDA 12.8 index and not from PyPI (the PyPI wheel may not match the GPU).
- **`[dependency-groups] dev`**: tools that are not needed to run the code (lint, tests, notebooks).

Known weakness: `requests` (used by `llm_eval.py`) and `huggingface_hub` (used by several scripts) are imported directly but not declared. They arrive as dependencies of other packages, so they work today, but a future version of those packages could drop them. The fix is `uv add requests huggingface-hub`.

### 6. The CUDA check

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list()); x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16); print((x@x).float().mean().item()); print(torch.cuda.mem_get_info())"
```

What each part proves:

| Part | Proves |
|---|---|
| `torch.__version__, torch.version.cuda` | The wheel is a CUDA build and which version |
| `torch.cuda.is_available()` | The driver is visible. This alone is not enough |
| `torch.cuda.get_arch_list()` | The wheel contains kernels for this GPU. `sm_120` must appear |
| `torch.randn(4096,4096, device='cuda', dtype=torch.bfloat16)` then `x @ x` | Compiled kernels actually run on this GPU in bf16, the format used for training |
| `.float().mean().item()` | Brings one number back to the CPU, proving the result is readable |
| `torch.cuda.mem_get_info()` | Free and total VRAM in bytes |

### 7. Repo layout and `.gitignore`

```bash
mkdir -p src/arabic_sentiment data/{raw,processed} configs notebooks results tests docker
touch src/arabic_sentiment/__init__.py
```

The final `.gitignore`, and why each line exists:

```
data/raw/            datasets (licenses are not stated, Chapter 3)
data/processed/
.venv/               the virtual environment, rebuilt from uv.lock
__pycache__/
mlruns/              old MLflow folder format
wandb/               not used, kept in case
*.pt
*.safetensors        model weights (hundreds of MB)
outputs/             saved fine-tuned models
.ipynb_checkpoints/
mlflow.db            MLflow tracking database
.env                 keys, if a .env file is ever used
results/*.log
results/preds/       files containing dataset texts
results/llm/*.jsonl
results/dialect_tags.parquet
```

### 8. Git and GitHub

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git init -b main
git add .
git commit -m "chore: initialize project with uv, CUDA-enabled PyTorch, and repo layout"
```

Create the GitHub repo, then add the SSH key (`ssh-keygen -t ed25519`, paste the public key into GitHub Settings, SSH and GPG keys), test with `ssh -T git@github.com`, and:

```bash
git remote add origin git@github.com:Muhamm-dk/arabic-sentiment-benchmark.git
git push -u origin main
```

## What went wrong

| Problem | Cause | Fix |
|---|---|---|
| `.wslconfig` seemed ignored: swap stayed at 4 GB | I did not confirm the cause. It was fixed by rewriting the file with `Set-Content ... -Encoding ascii` and running `wsl --shutdown`. Likely an encoding or filename problem from the editor | Write the file from PowerShell, then check `free -h` |
| `nvidia-smi` showed an 80 W power cap, not 115 W | The shown limit depends on the Lenovo power mode and Dynamic Boost. In Performance mode with the charger plugged in, it showed 115 W | Always train in Performance mode, on power |
| `git push` failed: `src refspec main does not match` | `git init` created `master`, not `main`, and the first commit had not been made | `git branch -M main` |
| Push rejected after creating the GitHub repo with a MIT license | The remote had a commit the local repo lacked | `git pull --rebase origin main`, never `--force` |
| `uv init` generated a `uv_build` build system, plus a `[project.scripts]` entry for a module that does not exist | `uv init` defaults. My appended hatchling lines then produced bare `requires` lines inside `[dependency-groups]` and a duplicate table | Rewrote `pyproject.toml` in one piece |
| Heredoc pastes into the terminal got corrupted twice | Long pastes splice commands into files | Use `nano` for large files (Appendix A) |

## Numbers

| Check | Result |
|---|---|
| PyTorch | 2.11.0+cu128, CUDA 12.8 |
| Arch list | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| bf16 matmul | ran without error (mean about -0.021 of random data) |
| VRAM | 7,140,802,560 B free of 8,518,041,600 B total, about 6.6 GiB free |
| WSL | about 21 GiB RAM, 8 GiB swap, 14 processors |
| Python and uv | 3.12, uv 0.13.1 |

**Planning rule:** budget about 6.5 GiB of VRAM, not 8. Windows and the display use the rest, so close browsers and other GPU programs during training. Later measurements: MARBERT used 3.2 GiB at batch 32 on tweets, XLM-R peaked at 6.0 GiB on 256-token reviews at batch 16 with gradient accumulation.

## Check yourself

1. Why must the NVIDIA driver not be installed inside WSL?
2. `torch.cuda.is_available()` returns True. Why is that not enough to start training on an RTX 5070?
3. What does `uv.lock` give you that `pyproject.toml` alone does not?
4. Why is the project stored in `~/projects` and not on `D:`?
5. Why bf16 and not fp16 for fine-tuning?
6. Which two packages does the code import without declaring, and what could go wrong?
