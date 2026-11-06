# HANDOFF — 环境与工程约定（给协作的 AI）

> 这份文档只写"别人不知道、但会踩坑"的部分：机器、网络、WSL、Python 环境、仓库、口径。

## 0. 一句话

项目跑在 WSL2 的 Ubuntu 24.04 里，Python 环境叫 `av`，仓库在 `~/projects/av-anomaly`。
所有命令默认在 Ubuntu 终端或 VS Code 的 WSL 终端里执行，**不是** Windows PowerShell。

## 1. 机器

- Windows 11 25H2（Build 26200.9711），联想 83EG
- AMD Ryzen 7 7840H，8 核 16 线程，16GB 内存
- NVIDIA RTX 4060 Laptop，**8GB 显存**，驱动 560.94，支持 CUDA 12.6
- 单块 477GB NVMe：C: 200GB（剩约 28GB），D: 275GB（剩约 100GB）

## 2. WSL

- WSL 版本 3.0.3.0，内核 6.18.54.1-1
- 发行版 `Ubuntu-24.04`，VERSION 2，已设为默认
- 系统磁盘：`D:\WSL\Ubuntu-24.04\ext4.vhdx`（不是 C 盘）
- 交换文件：`D:\WSL\swap.vhdx`
- Linux 用户名 `qiaojiahao`，家目录 `/home/qiaojiahao`
- 另有一个 `docker-desktop` 发行版（Docker Desktop 的内部件，**不要动**）

`C:\Users\乔佳豪\.wslconfig`：

```ini
[wsl2]
memory=6GB
processors=8
swap=8GB
swapFile=D:/WSL/swap.vhdx
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
```

注意 `swapFile` 必须是正斜杠，反斜杠会被当成转义符报错。

## 3. 网络（最容易卡住的地方）

- 代理：Clash Verge，规则模式，TUN 关闭，系统代理 `127.0.0.1:7897`
- WSL 用 mirrored 网络，自动继承 `http_proxy` / `https_proxy`
- **DNS 被 fake-ip 劫持**，国内域名也解析成 `198.18.x.x`，所以：
  - 不走代理的直连会超时（`http=000`）
  - 一切网络操作都必须经过代理，或让代理按规则直连
- apt 已配代理：`/etc/apt/apt.conf.d/99proxy`
- apt 源：清华 deb822（`/etc/apt/sources.list.d/ubuntu.sources`）
- pip 源：清华（`~/.config/pip/pip.conf`）

**大文件下载的铁律**：pip 下 500MB 以上的包容易卡在几十 kB/s 且不会自己恢复。
改用：

```bash
curl -L -C - -o pkg.whl "<url>"
pip install ./pkg-1.2.3-cp311-....whl
```

文件名必须保留标准 wheel 格式，`torch.whl` 这种会报 `Invalid wheel filename`。

## 4. Python 环境

- Miniforge 装在 `~/miniforge3`，环境名 `av`
- Python **3.11.18**
- 解释器：`/home/qiaojiahao/miniforge3/envs/av/bin/python`
- 激活：`conda activate av`（VS Code 默认终端已自动激活）

已装：

| 包 | 版本 |
|---|---|
| torch | 2.14.1+cu126 |
| torchvision | 0.29.1+cu126 |

未装：pandas、duckdb、scikit-learn、jupyter

验证 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))"
```

预期：`True NVIDIA GeForce RTX 4060 Laptop GPU 8.0`

## 5. 仓库

- 路径：`/home/qiaojiahao/projects/av-anomaly`
- 远程：`git@github.com:jiahaoqiao770-collab/av-anomaly.git`（private）
- 分支 `main`，已有首个提交 `a269c2a`
- 已有 `.gitignore`、`README.md`
- Git 身份：`乔佳豪` / `jiahaoqiao770@gmail.com`
- SSH 密钥（ed25519）已配置，`ssh -T git@github.com` 通过
- 提交前缀约定：`feat:` / `fix:` / `chore:` / `docs:`

## 6. 目录约定

```
~/projects/av-anomaly     代码（git 仓库）
~/datasets/av-raw         原始 CSV（.gitignore 已忽略）
~/models                  模型权重（忽略）
~/notebooks               临时分析（忽略）
~/datasets                数据集
```

