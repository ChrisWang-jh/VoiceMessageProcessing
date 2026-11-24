# 中文情感语音生成

语音信息处理课程大作业：输入台词、情感及可选背景，自动扩写情景和表演指令，再生成语音。

```text
用户 prompt → Qwen3-4B-Instruct-2507 → 场景 JSON → Qwen3-TTS CustomVoice → WAV
```

## 1. 安装环境

需要 Linux、[uv](https://docs.astral.sh/uv/) 和可用的 NVIDIA GPU。当前使用 Python 3.12、PyTorch 2.8.0（CUDA 12.8）和 BF16，已在 RTX A6000 上验证；无需安装 FlashAttention。

以下命令均在项目根目录执行：

```bash
uv python install 3.12
uv sync --locked
```

Ubuntu 若缺少音频系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y sox libsndfile1
```

## 2. 下载模型

通过 ModelScope 下载完整权重，已有模型可跳过：

```bash
uv run --locked modelscope download \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --local_dir models/Qwen3-4B-Instruct-2507

uv run --locked modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local_dir models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

模型来源：[语言模型](https://www.modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)、[TTS 模型](https://www.modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)。脚本从项目的 `models/` 加载权重，TTS 目录中的 `speech_tokenizer/` 也必须保留。

## 3. 一条命令生成语音

先通过 `nvidia-smi` 确认空闲 GPU，将下面的 `0` 替换为对应编号：

```bash
CUDA_VISIBLE_DEVICES=0 uv run --locked python pipeline.py \
  '台词是“你回来了”，情感是欣喜，背景是久别重逢。' \
  --speaker Uncle_Fu \
  --output outputs/happy.wav
```

输出：

- `outputs/happy.json`：原句、情感、场景、扩写台词、表演指令。
- `outputs/happy.wav`：使用所选音色合成的中文语音。

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `prompt` | 必填，描述台词、情感及可选背景 | 无 |
| `--speaker` | 预置音色 | `Vivian` |
| `--output` | WAV 路径，必须以 `.wav` 结尾；JSON 保存为同名前缀 | `outputs/speech.wav` |

可选音色：`Vivian`、`Serena`、`Uncle_Fu`、`Dylan`、`Eric`、`Ryan`、`Aiden`、`Ono_Anna`、`Sohee`。

总脚本顺序启动两个进程，第一步成功退出后才运行 TTS。同名输出会被覆盖。

## 4. 分步运行

```bash
uv run --locked python generate_scene.py \
  '台词是“你回来了”，情感是愤怒。' \
  --output outputs/angry.json

uv run --locked python synthesize.py outputs/angry.json \
  --speaker Serena \
  --output outputs/angry.wav
```

`generate_scene.py` 顶部的 `SYSTEM_PROMPT` 控制扩写规则。JSON 固定包含 `original_line`、`emotion`、`scene`、`text`、`instruct`；TTS 分别读取 `text` 作为朗读内容、`instruct` 作为表演指令。第一步报错时应先解决错误，再运行 TTS。

依赖由 `pyproject.toml` 和 `uv.lock` 管理。模型、虚拟环境和生成结果已通过 `.gitignore` 排除。当前已验证完整生成链路，情感效果尚未系统评估；TTS 未固定随机种子，重复运行的音频可能不同。
