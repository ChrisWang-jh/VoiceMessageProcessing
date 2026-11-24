import argparse
import json
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel


ROOT = Path(__file__).resolve().parent


def main():
    """读取场景 JSON 的 text 和 instruct，用指定说话人合成 WAV 音频。"""
    # 读取上一步的 JSON、说话人和音频保存位置。
    parser = argparse.ArgumentParser(description = "读取场景 JSON 并生成情感语音")
    parser.add_argument("json_file", type = Path, help = "generate_scene.py 输出的 JSON 文件")
    parser.add_argument("--speaker", default = "Vivian")
    parser.add_argument("--output", type = Path, default = ROOT / "outputs" / "speech.wav")
    args = parser.parse_args()
    scene = json.loads(args.json_file.read_text(encoding = "utf-8"))

    # 加载本地 CustomVoice 模型，使用 SDPA，无需安装 FlashAttention。
    model = Qwen3TTSModel.from_pretrained(
        str(ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        dtype = torch.bfloat16,
        device_map = "cuda:0",
        attn_implementation = "sdpa",
        local_files_only = True,
    )

    # 将朗读台词和表演指令分别传入，按模型返回的采样率保存。
    wavs, sample_rate = model.generate_custom_voice(
        text = scene["text"],
        instruct = scene["instruct"],
        language = "Chinese",
        speaker = args.speaker,
    )
    args.output.parent.mkdir(parents = True, exist_ok = True)
    sf.write(args.output, wavs[0], sample_rate)
    print(f"已保存：{args.output}（{len(wavs[0]) / sample_rate:.2f} 秒）")


if __name__ == "__main__":
    main()
