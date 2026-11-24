import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    """顺序调用语言模型和 TTS，将同名 JSON 与 WAV 保存到指定位置。"""
    # 读取用户要求、预置角色和音频保存位置。
    parser = argparse.ArgumentParser(description = "输入 prompt，一次生成场景 JSON 和情感语音")
    parser.add_argument("prompt", help = "例如：台词是‘你回来了’，情感是欣喜")
    parser.add_argument(
        "--speaker",
        default = "Vivian",
        choices = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"],
    )
    parser.add_argument("--output", type = Path, default = ROOT / "outputs" / "speech.wav")
    args = parser.parse_args()

    # JSON 与音频共用文件名前缀，便于对应每次生成结果。
    audio_path = args.output.resolve()
    if audio_path.suffix.lower() != ".wav":
        parser.error("--output 请使用 .wav 扩展名")
    json_path = audio_path.with_suffix(".json")

    # 使用当前 Python 环境生成 JSON，等待进程退出并释放语言模型显存。
    subprocess.run(
        [sys.executable, str(ROOT / "generate_scene.py"), args.prompt, "--output", str(json_path)],
        check = True,
    )

    # 仅在上一步成功后启动 TTS，将所选角色和刚生成的 JSON 传入。
    subprocess.run(
        [
            sys.executable, str(ROOT / "synthesize.py"), str(json_path),
            "--speaker", args.speaker, "--output", str(audio_path),
        ],
        check = True,
    )
    print(f"JSON：{json_path}\n音频：{audio_path}")


if __name__ == "__main__":
    main()
