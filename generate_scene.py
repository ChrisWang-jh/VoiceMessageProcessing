import argparse
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
SYSTEM_PROMPT = """
你是中文台词编剧和配音导演。根据用户的要求提取原始台词和目标情感，
补充合理场景、台词和表演指令。未指定的细节由你合理补全。
只输出一个合法 JSON 对象，包含以下五个非空字符串字段，不要 Markdown 或解释：
{
  "original_line": "用户的原始台词",
  "emotion": "目标情感",
  "scene": "用一句话描述引出该情感的场景",
  "text": "保留原始台词的文字和顺序，前后补充一到两句，总长不超过100字",
  "instruct": "符合场景和目标情感的中文配音指令，描述语气、语速、重音和停顿"
}
text 只包含同一个人需要朗读的台词，不包含旁白、角色标签或舞台动作。
"""


class Scene(BaseModel):
    """两个脚本交换的 JSON：原句、情感、场景、朗读台词、表演指令。"""

    model_config = ConfigDict(extra = "forbid", str_min_length = 1, str_strip_whitespace = True)
    original_line: str
    emotion: str
    scene: str
    text: str = Field(max_length = 100)
    instruct: str


def main():
    """调用本地语言模型，将用户 prompt 补全为经过校验的 JSON 文件。"""
    # 读取用户要求和 JSON 保存位置。
    parser = argparse.ArgumentParser(description = "根据台词和情感生成场景 JSON")
    parser.add_argument("prompt", help = "例如：台词是‘你回来了’，情感是欣喜")
    parser.add_argument("--output", type = Path, default = ROOT / "outputs" / "scene.json")
    args = parser.parse_args()

    # 加载项目内的语言模型，使用可见的第一张 GPU。
    model_path = ROOT / "models" / "Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype = torch.bfloat16,
        device_map = "cuda:0",
        attn_implementation = "sdpa",
        local_files_only = True,
    )

    # 按模型的聊天模板编码提示词，只解码新生成的内容。
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens = 512, do_sample = False)
    response = tokenizer.decode(tokens[0, inputs.input_ids.shape[1]:], skip_special_tokens = True)

    # 校验字段和原句保留情况，通过后才保存给 TTS 使用。
    scene = Scene.model_validate_json(response)
    if scene.original_line not in scene.text:
        raise ValueError("生成的 text 未完整保留 original_line")
    output = scene.model_dump_json(indent = 2)
    args.output.parent.mkdir(parents = True, exist_ok = True)
    args.output.write_text(output + "\n", encoding = "utf-8")
    print(output)


if __name__ == "__main__":
    main()
