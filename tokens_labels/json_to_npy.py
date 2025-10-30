import json
import argparse
from pathlib import Path
import numpy as np
import re

def label_studio_to_tokens_labels(label_studio_file):
    """读取 Label Studio 导出的 JSON 文件，解析为 tokens + labels"""
    p = Path(label_studio_file)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {label_studio_file}")

    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)

    results = []
    for item in data:
        # 兼容不同字段名，优先取 sentence，其次 text
        data_field = item.get("data", {})
        sentence = data_field.get("sentence") or data_field.get("text") or ""
        sentence = sentence.replace('\n','').strip()  # 清理换行符和首尾空格

        annotations = item.get("annotations") or []
        if not annotations:
            results.append({"sentence": sentence, "tokens": [], "labels": []})
            continue

        # 取最后一次标注
        ann = annotations[-1]
        res = ann.get("result", [])

        # 按 start 排序
        def get_start(x):
            v = x.get("value", {})
            return v.get("start", 0)
        res_sorted = sorted(res, key=get_start)

        tokens, labels = [], []
        for r in res_sorted:
            v = r.get("value", {})
            token_text = v.get("text", "")
            token_text = token_text.replace('\\n','').strip()  # 清理换行符和首尾空格
            lab = v.get("labels") or v.get("label") or []
            if isinstance(lab, list) and lab:
                try:
                    lab_val = int(lab[0])
                except Exception:
                    lab_val = lab[0]
            elif isinstance(lab, str):
                try:
                    lab_val = int(lab)
                except Exception:
                    lab_val = lab
            else:
                lab_val = 0

            tokens.append(token_text)
            labels.append(lab_val)

        # 自动处理 label=[0] 的未分词句子
        if len(labels) == 1 and labels[0] == 0:
            # 用正则拆分句子
            words = re.findall(r'\w+|[^\w\s]', sentence)
            tokens = words
            labels = [0] * len(words)    

        results.append({"sentence": sentence, "tokens": tokens, "labels": labels})

    return results

def save_to_npy(results, out_file):
    """将 results 转换为三个列表并保存为 .npy 文件"""
    sentences = [[item["sentence"]] for item in results]
    tokens = [item["tokens"] for item in results]
    labels = [item["labels"] for item in results]

    data = {
        "sentences": sentences,
        "tokens": tokens,
        "labels": labels
    }

    np.save(out_file, data, allow_pickle=True)
    print(f"✅ 已保存为 {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 Label Studio JSON 转换为 .npy")
    parser.add_argument("--input", "-i", type=str, required=True, help="输入的 Label Studio JSON 文件路径")
    parser.add_argument("--output", "-o", type=str, required=True, help=".npy 输出文件路径")
    args = parser.parse_args()

    results = label_studio_to_tokens_labels(args.input)

    # 保存为 .npy 文件
    save_to_npy(results, args.output)

    # 读取验证示例
    loaded = np.load(args.output, allow_pickle=True).item()
    print("🔍 读取验证示例：")
    #前2条样本
    print("前2条样本：")
    print("sentences 示例:", loaded["sentences"][:2])
    print("tokens 示例:", loaded["tokens"][:2])
    print("labels 示例:", loaded["labels"][:2])
    #后2条样本
    print("后2条样本：")
    print("sentences 示例:", loaded["sentences"][-2:])
    print("tokens 示例:", loaded["tokens"][-2:])
    print("labels 示例:", loaded["labels"][-2:])