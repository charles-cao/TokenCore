from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
def print_tokenization_alignment(tokenizer, tokens):
    """
    Print how each original token is split into subwords.
    Useful for debugging and understanding tokenization.
    """
    print("Tokenization alignment:")
    print("-" * 40)

    # Tokenize without tensors to get subwords
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_offsets_mapping=False,
        add_special_tokens=True,  # Include [CLS], [SEP]
    )

    input_ids = encoded["input_ids"]
    sub_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    word_ids = encoded.word_ids(batch_index=0)

    current_word = None
    for i, (sub_token, word_idx) in enumerate(zip(sub_tokens, word_ids)):
        if word_idx is None:
            print(f"  [Special] {sub_token:10} -> position {i} (ignored)")
        elif word_idx != current_word:
            original_token = tokens[word_idx]
            print(f"  Token {word_idx:2} '{original_token:10}' -> subword '{sub_token:10}' at position {i} (used)")
            current_word = word_idx
        else:
            print(f"           {' ':10}     subword '{sub_token:10}' at position {i} (skipped)")


def extract_word_embeddings(
    model,
    tokenizer,
    all_tokens,
    all_labels=None,
    model_name="bert-base-cased",
    device=None,
):
    """
    Extract word-level embeddings with full alignment logging.
    Returns list of dicts: [{'token', 'embedding', 'label', 'subword_position'}, ...]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    local_cache_dir = "D:\model"

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_cache_dir)
    if model is None:
        model = AutoModel.from_pretrained(model_name, cache_dir=local_cache_dir)
    model = model.to(device).eval()

    result_sentences = []
    if all_labels is None:
        all_labels = [None] * len(all_tokens)

    with torch.no_grad():
        for i, (tokens, labels) in enumerate(zip(all_tokens, all_labels)):
            print(f"\n" + "="*60)
            print(f"Sentence {i+1}/{len(all_tokens)}: {' '.join(tokens)}")
            print_tokenization_alignment(tokenizer, tokens)

            # Actual tokenization for model input
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=False,
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            word_ids = encoded.word_ids(batch_index=0)
# --------------------------------------------------------------------------------------------
            # sentence_data = []
            # previous_word_idx = None

            # for token_position, word_idx in enumerate(word_ids):
            #     if word_idx is None:
            #         continue  # Skip [CLS], [SEP], etc.

            #     if word_idx != previous_word_idx:
            #         emb = last_hidden_state[0, token_position, :].cpu().numpy()
            #         token_str = tokens[word_idx]
            #         label = labels[word_idx] if labels else None

            #         sentence_data.append({
            #             "token": token_str,
            #             "embedding": emb,
            #             "label": label,
            #             "subword_position": token_position,
            #         })
            #     previous_word_idx = word_idx
# ----------------------------------------------------------------------------------------------
            sentence_data = []
            previous_word_idx = None
            current_subwords = []  # 用于累积当前词的所有 subword embedding
            current_token = None
            current_label = None
            current_positions = []

            for token_position, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue  # Skip [CLS], [SEP], etc.

                if word_idx != previous_word_idx:
                    # 当进入新词时，先输出前一个词的平均 embedding
                    if current_subwords:
                        avg_emb = np.max(np.stack(current_subwords), axis=0)
                        sentence_data.append({
                            "token": current_token,
                            "embedding": avg_emb,
                            "label": current_label,
                            "subword_position": current_positions,
                        })
                    # 开始新词
                    current_token = tokens[word_idx]
                    current_label = labels[word_idx] if labels else None
                    current_subwords = [last_hidden_state[0, token_position, :].cpu().numpy()]
                    current_positions = [token_position]
                else:
                    # 同一个词的后续 subword
                    current_subwords.append(last_hidden_state[0, token_position, :].cpu().numpy())
                    current_positions.append(token_position)

                previous_word_idx = word_idx

            # 处理最后一个词
            if current_subwords:
                avg_emb = np.max(np.stack(current_subwords), axis=0)
                sentence_data.append({
                    "token": current_token,
                    "embedding": avg_emb,
                    "label": current_label,
                    "subword_position": current_positions,
                })
#--------------------------------------------------------------------------------------------------
            result_sentences.append(sentence_data)
            print(f"✅ Extracted {len(sentence_data)} word embeddings for this sentence.")

    print(f"\n🎉 All done. Total sentences processed: {len(result_sentences)}")
    return result_sentences
