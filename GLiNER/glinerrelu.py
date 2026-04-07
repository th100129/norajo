import json
import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# =========================================================
# 0. 기본 설정
# =========================================================
BATCH_SIZE = 16
BACKBONE_MODEL = "klue/bert-base"
MAX_SPAN_WIDTH = 10
LR = 2e-5
EPOCHS = 5
THRESHOLD = 0.5
D_FF = 512

TRAIN_FILE = "train_gliner.jsonl"
VALID_FILE = "val_gliner.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 1. 유틸 함수
# =========================================================
def infer_entity_types(jsonl_path):
    """
    train jsonl의 span_labels를 보고 클래스 개수 자동 추론
    label 이름을 모르면 label_0, label_1 ... 로 생성
    """
    all_label_ids = set()
    label_counter = Counter()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            for span in item["span_labels"]:
                ent_idx = span[2]
                all_label_ids.add(ent_idx)
                label_counter[ent_idx] += 1

    if len(all_label_ids) == 0:
        raise ValueError("span_labels가 비어 있습니다.")

    max_label_id = max(all_label_ids)
    expected_ids = set(range(max_label_id + 1))

    if all_label_ids != expected_ids:
        raise ValueError(
            f"label id가 연속적이지 않습니다. 발견된 id={sorted(all_label_ids)}"
        )

    entity_types = [f"label_{i}" for i in range(max_label_id + 1)]

    print("===== Entity Type Inference =====")
    print("Detected label ids:", sorted(all_label_ids))
    print("Num classes:", len(entity_types))
    print("Label distribution:", dict(sorted(label_counter.items())))
    print("Entity types:", entity_types)
    print("=================================\n")

    return entity_types


def convert_gold_to_set(span_labels, entity_types):
    gold_set = set()
    for s, e, ent_idx in span_labels:
        if 0 <= ent_idx < len(entity_types):
            gold_set.add((s, e, entity_types[ent_idx]))
    return gold_set


def convert_pred_to_set(pred_items):
    pred_set = set()
    for item in pred_items:
        pred_set.add((item["start"], item["end"], item["label"]))
    return pred_set


def compute_span_f1(pred_batch, gold_batch, entity_types):
    tp, fp, fn = 0, 0, 0

    for pred_items, gold_labels in zip(pred_batch, gold_batch):
        pred_set = convert_pred_to_set(pred_items)
        gold_set = convert_gold_to_set(gold_labels, entity_types)

        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn


# =========================================================
# 2. Dataset / Collate
# =========================================================
class GLiNERDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "text_mask": torch.tensor(item["text_mask"], dtype=torch.long),
            "ent_mask": torch.tensor(item["ent_mask"], dtype=torch.long),
            "word_index": torch.tensor(item["word_index"], dtype=torch.long),
            "span_labels": item["span_labels"]
        }


def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=0
    )

    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0
    )

    text_mask = torch.nn.utils.rnn.pad_sequence(
        [b["text_mask"] for b in batch],
        batch_first=True,
        padding_value=0
    )

    ent_mask = torch.nn.utils.rnn.pad_sequence(
        [b["ent_mask"] for b in batch],
        batch_first=True,
        padding_value=0
    )

    word_index = torch.nn.utils.rnn.pad_sequence(
        [b["word_index"] for b in batch],
        batch_first=True,
        padding_value=-1
    )

    span_labels = [b["span_labels"] for b in batch]

    return input_ids, attention_mask, text_mask, ent_mask, word_index, span_labels


# =========================================================
# 3. Encoder
# =========================================================
class EncoderModule(nn.Module):
    def __init__(self, backbone_model: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_model)
        self.hidden_size = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # [B, L, H]


# =========================================================
# 4. Representation 모듈
# =========================================================
class EntityTokenRepresentation(nn.Module):
    """
    ent_mask == 1 인 토큰들을 추출
    모든 샘플에서 ent 개수는 같아야 함
    """
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, ent_mask):
        ent_vectors = []
        ent_lens = []

        for hs, mask in zip(hidden_states, ent_mask):
            selected = hs[mask.bool()]  # [E_i, H]
            ent_vectors.append(selected)
            ent_lens.append(selected.size(0))

        if len(set(ent_lens)) != 1:
            raise ValueError(
                f"샘플마다 ent_mask.sum() 값이 다릅니다: {ent_lens}"
            )

        return torch.stack(ent_vectors, dim=0)  # [B, E, H]


class WordTokenRepresentation(nn.Module):
    """
    text_mask == 1 인 text token들을 word_index 기준으로 word-level로 pooling
    span_labels가 word 단위라고 가정하고 맞춰줌
    """
    def __init__(self, pooling="mean"):
        super().__init__()
        assert pooling in ["mean", "first"]
        self.pooling = pooling

    def forward(self, hidden_states, text_mask, word_index):
        """
        hidden_states: [B, L, H]
        text_mask:     [B, L]
        word_index:    [B, L]
        return:
          padded_word_embeddings: [B, W_max, H]
          word_lengths: [B]
        """
        batch_word_embeddings = []
        word_lengths = []

        B, L, H = hidden_states.shape

        for b in range(B):
            hs = hidden_states[b]       # [L, H]
            tm = text_mask[b]           # [L]
            wi = word_index[b]          # [L]

            valid_pos = (tm == 1) & (wi >= 0)

            hs_valid = hs[valid_pos]    # [T, H]
            wi_valid = wi[valid_pos]    # [T]

            if hs_valid.size(0) == 0:
                raise ValueError("text_mask와 word_index 기준으로 유효한 text token이 없습니다.")

            max_word_idx = int(wi_valid.max().item())
            num_words = max_word_idx + 1

            word_vecs = []
            for word_id in range(num_words):
                pos = (wi_valid == word_id)

                token_vecs = hs_valid[pos]  # [n_subtokens, H]
                if token_vecs.size(0) == 0:
                    raise ValueError(f"word_id={word_id}에 해당하는 token이 없습니다.")

                if self.pooling == "mean":
                    word_vec = token_vecs.mean(dim=0)
                else:
                    word_vec = token_vecs[0]

                word_vecs.append(word_vec)

            word_vecs = torch.stack(word_vecs, dim=0)  # [W_i, H]
            batch_word_embeddings.append(word_vecs)
            word_lengths.append(word_vecs.size(0))

        max_len = max(word_lengths)
        padded = []

        for x in batch_word_embeddings:
            pad_len = max_len - x.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, x.size(1), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            padded.append(x)

        padded_word_embeddings = torch.stack(padded, dim=0)  # [B, W_max, H]
        word_lengths = torch.tensor(word_lengths, device=hidden_states.device, dtype=torch.long)

        return padded_word_embeddings, word_lengths


class EntityRepresentation(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            # 🔥 GELU에서 LeakyReLU로 변경!
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class SpanRepresentation(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_ff),
            # 🔥 GELU에서 LeakyReLU로 변경!
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, start_token, end_token):
        span = torch.cat([start_token, end_token], dim=-1)
        return self.net(span)


class Matching(nn.Module):
    """
    logits 반환
    span:   [1, N, H]
    entity: [1, E, H]
    out:    [1, N, E]
    """
    def forward(self, entity, span):
        entity_t = entity.transpose(-1, -2)   # [1, H, E]
        logits = span @ entity_t              # [1, N, E]
        return logits


# =========================================================
# 5. Decoder
# =========================================================
class Decoder(nn.Module):
    def __init__(self, entity_types, threshold=0.5, mode="flat"):
        super().__init__()
        self.entity_types = entity_types
        self.threshold = threshold
        self.mode = mode

    def overlapping(self, span1, span2):
        s1, e1 = span1
        s2, e2 = span2
        return not (e1 < s2 or e2 < s1)

    def partial_overlap(self, span1, span2):
        s1, e1 = span1
        s2, e2 = span2
        overlap = not (e1 < s2 or e2 < s1)
        if not overlap:
            return False
        nested = (s1 <= s2 and e2 <= e1) or (s2 <= s1 and e1 <= e2)
        return not nested

    def decode_one(self, probs, span_indices):
        candidates = []
        n_span, n_ent = probs.shape

        for n in range(n_span):
            s, e = span_indices[n]
            for ent_idx in range(n_ent):
                score = probs[n, ent_idx].item()
                if score > self.threshold:
                    candidates.append({
                        "start": s,
                        "end": e,
                        "label": self.entity_types[ent_idx],
                        "score": score
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        selected = []
        for cand in candidates:
            conflict = False
            c_span = (cand["start"], cand["end"])

            for picked in selected:
                p_span = (picked["start"], picked["end"])

                if self.mode == "flat":
                    if self.overlapping(c_span, p_span):
                        conflict = True
                        break
                elif self.mode == "nested":
                    if self.partial_overlap(c_span, p_span):
                        conflict = True
                        break

            if not conflict:
                selected.append(cand)

        return selected

    def decode_batch(self, probs_batch, span_indices_batch):
        results = []
        for probs, span_indices in zip(probs_batch, span_indices_batch):
            results.append(self.decode_one(probs, span_indices))
        return results


# =========================================================
# 6. GLiNER 모델
# =========================================================
class GLiNER(nn.Module):
    def __init__(
        self,
        backbone_model,
        entity_types,
        d_ff=512,
        max_span_width=12,
        threshold=0.5,
        mode="flat",
        word_pooling="mean"
    ):
        super().__init__()
        self.entity_types = entity_types
        self.num_entity_types = len(entity_types)
        self.max_span_width = max_span_width

        self.encoder = EncoderModule(backbone_model)
        d_model = self.encoder.hidden_size

        self.word_rep = WordTokenRepresentation(pooling=word_pooling)
        self.ent_token_rep = EntityTokenRepresentation()

        self.ent_rep = EntityRepresentation(d_model, d_ff)
        self.span_rep = SpanRepresentation(d_model, d_ff)
        self.matching = Matching()
        self.decoder = Decoder(entity_types, threshold, mode)

    def build_span_representations(self, word_embeddings, word_lengths):
        """
        word_embeddings: [B, W_max, H]
        word_lengths: [B]
        return:
          span_vectors_batch: list([N_i, H])
          span_indices_batch: list([(start, end), ...])
        """
        batch_span_vectors = []
        batch_span_indices = []

        B, _, H = word_embeddings.shape

        for b in range(B):
            w_len = word_lengths[b].item()
            cur_words = word_embeddings[b]  # [W_max, H]

            starts = []
            ends = []
            span_indices = []

            for start in range(w_len):
                max_end = min(w_len, start + self.max_span_width)
                for end in range(start, max_end):
                    starts.append(start)
                    ends.append(end)
                    span_indices.append((start, end))

            if len(span_indices) == 0:
                span_vecs = torch.zeros(0, H, device=cur_words.device, dtype=cur_words.dtype)
            else:
                start_embs = cur_words[starts]   # [N_i, H]
                end_embs = cur_words[ends]       # [N_i, H]
                span_vecs = self.span_rep(start_embs, end_embs)

            batch_span_vectors.append(span_vecs)
            batch_span_indices.append(span_indices)

        return batch_span_vectors, batch_span_indices

    def forward(self, input_ids, attention_mask, text_mask, ent_mask, word_index):
        hidden_states = self.encoder(input_ids, attention_mask)  # [B, L, H]

        word_embeddings, word_lengths = self.word_rep(
            hidden_states=hidden_states,
            text_mask=text_mask,
            word_index=word_index
        )  # [B, W_max, H], [B]

        ent_embeddings = self.ent_token_rep(
            hidden_states=hidden_states,
            ent_mask=ent_mask
        )  # [B, E, H]

        ent_vectors = self.ent_rep(ent_embeddings)  # [B, E, H]

        span_vectors_batch, span_indices_batch = self.build_span_representations(
            word_embeddings, word_lengths
        )

        logits_batch = []
        for b in range(input_ids.size(0)):
            span_vectors = span_vectors_batch[b]                # [N_i, H]
            ent_vec = ent_vectors[b].unsqueeze(0)              # [1, E, H]
            span_vec = span_vectors.unsqueeze(0)               # [1, N_i, H]
            logits = self.matching(ent_vec, span_vec).squeeze(0)  # [N_i, E]
            logits_batch.append(logits)

        return logits_batch, span_indices_batch

    @torch.no_grad()
    def predict(self, input_ids, attention_mask, text_mask, ent_mask, word_index):
        self.eval()
        logits_batch, span_indices_batch = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_mask=text_mask,
            ent_mask=ent_mask,
            word_index=word_index
        )
        probs_batch = [torch.sigmoid(x) for x in logits_batch]
        return self.decoder.decode_batch(probs_batch, span_indices_batch)


# =========================================================
# 7. Loss 관련
# =========================================================
def build_targets_for_batch(logits_batch, span_indices_batch, span_labels_batch, num_entity_types, device):
    target_batch = []

    for logits, span_indices, gold_labels in zip(logits_batch, span_indices_batch, span_labels_batch):
        n_span = logits.size(0)
        target = torch.zeros((n_span, num_entity_types), device=device)

        span_to_idx = {span: i for i, span in enumerate(span_indices)}

        for label in gold_labels:
            l_start, l_end, ent_idx = label[0], label[1], label[2]
            span_tuple = (l_start, l_end)

            if span_tuple in span_to_idx and 0 <= ent_idx < num_entity_types:
                target[span_to_idx[span_tuple], ent_idx] = 1.0

        target_batch.append(target)

    return target_batch


def compute_loss(logits_batch, target_batch, pos_weight=None):
    total_loss = 0.0
    total_count = 0

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="sum")

    for logits, target in zip(logits_batch, target_batch):
        if logits.numel() == 0:
            continue
        loss = criterion(logits, target)
        total_loss += loss
        total_count += target.numel()

    if total_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return total_loss / total_count


# =========================================================
# 8. Train / Validate
# =========================================================
def train_one_epoch(model, dataloader, optimizer, entity_types, pos_weight=None):
    model.train()
    total_loss = 0.0

    train_bar = tqdm(dataloader, desc="[Train]")

    for batch in train_bar:
        input_ids, attention_mask, text_mask, ent_mask, word_index, span_labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_mask = text_mask.to(device)
        ent_mask = ent_mask.to(device)
        word_index = word_index.to(device)

        optimizer.zero_grad()

        logits_batch, span_indices_batch = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_mask=text_mask,
            ent_mask=ent_mask,
            word_index=word_index
        )

        target_batch = build_targets_for_batch(
            logits_batch=logits_batch,
            span_indices_batch=span_indices_batch,
            span_labels_batch=span_labels,
            num_entity_types=len(entity_types),
            device=device
        )

        loss = compute_loss(logits_batch, target_batch, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def validate_one_epoch(model, dataloader, entity_types, pos_weight=None):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_golds = []

    valid_bar = tqdm(dataloader, desc="[Valid]")

    for batch in valid_bar:
        input_ids, attention_mask, text_mask, ent_mask, word_index, span_labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_mask = text_mask.to(device)
        ent_mask = ent_mask.to(device)
        word_index = word_index.to(device)

        logits_batch, span_indices_batch = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_mask=text_mask,
            ent_mask=ent_mask,
            word_index=word_index
        )

        target_batch = build_targets_for_batch(
            logits_batch=logits_batch,
            span_indices_batch=span_indices_batch,
            span_labels_batch=span_labels,
            num_entity_types=len(entity_types),
            device=device
        )

        loss = compute_loss(logits_batch, target_batch, pos_weight=pos_weight)
        total_loss += loss.item()

        probs_batch = [torch.sigmoid(x) for x in logits_batch]
        pred_batch = model.decoder.decode_batch(probs_batch, span_indices_batch)

        all_preds.extend(pred_batch)
        all_golds.extend(span_labels)

        valid_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    precision, recall, f1, tp, fp, fn = compute_span_f1(all_preds, all_golds, entity_types)

    return avg_loss, precision, recall, f1, tp, fp, fn


# =========================================================
# 9. 디버깅용 체크 함수
# =========================================================
def inspect_dataset(file_path, num_samples=3):
    print(f"===== Inspect Dataset: {file_path} =====")
    with open(file_path, "r", encoding="utf-8") as f:
        for i in range(num_samples):
            item = json.loads(next(f))
            print(f"[Sample {i}]")
            print("len(input_ids)     =", len(item["input_ids"]))
            print("sum(attention_mask)=", sum(item["attention_mask"]))
            print("sum(text_mask)     =", sum(item["text_mask"]))
            print("sum(ent_mask)      =", sum(item["ent_mask"]))
            print("max(word_index)    =", max(item["word_index"]))
            print("span_labels        =", item["span_labels"])
            print("-" * 50)
    print("========================================\n")


# =========================================================
# 10. Main
# =========================================================
if __name__ == "__main__":
    inspect_dataset(TRAIN_FILE, num_samples=2)

    entity_types = infer_entity_types(TRAIN_FILE)

    model = GLiNER(
        backbone_model=BACKBONE_MODEL,
        entity_types=entity_types,
        d_ff=D_FF,
        max_span_width=MAX_SPAN_WIDTH,
        threshold=THRESHOLD,
        mode="flat",
        word_pooling="mean"   # "mean" 또는 "first"
    ).to(device)

    train_dataset = GLiNERDataset(TRAIN_FILE)
    valid_dataset = GLiNERDataset(VALID_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    # 클래스 불균형 심하면 여기에 weight 넣어도 됨
    # 예: torch.tensor([3.,3.,3.,3.,3.,3.], device=device)
    pos_weight = None

    best_f1 = 0.0

    print(f"Device          : {device}")
    print(f"Backbone        : {BACKBONE_MODEL}")
    print(f"Entity Types    : {entity_types}")
    print(f"Num Entity Types: {len(entity_types)}")
    print(f"Max Span Width  : {MAX_SPAN_WIDTH}")
    print(f"Threshold       : {THRESHOLD}")
    print("=" * 80)

    for epoch in range(EPOCHS):
        print(f"\n===== Epoch {epoch + 1}/{EPOCHS} =====")

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            entity_types=entity_types,
            pos_weight=pos_weight
        )

        valid_loss, precision, recall, f1, tp, fp, fn = validate_one_epoch(
            model=model,
            dataloader=valid_loader,
            entity_types=entity_types,
            pos_weight=pos_weight
        )

        print(f"\n[Epoch {epoch + 1} 결과]")
        print(f"Train Loss : {train_loss:.4f}")
        print(f"Valid Loss : {valid_loss:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1 Score   : {f1:.4f}")
        print(f"TP / FP / FN = {tp} / {fp} / {fn}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "entity_types": entity_types,
                    "threshold": THRESHOLD,
                    "max_span_width": MAX_SPAN_WIDTH,
                    "backbone_model": BACKBONE_MODEL
                },
                "best_gliner.pt"
            )
            print("✅ best_gliner.pt 저장 완료")

    print("\n학습 종료")
    print(f"Best Valid F1: {best_f1:.4f}")
