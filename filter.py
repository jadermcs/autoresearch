#!/usr/bin/env python
# coding: utf-8
import argparse

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sentence_transformers import CrossEncoder, SentenceTransformer, SparseEncoder
from transformers import AutoTokenizer

m1, m2 = "<t>", "</t>"


def mark(word: str, sentence: str, limit: int = 100):
    lword = len(word)
    lsentence = len(sentence)
    idx = sentence.index(word)
    if idx < 0:
        print(f"{sentence} does not contain '{word}'")
        raise ValueError("Not found")

    return (
        sentence[max(0, idx - limit) : idx]
        + m1
        + word
        + m2
        + sentence[idx + lword : min(lsentence, idx + limit)]
    )


def subword_count(tokenizer, word: str) -> int:
    """Number of subword pieces the tokenizer splits `word` into."""
    toks = tokenizer.tokenize(word)
    return len(toks) if toks else 99


def target_relative_position(word: str, sentence: str) -> float:
    """Position of `word` in `sentence` normalized to [0, 1]. 0.5 = centered."""
    idx = sentence.find(word)
    if idx < 0 or len(sentence) == 0:
        return 0.5
    return idx / len(sentence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pierluigic/xl-lexeme")
    parser.add_argument("--cross_encoder", type=str, default="cross_model")
    parser.add_argument(
        "--subword_tokenizer",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer used to gate target words by subword count.",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--dataset", type=str, required=True)

    # Filter knobs
    parser.add_argument(
        "--max_subwords",
        type=int,
        default=2,
        help="Drop examples whose target word splits into more than this many subwords.",
    )
    parser.add_argument("--min_sentence_tokens", type=int, default=5)
    parser.add_argument("--max_sentence_chars", type=int, default=1000)
    parser.add_argument(
        "--min_lemma_per_label",
        type=int,
        default=2,
        help="Require at least this many examples per (lemma, label) in the output.",
    )
    parser.add_argument("--xl_same_thr", type=float, default=0.75)
    parser.add_argument("--xl_diff_thr", type=float, default=0.40)
    parser.add_argument("--ce_same_thr", type=float, default=0.70)
    parser.add_argument("--ce_diff_thr", type=float, default=0.40)
    parser.add_argument(
        "--require_agreement",
        action="store_true",
        default=True,
        help="Only keep rows where both models agree with the label.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = pd.read_json(args.dataset, orient="records")
    n0 = len(dataset)
    print(f"Loaded {n0} rows.")

    # --- Surface / tokenizer filters (cheap, run first) ---
    print("Applying surface filters...")
    tok = AutoTokenizer.from_pretrained(args.subword_tokenizer)

    sw1 = dataset["word1"].map(lambda w: subword_count(tok, w))
    sw2 = dataset["word2"].map(lambda w: subword_count(tok, w))
    subword_ok = (sw1 <= args.max_subwords) & (sw2 <= args.max_subwords)

    len1 = dataset["sentence1"].str.split().map(len)
    len2 = dataset["sentence2"].str.split().map(len)
    len_ok = (
        (len1 >= args.min_sentence_tokens)
        & (len2 >= args.min_sentence_tokens)
        & (dataset["sentence1"].str.len() <= args.max_sentence_chars)
        & (dataset["sentence2"].str.len() <= args.max_sentence_chars)
    )

    contains_ok = dataset.apply(
        lambda r: (r["word1"] in r["sentence1"]) and (r["word2"] in r["sentence2"]),
        axis=1,
    )

    surface_mask = subword_ok & len_ok & contains_ok
    print(
        f"  subword<= {args.max_subwords}: keep {subword_ok.sum()}/{n0}; "
        f"length ok: {len_ok.sum()}/{n0}; contains ok: {contains_ok.sum()}/{n0}"
    )
    dataset = dataset[surface_mask].reset_index(drop=True)
    print(f"After surface filters: {len(dataset)}")

    if len(dataset) == 0:
        print("Nothing left after surface filters; aborting.")
        return

    # --- Prepare marked sentences ---
    s1 = dataset.apply(lambda x: mark(x["word1"], x["sentence1"]), axis=1)
    s2 = dataset.apply(lambda x: mark(x["word2"], x["sentence2"]), axis=1)

    # --- xl-lexeme similarity ---
    print("Scoring with xl-lexeme...")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.tokenizer.max_seq_length = 128
    model.eval()

    with torch.no_grad():
        embs1 = model.encode(
            s1.values, batch_size=args.batch_size, show_progress_bar=True
        )
        embs2 = model.encode(
            s2.values, batch_size=args.batch_size, show_progress_bar=True
        )
    xl_sim = model.similarity_pairwise(embs1, embs2).cpu().numpy()
    dataset["label_xl-lexeme"] = xl_sim

    del model, embs1, embs2
    torch.cuda.empty_cache()

    # --- Cross-encoder scoring ---
    print(f"Scoring with cross-encoder {args.cross_encoder}...")
    ce = CrossEncoder(args.cross_encoder, max_length=256)
    ce_scores = ce.predict(
        list(zip(s1.values.tolist(), s2.values.tolist())),
        batch_size=args.batch_size,
        show_progress_bar=True,
    )
    ce_scores = np.asarray(ce_scores)
    if ce_scores.ndim == 2 and ce_scores.shape[1] > 1:
        # classification head: take prob of "same sense" class (index 1)
        ce_scores = ce_scores[:, 1]
    dataset["label_cross-encoder"] = ce_scores

    del ce
    torch.cuda.empty_cache()

    # --- Per-model decisions ---
    dataset["xl_pred"] = np.where(
        xl_sim >= args.xl_same_thr,
        1,
        np.where(xl_sim <= args.xl_diff_thr, 0, -1),
    )
    dataset["ce_pred"] = np.where(
        ce_scores >= args.ce_same_thr,
        1,
        np.where(ce_scores <= args.ce_diff_thr, 0, -1),
    )

    # --- Ensemble agreement with the label ---
    xl_agrees = dataset["xl_pred"] == dataset["label"]
    ce_agrees = dataset["ce_pred"] == dataset["label"]

    print("Before ensemble:", len(dataset))
    if args.require_agreement:
        keep = xl_agrees & ce_agrees
    else:
        # softer: both models must not disagree; one can be uncertain (-1)
        keep = (
            ~((dataset["xl_pred"] != -1) & (dataset["xl_pred"] != dataset["label"]))
        ) & (~((dataset["ce_pred"] != -1) & (dataset["ce_pred"] != dataset["label"])))
    dataset = dataset[keep].copy()
    print(f"After ensemble agreement: {len(dataset)}")

    # --- Per-(lemma, label) minimum support ---
    if args.min_lemma_per_label > 1:
        counts = dataset.groupby(["lemma", "label"]).size()
        good = counts[counts >= args.min_lemma_per_label].index
        mask = dataset.set_index(["lemma", "label"]).index.isin(good)
        dataset = dataset[mask].copy()
        print(f"After min_lemma_per_label>= {args.min_lemma_per_label}: {len(dataset)}")

    # --- Final dedup of near-identical pairs (exact string match) ---
    dataset = dataset.drop_duplicates(subset=["sentence1", "sentence2"]).copy()
    print(f"After dedup: {len(dataset)}")

    print(f"Final: {len(dataset)} / {n0} ({len(dataset) / max(n0, 1):.1%} kept)")

    dataset = dataset[
        ["lemma", "pos", "word1", "word2", "sentence1", "sentence2", "label"]
    ]
    dataset.to_json("final.train.json", indent=2, orient="records")

    dataset = Dataset.from_pandas(dataset)
    dataset_test = load_dataset("json", "mcl-wic.test.json")

    model = SparseEncoder("bert-base-uncased")
    mlm_transformer = MLMTransformer(
        args.model,
        max_seq_length=args.max_seq_length,
        config_args={
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
        },
    )
    splade_pooling = models.SpladePooling()
    model = SparseEncoder(
        modules=[mlm_transformer, splade_pooling],
        device=device,
        tokenizer_kwargs={"max_seq_length": 64},
    )
    model.tokenizer.add_tokens([m1, m2])
    model[0].auto_model.resize_token_embeddings(len(model.tokenizer))

    training_args = SparseEncoderTrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=20,
        weight_decay=0.01,
        eval_strategy="epochs",
        save_strategy="epochs",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="sensesimx_cosine_accuracy",
        greater_is_better=True,
        fp16=True,
        warmup_steps=0.1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True if device.type == "cuda" else False,
        report_to="none",
        seed=args.seed,
    )

    eval = evaluation.SparseBinaryClassificationEvaluator
    dev_evaluator = eval(
        sentences1=dataset_dev[args.datasets_dev]["sentence1"],
        sentences2=dataset_dev[args.datasets_dev]["sentence2"],
        labels=dataset_dev[args.datasets_dev]["label"],
        name="sensesimx",
    )

    losses.SpladeLoss.forward = loss_forward
    print(f"Using loss: {args.datasets_loss}")
    con_loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseAnglELoss(model, scale=args.temperature),
        use_document_regularizer_only=True,
        document_regularizer_weight=args.doc_regularization,
    )
    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        loss=con_loss,
        evaluator=dev_evaluator,
    )

    # Train the model
    trainer.train()
    test_evaluator = eval(
        sentences1=dataset_test["sentence1"],
        sentences2=dataset_test["sentence2"],
        labels=dataset_test["label"],
        name="sensesimx-test",
    )

    test_metrics = test_evaluator(model)
    for k, v in test_metrics.items():
        print(f"{k}\t\t{v}")


if __name__ == "__main__":
    main()
