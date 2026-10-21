"""BERTScore recall of every post against the 38 topic sentences, without
re-encoding the topic sentences 590,000 times.

The 2025 run called `bertscore.compute(predictions=[post]*38, references=topics)`
once per post. That re-embeds the 38 topic sentences for every post, and with a 750M
parameter model it measured out at ~16 posts/s, or about ten hours for the sample.

Here the topic sentences are embedded once and the posts are embedded in batches;
the pairwise scores then come from `bert_score.utils.greedy_cos_idf`, which is the
library's own kernel. Using the library's kernel rather than reimplementing the
greedy matching is deliberate: agreement with the published numbers is then a
property of the construction, not something to be hoped for. `validate_against`
checks it anyway.

Settings match the original exactly: microsoft/deberta-xlarge-mnli, idf=False,
rescale_with_baseline=True, lang='en', and negative recalls clipped to zero.
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from bert_score.utils import (get_bert_embedding, get_model, get_tokenizer,
                              greedy_cos_idf, model2layers)

MODEL_TYPE = "microsoft/deberta-xlarge-mnli"

# download_model.sh puts the weights here. Pointing at a local directory avoids
# going back to the Hub, whose recent versions restart a stalled download from zero
# rather than resuming it.
LOCAL_MODEL_DIR = os.path.expanduser("~/models/deberta-xlarge-mnli")

# DeBERTa's max_position_embeddings. Only used to replace the sentinel value some
# tokenizer configs carry; see TopicScorer.__init__.
MAX_SEQ_LENGTH = 512


def resolve_model(model_type=None):
    """Prefer the locally downloaded copy; fall back to the Hub name."""
    if model_type:
        return model_type
    if os.path.isdir(LOCAL_MODEL_DIR) and os.path.exists(
            os.path.join(LOCAL_MODEL_DIR, "config.json")):
        return LOCAL_MODEL_DIR
    return MODEL_TYPE


def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def baseline_recall(model_type=MODEL_TYPE, lang="en"):
    """
    The rescaling constant bert_score applies when rescale_with_baseline=True.

    Keyed on the Hub name even when the weights are loaded from a local directory:
    the baseline belongs to the model, not to where the file happens to sit.
    """
    import bert_score
    path = os.path.join(os.path.dirname(bert_score.__file__),
                        "rescale_baseline", lang, MODEL_TYPE + ".tsv")
    return float(pd.read_csv(path).iloc[model2layers[MODEL_TYPE]]["R"])


class TopicScorer:
    """
    Score posts against a fixed set of reference sentences.

        scorer = TopicScorer(cfg.TOPIC_SENTENCES)
        recall = scorer.score(list_of_posts)      # (n_posts, 38), negatives clipped

    The reference embeddings are computed once in __init__ and reused.
    """

    def __init__(self, references, model_type=None, device=None, batch_size=16):
        self.device = pick_device(device)
        self.model_type = resolve_model(model_type)
        # The layer is a property of the architecture, so look it up by Hub name
        # even when the weights come off disk.
        self.num_layers = model2layers[MODEL_TYPE]
        self.batch_size = batch_size

        self.tokenizer = get_tokenizer(self.model_type, use_fast=False)
        # bert_score passes tokenizer.model_max_length straight through to
        # encode(max_length=...). When the tokenizer config does not set it,
        # transformers fills in a sentinel of ~1e30, which transformers 5.x then
        # refuses ("int too big to convert"). DeBERTa's real limit is 512.
        if self.tokenizer.model_max_length > 100000:
            self.tokenizer.model_max_length = MAX_SEQ_LENGTH
        self.model = get_model(self.model_type, self.num_layers, all_layers=False)
        self.model.eval().to(self.device)

        # idf=False means every token counts equally; the separators count for nothing
        self.idf_dict = defaultdict(lambda: 1.0)
        self.idf_dict[self.tokenizer.sep_token_id] = 0
        self.idf_dict[self.tokenizer.cls_token_id] = 0

        self.references = list(references)
        self.n_ref = len(self.references)
        with torch.no_grad():
            self.ref_emb, self.ref_masks, self.ref_idf = get_bert_embedding(
                self.references, self.model, self.tokenizer, self.idf_dict,
                device=self.device, all_layers=False)
        self.baseline = baseline_recall()

    def score(self, posts, clip_negative=True, progress=None):
        """
        Return an (n_posts, n_references) array of rescaled recall scores.

        Posts that are empty after cleaning score zero on every topic without being
        sent to the model. That is the right answer -- a post with no content left
        mentions no topic -- and it also avoids bert_score's empty-string branch,
        which calls `tokenizer.build_inputs_with_special_tokens`, a method
        transformers 5.x no longer provides on the slow tokenizers.
        """
        texts = [str(p) for p in posts]
        out = np.zeros((len(texts), self.n_ref), dtype=np.float32)
        keep = [i for i, t in enumerate(texts) if t.strip()]
        if not keep:
            return out

        for start in range(0, len(keep), self.batch_size):
            idx = keep[start:start + self.batch_size]
            batch = [texts[i] for i in idx]
            with torch.no_grad():
                hyp_emb, hyp_masks, hyp_idf = get_bert_embedding(
                    batch, self.model, self.tokenizer, self.idf_dict,
                    device=self.device, all_layers=False)
                b = len(batch)
                # every post against every reference: repeat posts, tile references
                he = hyp_emb.repeat_interleave(self.n_ref, dim=0)
                hm = hyp_masks.repeat_interleave(self.n_ref, dim=0)
                hi = hyp_idf.repeat_interleave(self.n_ref, dim=0)
                re_ = self.ref_emb.repeat(b, 1, 1).clone()
                rm = self.ref_masks.repeat(b, 1)
                ri = self.ref_idf.repeat(b, 1)
                _, R, _ = greedy_cos_idf(re_, rm, ri, he, hm, hi)
                R = (R.float().cpu().numpy() - self.baseline) / (1 - self.baseline)
            out[idx] = R.reshape(b, self.n_ref)
            if progress:
                progress(min(start + b, len(keep)), len(keep))
        if clip_negative:
            out[out < 0] = 0.0
        return out


def validate_against(published_csv=None, scorer=None, n=None, seed=8521,
                     text_col="Twitter_processed"):
    """
    Rescore posts the 2025 run already scored, and compare.

    Defaults to `assets/validation_sample.csv`, 200 posts with the recall values the
    published run recorded for them. Bundling the sample means the check runs anywhere
    — including on Colab — without shipping the 215 MB results file. Pass a path to
    check against the full file instead.

    Agreement should be at the level of float32 rounding. It is also worth running on
    each machine the scoring is done on: CUDA and Apple's MPS do not have to produce
    bit-identical results.
    """
    if published_csv is None:
        here = os.path.dirname(os.path.abspath(__file__))
        published_csv = os.path.join(here, "assets", "validation_sample.csv")
    df = pd.read_csv(published_csv, low_memory=False)
    if n is not None and n < len(df):
        df = df.sample(n, random_state=seed)
    cols = [f"Topic_{i}" for i in range(1, 39)]
    got = scorer.score(df[text_col].fillna("").tolist())
    want = df[cols].to_numpy(dtype=np.float32)
    diff = np.abs(got - want)
    return {
        "n_posts": len(df),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "correlation": float(np.corrcoef(got.ravel(), want.ravel())[0, 1]),
        "share_within_1e-4": float((diff < 1e-4).mean()),
    }
