"""Model 2 (Yash): spectrum -> SELFIES transformer for novel (Class 3) molecules.

Idea in one line: read the peaks of a spectrum with a transformer encoder and write the molecule
out token by token with a transformer decoder (SELFIES, so every token string decodes to a valid
molecule). Beam search proposes up to 25 structures per spectrum; we keep the ones whose exact mass
is within 15 ppm of the query mass M and hand them to the pipeline through extra_candidates().

  encoder input : one token per peak (Fourier features of m/z and neutral loss, plus intensity),
                  plus one "meta" token (adduct one-hot, collision energy, precursor m/z, and
                  Fourier features of the neutral mass M).
  decoder       : 4 layers, causal self-attention + cross-attention to the peak tokens,
                  SELFIES tokens (stereo removed first: the metric ignores stereochemistry).
  training      : teacher forcing on load_training(max_per_structure=2). Validation molecules are
                  never in the training data (load_training removes them); early stopping uses a
                  structure-disjoint split of the training data.
  scoring       : library candidates keep the baseline cosine. A generated structure gets the mean
                  over the molecule's spectra of exp(mean token log-prob) of its best beam (0 for a
                  spectrum that did not produce it), times gen_weight.

Run from the repo root:
    python -m members.YashShevkar30.model_smiles_transformer train --sample 20000 --epochs 1 \
        --out members/YashShevkar30/weights/smoke.pt                                   # smoke test
    python -m members.YashShevkar30.model_smiles_transformer evaluate \
        --weights members/YashShevkar30/weights/model.pt --hide_true --tag E702        # E702
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.spectra import neutral_mass
from src.evaluation.metric import smiles_to_ik14
from src.features.featurize import metadata_vector, peak_tensor
from src.models.base import Scorer

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 4722
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]
MAX_PEAKS = 64      # peak_tensor rows (row 0 = precursor)
MAX_TOK = 96        # sequence length including <bos> and <eos>


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- tokenising structures
def smiles_to_tokens(smiles):
    """SMILES -> list of SELFIES tokens (no stereo), or None if RDKit/SELFIES cannot handle it."""
    import selfies as sf
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        flat = Chem.MolToSmiles(mol, isomericSmiles=False)
        return list(sf.split_selfies(sf.encoder(flat)))
    except Exception:
        return None


def tokens_to_smiles(tokens):
    """List of SELFIES tokens -> canonical SMILES (RDKit), or None if invalid."""
    import selfies as sf
    from rdkit import Chem
    try:
        smi = sf.decoder("".join(tokens))
        mol = Chem.MolFromSmiles(smi)
        return None if mol is None else Chem.MolToSmiles(mol)
    except Exception:
        return None


def tokenize_structures(df, max_tok=MAX_TOK):
    """One tokenisation per structure. Returns {inchikey14: tokens} (too-long / failed ones dropped)."""
    out = {}
    uniq = df.drop_duplicates("inchikey14")[["inchikey14", "smiles"]]
    for key, smi in zip(uniq.inchikey14.values, uniq.smiles.values):
        t = smiles_to_tokens(smi)
        if t is not None and 0 < len(t) <= max_tok - 2:
            out[key] = t
    return out


def encode_tokens(token_lists, stoi, max_tok=MAX_TOK):
    arr = np.full((len(token_lists), max_tok), PAD, np.int16)
    for i, t in enumerate(token_lists):
        ids = [BOS] + [stoi.get(x, UNK) for x in t] + [EOS]
        arr[i, :len(ids)] = ids
    return arr


# ---------------------------------------------------------------- spectra -> arrays
def spectra_to_arrays(mz_list, int_list, precursor, adducts, ces):
    """Peak tensors (N, 64, 3), masks (N, 64), metadata (N, 12), neutral masses (N,)."""
    n = len(mz_list)
    peaks = np.zeros((n, MAX_PEAKS, 3), np.float32)
    mask = np.zeros((n, MAX_PEAKS), bool)
    meta = np.zeros((n, 12), np.float32)
    mass = np.zeros(n, np.float32)
    for i in range(n):
        peaks[i], mask[i] = peak_tensor(mz_list[i], int_list[i], precursor[i], MAX_PEAKS)
        meta[i] = metadata_vector(adducts[i], ces[i], precursor[i])
        m = neutral_mass(precursor[i], adducts[i])
        mass[i] = m if np.isfinite(m) else precursor[i]
    return peaks, mask, meta, mass


# ---------------------------------------------------------------- model
class Fourier(nn.Module):
    """sin/cos features of a scalar (m/z or mass) at log-spaced periods from 0.05 Da to 2000 Da,
    so the network can read masses to a fraction of a Da."""

    def __init__(self, n_freq=32, min_period=0.05, max_period=2000.0):
        super().__init__()
        periods = torch.logspace(math.log10(min_period), math.log10(max_period), n_freq)
        self.register_buffer("w", 2 * math.pi / periods)
        self.dim = 2 * n_freq

    def forward(self, x):
        a = x.unsqueeze(-1) * self.w
        return torch.cat([a.sin(), a.cos()], -1)


class Spec2Selfies(nn.Module):
    def __init__(self, vocab_size, d=256, heads=8, layers=4, ff=1024, drop=0.1):
        super().__init__()
        self.d = d
        self.four = Fourier()
        self.peak_in = nn.Linear(2 * self.four.dim + 1, d)
        self.meta_in = nn.Linear(12 + self.four.dim, d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d, heads, ff, drop, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, layers, norm=nn.LayerNorm(d))
        self.tok = nn.Embedding(vocab_size, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_TOK, d)
        self.out = nn.Linear(d, vocab_size)

    def encode(self, peaks, mask, meta, mass):
        x = torch.cat([self.four(peaks[..., 0]), self.four(peaks[..., 2]), peaks[..., 1:2]], -1)
        x = self.peak_in(x)
        m = self.meta_in(torch.cat([meta, self.four(mass)], -1)).unsqueeze(1)
        x = torch.cat([m, x], 1)
        valid = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        return self.enc(x, src_key_padding_mask=~valid), ~valid

    def decode(self, tgt, memory, mem_pad):
        t = tgt.size(1)
        causal = torch.triu(torch.ones(t, t, dtype=torch.bool, device=tgt.device), 1)
        h = self.tok(tgt) * math.sqrt(self.d) + self.pos(torch.arange(t, device=tgt.device))
        h = self.dec(h, memory, tgt_mask=causal, memory_key_padding_mask=mem_pad)
        return self.out(h)

    def forward(self, peaks, mask, meta, mass, tgt_in):
        memory, mem_pad = self.encode(peaks, mask, meta, mass)
        return self.decode(tgt_in, memory, mem_pad)


@torch.no_grad()
def beam_search(model, peaks, mask, meta, mass, beam=25, max_len=MAX_TOK - 1):
    """Plain beam search (sum of log-probs). Returns token ids (S, beam, T) and summed log-probs
    (S, beam). Finished beams are padded with <pad> and keep their score."""
    memory, mem_pad = model.encode(peaks, mask, meta, mass)
    s = memory.size(0)
    memory = memory.repeat_interleave(beam, 0)
    mem_pad = mem_pad.repeat_interleave(beam, 0)
    dev = memory.device
    seqs = torch.full((s * beam, 1), BOS, dtype=torch.long, device=dev)
    scores = torch.zeros(s, beam, device=dev)
    scores[:, 1:] = -1e9                     # only one live beam at the start
    done = torch.zeros(s * beam, dtype=torch.bool, device=dev)
    base = (torch.arange(s, device=dev) * beam).unsqueeze(1)
    for _ in range(max_len):
        logp = F.log_softmax(model.decode(seqs, memory, mem_pad)[:, -1].float(), -1)
        logp[:, PAD] = -1e9
        logp[:, BOS] = -1e9
        logp[done] = -1e9
        logp[done, PAD] = 0.0                # a finished beam only extends with <pad>, at no cost
        cand = (scores.view(-1, 1) + logp).view(s, -1)
        top, idx = cand.topk(beam, dim=1)
        v = logp.size(1)
        sel = (base + idx // v).view(-1)
        tok = (idx % v).view(-1)
        seqs = torch.cat([seqs[sel], tok.unsqueeze(1)], 1)
        done = done[sel] | (tok == EOS)
        scores = top
        if bool(done.all()):
            break
    return seqs[:, 1:].view(s, beam, -1).cpu().numpy(), scores.cpu().numpy()


# ---------------------------------------------------------------- training
def save_ckpt(path, model, itos, cfg, extra=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "itos": itos, "cfg": cfg, **(extra or {})}, path)


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = Spec2Selfies(len(ck["itos"]), d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"],
                         ff=cfg["ff"], drop=cfg["drop"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck["itos"]


def train(args):
    from src.data.dataset import load_training, split_by_structure

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = pick_device()
    t0 = time.time()
    df = load_training(train_path=args.train_path, max_per_structure=args.max_per_structure,
                       sample=args.sample, seed=SEED)
    print(f"loaded {len(df):,} spectra / {df.inchikey14.nunique():,} structures in {time.time() - t0:.0f}s", flush=True)

    tok_map = tokenize_structures(df)
    df = df[df.inchikey14.isin(tok_map)].reset_index(drop=True)
    itos = SPECIALS + sorted({t for ts in tok_map.values() for t in ts})
    stoi = {t: i for i, t in enumerate(itos)}
    print(f"{len(tok_map):,} structures tokenised, vocab {len(itos)}, {len(df):,} spectra kept "
          f"({time.time() - t0:.0f}s)", flush=True)

    tr, va = split_by_structure(df, val_frac=args.val_frac, seed=SEED)

    def pack(d):
        peaks, mask, meta, mass = spectra_to_arrays(
            d.mz.values, d.intensity.values, d.precursor_mz.values, d.adduct.values, d.collision_energy.values)
        tgt = encode_tokens([tok_map[k] for k in d.inchikey14.values], stoi)
        return [torch.from_numpy(a) for a in (peaks, mask, meta, mass, tgt)]

    TR, VA = pack(tr), pack(va)
    print(f"train {len(tr):,} spectra, val {len(va):,} spectra ({time.time() - t0:.0f}s)", flush=True)

    cfg = dict(d=args.d, heads=args.heads, layers=args.layers, ff=args.ff, drop=args.drop)
    model = Spec2Selfies(len(itos), **cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr) / args.batch)
    total = steps_per_epoch * args.epochs
    warm = min(args.warmup, max(total // 10, 1))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0)))))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"device {dev}, {n_params / 1e6:.1f}M params, {steps_per_epoch} steps/epoch, "
          f"batch {args.batch}, lr {args.lr}", flush=True)

    def batch_loss(T, ix):
        peaks, mask, meta, mass, tgt = (x[ix].to(dev) for x in T)
        tgt = tgt.long()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            logits = model(peaks, mask, meta, mass, tgt[:, :-1])
        gold = tgt[:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), gold.reshape(-1),
                               ignore_index=PAD, label_smoothing=args.label_smoothing)
        keep = gold != PAD
        acc = ((logits.argmax(-1) == gold) & keep).sum().item() / keep.sum().item()
        return loss, acc

    @torch.no_grad()
    def validate():
        model.eval()
        ls, ac, n = 0.0, 0.0, 0
        for i in range(0, len(VA[0]), 512):
            ix = torch.arange(i, min(i + 512, len(VA[0])))
            loss, acc = batch_loss(VA, ix)
            ls += loss.item() * len(ix)
            ac += acc * len(ix)
            n += len(ix)
        model.train()
        return ls / n, ac / n

    history, best, bad = [], 1e9, 0
    model.train()
    for ep in range(args.epochs):
        perm = torch.randperm(len(TR[0]))
        run_l, run_a, run_n, te = 0.0, 0.0, 0, time.time()
        for b in range(steps_per_epoch):
            ix = perm[b * args.batch:(b + 1) * args.batch]
            loss, acc = batch_loss(TR, ix)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_l += loss.item() * len(ix)
            run_a += acc * len(ix)
            run_n += len(ix)
            if (b + 1) % args.log_every == 0:
                print(f"  ep {ep + 1} step {b + 1}/{steps_per_epoch} loss {run_l / run_n:.4f} "
                      f"acc {run_a / run_n:.3f} {time.time() - te:.0f}s", flush=True)
        vl, vacc = validate()
        row = dict(epoch=ep + 1, train_loss=run_l / run_n, train_token_acc=run_a / run_n,
                   val_loss=vl, val_token_acc=vacc, epoch_seconds=time.time() - te,
                   lr=sched.get_last_lr()[0])
        history.append(row)
        print(json.dumps(row), flush=True)
        if vl < best:
            best, bad = vl, 0
            save_ckpt(args.out, model, itos, cfg, {"epoch": ep + 1, "val_loss": vl})
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping", flush=True)
                break
    info = dict(args=vars(args), n_train_spectra=len(tr), n_val_spectra=len(va), n_structures=len(tok_map),
                vocab_size=len(itos), n_params=n_params, device=str(dev), history=history,
                best_val_loss=best, total_seconds=time.time() - t0)
    hist_path = args.history or os.path.join(HERE, "outputs", "E702_train_history.json")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(info, f, indent=1)
    print(f"best val loss {best:.4f}; checkpoint {args.out}; history {hist_path}", flush=True)


# ---------------------------------------------------------------- scorer
def query_mass(spectra):
    masses = [neutral_mass(s["precursor_mz"], s["adduct"]) for s in spectra]
    masses = [m for m in masses if np.isfinite(m)]
    return float(np.median(masses)) if masses else np.nan


class SmilesTransformerScorer(Scorer):
    name = "spec2selfies"

    def __init__(self, weights, beam=25, ppm=15.0, max_spectra=8, max_products=50,
                 gen_weight=1.0, gate=None, gated_weight=0.3, device=None):
        self.dev = device or pick_device()
        self.model, self.itos = load_ckpt(weights, self.dev)
        self.beam, self.ppm = beam, ppm
        self.max_spectra, self.max_products = max_spectra, max_products
        self.gen_weight, self.gate, self.gated_weight = gen_weight, gate, gated_weight
        self._gen = {}
        self.stats = []

    def extra_candidates(self, spectra):
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

        t0 = time.time()
        self._gen = {}
        st = {"n_spectra": 0, "n_beams": 0, "n_valid": 0, "n_in_mass": 0, "n_unique_valid": 0,
              "n_unique": 0}
        M = query_mass(spectra)
        if not np.isfinite(M):
            self.stats.append({**st, "keys": [], "valid": [], "seconds": 0.0})
            return []
        use = spectra[: self.max_spectra]
        peaks, mask, meta, mass = spectra_to_arrays(
            [s["mz"] for s in use], [s["intensity"] for s in use], [s["precursor_mz"] for s in use],
            [s["adduct"] for s in use], [s["collision_energy"] for s in use])
        mass[:] = M                                    # every spectrum of a molecule shares one M
        ids, logp = beam_search(
            self.model, torch.from_numpy(peaks).to(self.dev), torch.from_numpy(mask).to(self.dev),
            torch.from_numpy(meta).to(self.dev), torch.from_numpy(mass).to(self.dev), beam=self.beam)
        st["n_spectra"] = len(use)
        agg, smi_of, valid_all = {}, {}, {}
        for si in range(ids.shape[0]):
            best_here = {}
            for bi in range(ids.shape[1]):
                toks = []
                for t in ids[si, bi]:
                    if t == EOS or t == PAD:
                        break
                    toks.append(self.itos[t])
                st["n_beams"] += 1
                smi = tokens_to_smiles(toks) if toks else None
                if smi is None:
                    continue
                st["n_valid"] += 1
                key = smiles_to_ik14(smi, tautomer=False)
                if not key:
                    continue
                valid_all[key] = smi
                if abs(CalcExactMolWt(MolFromSmiles(smi)) - M) > M * self.ppm * 1e-6:
                    continue
                st["n_in_mass"] += 1
                s = math.exp(float(logp[si, bi]) / (len(toks) + 1))     # +1: the <eos> token
                if s > best_here.get(key, 0.0):
                    best_here[key] = s
                smi_of[key] = smi
            for key, s in best_here.items():
                agg[key] = agg.get(key, 0.0) + s / len(use)
        ranked = sorted(agg.items(), key=lambda kv: -kv[1])[: self.max_products]
        self._gen = dict(ranked)
        st["n_unique_valid"] = len(valid_all)
        st["n_unique"] = len(ranked)
        self.stats.append({**st, "keys": [k for k, _ in ranked], "valid": list(valid_all.values()),
                           "seconds": time.time() - t0})
        return [smi_of[k] for k, _ in ranked]

    def score(self, spectra, candidates):
        scores = candidates.cosine.values.astype(float).copy()
        gen = candidates.struct_idx.values == -1
        w = self.gen_weight
        if self.gate is not None and (~gen).any() and scores[~gen].max() >= self.gate:
            w = self.gated_weight
        scores[gen] = [w * self._gen.get(k, 0.0) for k in candidates.inchikey14.values[gen]]
        return scores


def summarise(res, model, structs_path="data/processed/structures.parquet"):
    """Generation diagnostics for the failure analysis, joined to evaluate()'s per-molecule table.

    The true structure of each validation molecule is looked up by InChIKey14 in the shared
    processed library table, only to measure how close the generated structures are (formula,
    similarity). It is never used for training or for ranking."""
    import pandas as pd
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    RDLogger.DisableLog("rdApp.*")
    structs = pd.read_parquet(structs_path).set_index("inchikey14")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    per = res["per_molecule"].copy()
    st = model.stats[: len(per)]
    for col in ["n_spectra", "n_beams", "n_valid", "n_in_mass", "n_unique_valid", "n_unique", "seconds"]:
        per[col] = [s[col] for s in st]
    true_gen, formula_hit, best_tan, mass_hit = [], [], [], []
    for mol_key, s in zip(per.molecule, st):
        true_smi = structs.smiles.get(mol_key)
        tm = Chem.MolFromSmiles(true_smi) if true_smi else None
        true_gen.append(mol_key in set(s["keys"]))
        if tm is None:
            formula_hit.append(False)
            best_tan.append(np.nan)
            mass_hit.append(False)
            continue
        tf = CalcMolFormula(tm)
        tfp = gen.GetFingerprint(tm)
        fh, bt = False, 0.0
        for smi in s["valid"]:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            if CalcMolFormula(m) == tf:
                fh = True
            bt = max(bt, DataStructs.TanimotoSimilarity(tfp, gen.GetFingerprint(m)))
        formula_hit.append(fh)
        best_tan.append(bt if s["valid"] else np.nan)
        mass_hit.append(s["n_in_mass"] > 0)
    per["true_generated"] = true_gen
    per["formula_match_any"] = formula_hit
    per["any_in_mass"] = mass_hit
    per["best_tanimoto_to_true"] = best_tan
    summary = {
        "model": res["model"], "mrr": res["mrr"], "top1": res["top1"], "top10": res["top10"],
        "top25": res["top25"], "n_molecules": len(per),
        "mean_beams": float(per.n_beams.mean()),
        "valid_rate": float(per.n_valid.sum() / max(per.n_beams.sum(), 1)),
        "in_mass_rate_of_valid": float(per.n_in_mass.sum() / max(per.n_valid.sum(), 1)),
        "in_mass_rate_of_beams": float(per.n_in_mass.sum() / max(per.n_beams.sum(), 1)),
        "mean_unique_in_mass": float(per.n_unique.mean()),
        "mol_with_any_in_mass": float(per.any_in_mass.mean()),
        "mol_with_formula_match": float(per.formula_match_any.mean()),
        "recall_generated": float(per.true_generated.mean()),
        "mean_best_tanimoto": float(np.nanmean(per.best_tanimoto_to_true)),
        "sec_per_molecule": float(per.seconds.mean()),
    }
    return per, summary


def evaluate_cli(args):
    from src.evaluation.validate import evaluate

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = SmilesTransformerScorer(args.weights, beam=args.beam, gen_weight=args.gen_weight,
                                    gate=args.gate, max_spectra=args.max_spectra)
    res = evaluate(model, hide_true=args.hide_true, limit=args.limit)
    per, summary = summarise(res, model)
    summary.update(hide_true=args.hide_true, gen_weight=args.gen_weight, gate=args.gate,
                   beam=args.beam, max_spectra=args.max_spectra, limit=args.limit,
                   weights=os.path.basename(args.weights))
    print(json.dumps(summary, indent=1))
    if args.tag:
        out = os.path.join(HERE, "outputs")
        os.makedirs(out, exist_ok=True)
        per.to_csv(os.path.join(out, f"{args.tag}_per_molecule.csv"), index=False)
        with open(os.path.join(out, f"{args.tag}_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train_path", default="data/raw/train.parquet")
    t.add_argument("--max_per_structure", type=int, default=2)
    t.add_argument("--sample", type=int, default=None)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--val_frac", type=float, default=0.05)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--label_smoothing", type=float, default=0.0)
    t.add_argument("--d", type=int, default=256)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--ff", type=int, default=1024)
    t.add_argument("--drop", type=float, default=0.1)
    t.add_argument("--log_every", type=int, default=100)
    t.add_argument("--out", default=os.path.join(HERE, "weights", "model.pt"))
    t.add_argument("--history", default=None)
    e = sub.add_parser("evaluate")
    e.add_argument("--weights", default=os.path.join(HERE, "weights", "model.pt"))
    e.add_argument("--hide_true", action="store_true")
    e.add_argument("--limit", type=int, default=None)
    e.add_argument("--beam", type=int, default=25)
    e.add_argument("--max_spectra", type=int, default=8)
    e.add_argument("--gen_weight", type=float, default=1.0)
    e.add_argument("--gate", type=float, default=None)
    e.add_argument("--tag", default=None, help="experiment ID, e.g. E702; saves outputs")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else evaluate_cli(a)
