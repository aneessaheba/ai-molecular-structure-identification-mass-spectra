"""Model 1 (Yash): analog + modification enumeration for novel (Class 3) molecules.

Idea in one line: natural products come in families, so an unknown molecule is often a known
library molecule plus or minus one small chemical edit (+O, +CH2, -CH2, +H2, -H2, +/- a hexose).

For one query molecule:
  1. M = neutral mass from the precursor m/z and adduct (median over its spectra).
  2. For every edit with mass change d: the parent ("analog") must weigh M - d. Look up library
     structures at M - d (15 ppm), score each by spectral cosine against the query spectra and
     keep the best few. We never look up mass M itself, so the true structure is never pulled
     back in from the library (that would fake the hide_true score).
  3. Apply the edit at every possible site with an RDKit reaction, sanitise, and keep products
     whose exact mass is within 15 ppm of M.
  4. Return up to 200 unique products from extra_candidates(). score() gives each product its
     parent's cosine (times gen_weight); library candidates keep the baseline cosine.

Run from the repo root:
    python -m members.YashShevkar30.model_analog_enum --hide_true          # E701
    python -m members.YashShevkar30.model_analog_enum                      # Class-1 check
    python -m members.YashShevkar30.model_analog_enum --hide_true --limit 20   # quick debug
"""
import argparse
import json
import os
import time

import numpy as np

from src.data.spectra import cosine, neutral_mass
from src.evaluation.metric import smiles_to_ik14
from src.models.base import Scorer

VAL_LIB = "enveda-np-examples"
HEXOSE = "C1OC(CO)C(O)C(O)C1O"

# name -> (mass change product - parent in Da, list of reaction SMARTS parent>>product)
MODIFICATIONS = {
    "hydroxylation": (15.994915, ["[cH:1]>>[c:1]O"]),
    "methylation": (14.015650, ["[OX2H1:1]>>[O:1]C",
                                "[NX3;H1,H2;!$(N-[!#6;!#1]):1]>>[N:1]C"]),
    "demethylation": (-14.015650, ["[O:1]-[CH3]>>[O:1]", "[N:1]-[CH3]>>[N:1]"]),
    "hydrogenation": (2.015650, ["[C:1]=[C:2]>>[C:1]-[C:2]"]),
    "dehydrogenation": (-2.015650, ["[CX4;!H0:1]-[CX4;!H0:2]>>[C:1]=[C:2]"]),
    "glycosylation": (162.052824, ["[OX2H1:1]>>[O:1]" + HEXOSE]),
    "deglycosylation": (-162.052824, [
        "[*:2]-[O:1]-[CH1;R]1-O-[CH1](-[CH2]-[OH])-[CH1](-[OH])-[CH1](-[OH])-[CH1]1-[OH]>>[*:2]-[O:1]"]),
}


def query_mass(spectra):
    masses = [neutral_mass(s["precursor_mz"], s["adduct"]) for s in spectra]
    masses = [m for m in masses if np.isfinite(m)]
    return float(np.median(masses)) if masses else np.nan


def spectral_score(lib, spectra, struct_idx, max_ref=20):
    """Same rule as the baseline: mean over query spectra of the best cosine vs the structure's
    reference spectra in the same ionisation mode."""
    refs = list(lib.spectra_of(struct_idx))[:max_ref]
    per_query = []
    for q in spectra:
        best = 0.0
        for j in refs:
            if lib.mode[j] != q["ionization_mode"]:
                continue
            mz_r, in_r = lib.peaks(j)
            best = max(best, cosine(q["mz"], q["intensity"], mz_r, in_r))
        per_query.append(best)
    return float(np.mean(per_query)) if per_query else 0.0


class AnalogEnumScorer(Scorer):
    name = "analog-enum"

    def __init__(self, ref_dir="data/processed", exclude_libs=(VAL_LIB,), ppm=15.0,
                 analogs_per_mod=3, max_products=200, gen_weight=1.0, mods=None,
                 gate=None, gated_weight=0.3):
        from rdkit import RDLogger
        from rdkit.Chem import AllChem

        from src.models.library_search import ReferenceLibrary

        RDLogger.DisableLog("rdApp.*")
        # Validation: exclude the validation library, exactly like src.evaluation.validate does.
        # Kaggle: pass exclude_libs=() to use every library.
        self.lib = ReferenceLibrary(ref_dir, exclude_libs=list(exclude_libs))
        self.ppm = ppm
        self.analogs_per_mod = analogs_per_mod
        self.max_products = max_products
        self.gen_weight = gen_weight
        # Confidence gate: if the best library candidate already matches well (cosine >= gate),
        # the molecule is probably in the library, so generated products get gated_weight instead.
        self.gate = gate
        self.gated_weight = gated_weight
        mods = mods or list(MODIFICATIONS)
        self.rxns = {m: (MODIFICATIONS[m][0], [AllChem.ReactionFromSmarts(s) for s in MODIFICATIONS[m][1]])
                     for m in mods}
        self._gen = {}      # inchikey14 -> score for the current molecule's products
        self.stats = []     # one dict per molecule, aligned with evaluate()'s molecule order

    # ---- generation ----
    def _apply(self, rxns, parent_smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(parent_smiles)
        if mol is None:
            return [], 0
        out, n_raw = [], 0
        for rx in rxns:
            for prods in rx.RunReactants((mol,)):
                n_raw += 1
                p = Chem.Mol(prods[0])
                try:
                    Chem.SanitizeMol(p)
                except Exception:
                    continue
                out.append(p)
        return out, n_raw

    def extra_candidates(self, spectra):
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

        t0 = time.time()
        self._gen = {}
        st = {"n_parents": 0, "n_raw": 0, "n_valid": 0, "n_in_mass": 0, "n_unique": 0}
        M = query_mass(spectra)
        if not np.isfinite(M):
            self.stats.append({**st, "keys": [], "seconds": 0.0})
            return []
        best = {}   # inchikey14 -> (score, smiles, mod)
        for mod, (delta, rxns) in self.rxns.items():
            parents = self.lib.candidates(M - delta, self.ppm)
            if parents.size == 0:
                continue
            scored = sorted(((spectral_score(self.lib, spectra, s), s) for s in parents), reverse=True)
            for p_score, s in scored[: self.analogs_per_mod]:
                st["n_parents"] += 1
                prods, n_raw = self._apply(rxns, self.lib.structs.smiles.values[s])
                st["n_raw"] += n_raw
                st["n_valid"] += len(prods)
                for p in prods:
                    if abs(CalcExactMolWt(p) - M) > M * self.ppm * 1e-6:
                        continue
                    st["n_in_mass"] += 1
                    smi = Chem.MolToSmiles(p)
                    key = smiles_to_ik14(smi, tautomer=False)  # same key validate.py dedupes on
                    if key and (key not in best or p_score > best[key][0]):
                        best[key] = (p_score, smi, mod)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[: self.max_products]
        self._gen = {k: v[0] for k, v in ranked}
        st["n_unique"] = len(ranked)
        self.stats.append({**st, "keys": [k for k, _ in ranked], "mods": [v[2] for _, v in ranked],
                           "seconds": time.time() - t0})
        return [v[1] for _, v in ranked]

    # ---- scoring ----
    def score(self, spectra, candidates):
        scores = candidates.cosine.values.astype(float).copy()
        gen = candidates.struct_idx.values == -1
        w = self.gen_weight
        if self.gate is not None and (~gen).any() and scores[~gen].max() >= self.gate:
            w = self.gated_weight
        scores[gen] = [w * self._gen.get(k, 0.0) for k in candidates.inchikey14.values[gen]]
        return scores


def summarise(res, model):
    """Generation statistics for the report, joined to evaluate()'s per-molecule table."""
    per = res["per_molecule"].copy()
    st = model.stats[: len(per)]
    for col in ["n_parents", "n_raw", "n_valid", "n_in_mass", "n_unique", "seconds"]:
        per[col] = [s[col] for s in st]
    per["true_generated"] = [m in set(s["keys"]) for m, s in zip(per.molecule, st)]
    per["true_mod"] = [s["mods"][s["keys"].index(m)] if m in s["keys"] else None
                       for m, s in zip(per.molecule, st)]
    summary = {
        "model": res["model"], "mrr": res["mrr"], "top1": res["top1"], "top10": res["top10"],
        "top25": res["top25"], "n_molecules": len(per),
        "mean_products": float(per.n_unique.mean()),
        "valid_rate": float(per.n_valid.sum() / max(per.n_raw.sum(), 1)),
        "in_mass_rate": float(per.n_in_mass.sum() / max(per.n_valid.sum(), 1)),
        "recall_generated": float(per.true_generated.mean()),
        "true_by_mod": per.true_mod.value_counts().to_dict(),
        "sec_per_molecule": float(per.seconds.mean()),
    }
    return per, summary


if __name__ == "__main__":
    from src.evaluation.validate import evaluate

    ap = argparse.ArgumentParser()
    ap.add_argument("--hide_true", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gen_weight", type=float, default=1.0)
    ap.add_argument("--analogs_per_mod", type=int, default=3)
    ap.add_argument("--gate", type=float, default=None)
    ap.add_argument("--tag", default=None, help="experiment ID, e.g. E701; saves outputs")
    args = ap.parse_args()

    np.random.seed(4722)
    model = AnalogEnumScorer(gen_weight=args.gen_weight, analogs_per_mod=args.analogs_per_mod,
                             gate=args.gate)
    res = evaluate(model, hide_true=args.hide_true, limit=args.limit)
    per, summary = summarise(res, model)
    summary.update(hide_true=args.hide_true, gen_weight=args.gen_weight,
                   analogs_per_mod=args.analogs_per_mod, gate=args.gate, limit=args.limit)
    print(json.dumps(summary, indent=1))
    if args.tag:
        out = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(out, exist_ok=True)
        per.to_csv(os.path.join(out, f"{args.tag}_per_molecule.csv"), index=False)
        with open(os.path.join(out, f"{args.tag}_summary.json"), "w") as f:
            json.dump(summary, f, indent=1)
