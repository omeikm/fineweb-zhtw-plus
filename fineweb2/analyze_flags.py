"""
analyze_flags.py

輸入(都是 5_zhtwplus 產生的):
  --flags  rule_flags_<DUMP>.csv      每篇文件一列,五條規則的 T/F
  --hits   keyword_hits_<DUMP>.csv    關鍵字 + 所在句子

輸出(--outdir):
  rule_table.csv           id, rule1..rule4 (0/1)。例如只有第 4 條被觸發 -> 0 0 0 1
  rule_pattern_counts.csv  每種觸發組合(如 0001)有幾份文件
  keyword_impact.csv       規則 4 每個關鍵字的影響(含「拿掉這個詞能救回幾份文件」)
  annotation_sample.csv    每個高影響關鍵字抽樣的例句,留空的 label 欄位供人工標註

規則對應:
  rule1 = neg1_notlang        (FastText zh 不是 top1)
  rule2 = neg1_simplified     (含簡體專有字)
  rule3 = neg1_notraditional  (沒有任何繁體專有字)
  rule4 = neg1_zhtwfine       (命中維護的用語詞表)
  (neg1_plus18 不在這 4 條裡,但計算「救回」時會當作其他規則,避免把本來就會被丟的文件算進來)

用法:
  python analyze_flags.py --flags .../5_zhtwplus/rule_flags_X.csv \
                          --hits  .../5_zhtwplus/keyword_hits_X.csv \
                          --outdir analysis --topk 30 --n_samples 30
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict

RULES = ["neg1_notlang", "neg1_simplified", "neg1_notraditional", "neg1_zhtwfine"]
RULE4 = "neg1_zhtwfine"
PLUS18 = "neg1_plus18"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def tf(v):
    """'T' -> 1, 'F' -> 0, 其他(空白=沒有資料) -> None"""
    return 1 if v == "T" else 0 if v == "F" else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", required=True)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--outdir", default="analysis")
    ap.add_argument("--topk", type=int, default=30, help="抽樣標註的關鍵字數(依規則 4 獨殺文件數排序)")
    ap.add_argument("--n_samples", type=int, default=30, help="每個關鍵字抽幾句")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    # ------------------------------------------------------------------
    # Pass 1: 規則表 + 組合統計
    # ------------------------------------------------------------------
    pattern_counts = Counter()
    rule4_only = set()  # 只有規則 4 被觸發(且 plus18 也沒觸發)的文件
    n_docs = n_missing = 0
    with open(args.flags, newline="", encoding="utf-8-sig") as f, open(
        os.path.join(args.outdir, "rule_table.csv"), "w", newline="", encoding="utf-8"
    ) as out:
        reader = csv.DictReader(f)
        missing_cols = [c for c in RULES if c not in (reader.fieldnames or [])]
        if missing_cols:
            sys.exit(f"flags 檔缺少欄位: {missing_cols}")
        w = csv.writer(out)
        w.writerow(["id", "rule1", "rule2", "rule3", "rule4"])
        for row in reader:
            vals = [tf(row[c]) for c in RULES]
            if any(v is None for v in vals):
                n_missing += 1
            bits = [v or 0 for v in vals]
            w.writerow([row["id"], *bits])
            pattern_counts["".join(map(str, bits))] += 1
            n_docs += 1
            if bits == [0, 0, 0, 1] and tf(row.get(PLUS18, "")) != 1:
                rule4_only.add(row["id"])

    with open(os.path.join(args.outdir, "rule_pattern_counts.csv"), "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["pattern(rule1..rule4)", "n_docs", "ratio"])
        for pat, n in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            w.writerow([pat, n, f"{n / max(n_docs, 1):.6f}"])

    print(f"[pass1] docs={n_docs}, rule4-only docs={len(rule4_only)}")
    if n_missing:
        print(f"[warn] {n_missing} 份文件的規則欄位有空白(視為 0),請確認 stage 1 的旗標有被帶進 stage 5")

    # ------------------------------------------------------------------
    # Pass 2: 關鍵字影響
    # ------------------------------------------------------------------
    kw_docs_all = defaultdict(int)  # keyword -> 命中文件數
    doc_keywords = defaultdict(set)  # 只存 rule4-only 文件的規則 4 關鍵字集合
    with open(args.hits, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["rule"] != RULE4 or row["example_idx"] != "1":
                continue
            kw_docs_all[row["keyword"]] += 1
            if row["id"] in rule4_only:
                doc_keywords[row["id"]].add(row["keyword"])

    kw_rule4_only = defaultdict(int)  # 命中該詞、且只被規則 4 殺掉的文件數
    kw_saved = defaultdict(int)  # 該詞是文件唯一的命中詞 -> 拿掉這個詞就能救回
    for doc_id, kws in doc_keywords.items():
        for k in kws:
            kw_rule4_only[k] += 1
        if len(kws) == 1:
            kw_saved[next(iter(kws))] += 1

    total_rule4_only = max(len(rule4_only), 1)
    ranked = sorted(kw_docs_all, key=lambda k: (-kw_saved[k], -kw_rule4_only[k], -kw_docs_all[k]))
    with open(os.path.join(args.outdir, "keyword_impact.csv"), "w", newline="", encoding="utf-8-sig") as out:
        w = csv.writer(out)
        w.writerow(["keyword", "n_docs_hit", "n_docs_rule4_only", "n_docs_saved_if_removed", "pct_of_rule4_only_docs"])
        for k in ranked:
            w.writerow([k, kw_docs_all[k], kw_rule4_only[k], kw_saved[k], f"{kw_saved[k] / total_rule4_only:.4f}"])

    # ------------------------------------------------------------------
    # Pass 3: 抽樣例句供人工標註(reservoir sampling,只看 rule4-only 文件)
    # ------------------------------------------------------------------
    top_keywords = set(ranked[: args.topk])
    reservoirs = {k: [] for k in top_keywords}
    seen = Counter()
    with open(args.hits, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            k = row["keyword"]
            if row["rule"] != RULE4 or k not in top_keywords or row["id"] not in rule4_only:
                continue
            seen[k] += 1
            item = (k, row["id"], row["url"], row["sentence"])
            if len(reservoirs[k]) < args.n_samples:
                reservoirs[k].append(item)
            else:
                j = rng.randint(0, seen[k] - 1)
                if j < args.n_samples:
                    reservoirs[k][j] = item

    with open(os.path.join(args.outdir, "annotation_sample.csv"), "w", newline="", encoding="utf-8-sig") as out:
        w = csv.writer(out)
        w.writerow(["keyword", "id", "url", "sentence", "label(1=該丟 0=誤殺)", "note"])
        for k in ranked[: args.topk]:
            for item in reservoirs[k]:
                w.writerow([*item, "", ""])

    print(f"[done] 輸出在 {args.outdir}/")


if __name__ == "__main__":
    main()