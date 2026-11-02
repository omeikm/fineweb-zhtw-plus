"""
rule_flagger.py  (放在 custom_filter.py 旁邊即可)

用途:
  1. RuleFlagger          : 把原本「命中就丟」的 filter 包起來,改成只記錄 T/F,文件不會被丟掉。
                            T = 觸發該規則(原本會被丟掉), F = 未觸發。
                            結果同時寫進 doc.metadata(給下游使用)與每個 rank 一份 CSV 表格。
  2. KeywordContextLogger : 對指定的 regex,把「實際命中的關鍵字」和「關鍵字所在的句子」記錄下來。
  3. merge_csv_dir / summarize_keyword_hits : 把每個 rank 的 CSV 合併成單一檔案、統計每個關鍵字。
"""
import csv
import os
import re
from collections import defaultdict
from glob import glob

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep

# ----------------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------------
SENT_DELIMS = "。！？!?；;\n"


def _tf(value):
    """True -> 'T', False -> 'F', None -> ''(該欄位沒有資料)"""
    if value is None:
        return ""
    return "T" if value else "F"


def extract_sentence(text: str, start: int, end: int, max_len: int = 200) -> str:
    """
    取出 text[start:end] 所在的句子,並用【】標出關鍵字。
    句子邊界: 。！？!?；; 與換行。若句子太長(沒標點的網頁文字很常見),
    只保留關鍵字前後各 max_len/2 個字元。
    """
    left = max(text.rfind(d, 0, start) for d in SENT_DELIMS) + 1  # rfind 找不到回傳 -1 -> left=0
    rights = [i for i in (text.find(d, end) for d in SENT_DELIMS) if i != -1]
    right = min(rights) + 1 if rights else len(text)

    prefix = suffix = ""
    if right - left > max_len:
        half = max_len // 2
        if start - half > left:
            left = start - half
            prefix = "…"
        if end + half < right:
            right = end + half
            suffix = "…"

    sentence = f"{prefix}{text[left:start]}【{text[start:end]}】{text[end:right]}{suffix}"
    return sentence.replace("\n", " ").strip()


# ----------------------------------------------------------------------------
# 1. 規則旗標: 只記錄,不丟棄
# ----------------------------------------------------------------------------
class RuleFlagger(PipelineStep):
    """
    rules      : [(欄位名稱, filter 物件), ...]  filter 就是原本會丟文件的那個 datatrove filter,
                 這裡直接呼叫它的 .filter(doc),所以判斷邏輯與原本完全一致。
    columns    : 最終表格的所有規則欄位(依序)。不在 rules 裡的欄位會從 doc.metadata 取,
                 讓下一個 stage 可以把上一個 stage 的旗標一起寫進同一張表。
    table_dir  : 每個 rank 的 CSV 會寫在這裡 (00000.csv, 00001.csv, ...)
    extra_cols : 額外從 doc.metadata 帶進表格的欄位(找不到就留空)
    """

    type = "🏷️ - FLAGGER"
    name = "🏷️ Rule flagger"

    def __init__(self, rules, columns, table_dir, extra_cols=("url", "language", "language_score")):
        super().__init__()
        self.rules = list(rules)
        self.columns = list(columns)
        self.table_dir = table_dir
        self.extra_cols = list(extra_cols)

    @staticmethod
    def _triggered(flt, doc) -> bool:
        result = flt.filter(doc)
        if isinstance(result, tuple):  # 新版 datatrove 可能回傳 (bool, reason)
            result = result[0]
        return not result  # filter() 回傳 True 代表保留 => 不觸發

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        os.makedirs(self.table_dir, exist_ok=True)
        path = os.path.join(self.table_dir, f"{rank:05d}.csv")
        header = ["id", *self.extra_cols, *self.columns, "any_triggered"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for doc in data:
                with self.track_time():
                    for name, flt in self.rules:
                        doc.metadata[name] = self._triggered(flt, doc)
                    flags = [doc.metadata.get(c) for c in self.columns]
                    any_triggered = any(v is True for v in flags)
                    extras = [doc.metadata.get(c, "") for c in self.extra_cols]
                    writer.writerow([doc.id, *extras, *[_tf(v) for v in flags], _tf(any_triggered)])
                    self.stat_update("docs")
                    if any_triggered:
                        self.stat_update("any_triggered")
                yield doc


# ----------------------------------------------------------------------------
# 2. 關鍵字 + 所在句子
# ----------------------------------------------------------------------------
class KeywordContextLogger(PipelineStep):
    """
    patterns : [(規則名稱, regex 字串), ...]
    每個「文件 x 關鍵字」最多記 max_examples 句(用 example_idx 編號),
    n_occurrences_in_doc 是該關鍵字在這份文件中的總出現次數。
    """

    type = "🏷️ - FLAGGER"
    name = "🔎 Keyword context logger"

    def __init__(self, patterns, out_dir, max_examples=3, max_len=200):
        super().__init__()
        self.patterns = [(name, re.compile(p)) for name, p in patterns]
        self.out_dir = out_dir
        self.max_examples = max_examples
        self.max_len = max_len

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"{rank:05d}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["id", "url", "rule", "keyword", "n_occurrences_in_doc", "example_idx", "start", "sentence"]
            )
            for doc in data:
                with self.track_time():
                    text = doc.text
                    for rule_name, regex in self.patterns:
                        hits = defaultdict(list)  # keyword -> [(start, end), ...]
                        for m in regex.finditer(text):
                            hits[m.group(0)].append((m.start(), m.end()))
                        for keyword, spans in hits.items():
                            for idx, (s, e) in enumerate(spans[: self.max_examples], start=1):
                                writer.writerow(
                                    [
                                        doc.id,
                                        doc.metadata.get("url", ""),
                                        rule_name,
                                        keyword,
                                        len(spans),
                                        idx,
                                        s,
                                        extract_sentence(text, s, e, self.max_len),
                                    ]
                                )
                            self.stat_update(f"{rule_name}:{keyword}")
                yield doc


# ----------------------------------------------------------------------------
# 3. 合併 / 統計(executor.run() 結束後在主程序呼叫)
# ----------------------------------------------------------------------------
def merge_csv_dir(src_dir: str, dst_path: str) -> int:
    """把 src_dir 底下每個 rank 的 CSV 合併成一個檔案(utf-8-sig,Excel 直接開不會亂碼)。回傳資料列數。"""
    files = sorted(glob(os.path.join(src_dir, "*.csv")))
    n_rows = 0
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w", newline="", encoding="utf-8-sig") as out:
        writer = csv.writer(out)
        header_written = False
        for fp in files:
            with open(fp, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)
                    n_rows += 1
    return n_rows


def summarize_keyword_hits(hits_csv: str, out_csv: str) -> None:
    """每個 (規則, 關鍵字) 有幾份文件命中、總共出現幾次,依文件數由高到低排序。"""
    n_docs = defaultdict(int)
    n_hits = defaultdict(int)
    with open(hits_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["example_idx"] == "1":  # 每個 (文件, 關鍵字) 只算一次
                key = (row["rule"], row["keyword"])
                n_docs[key] += 1
                n_hits[key] += int(row["n_occurrences_in_doc"])
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as out:
        writer = csv.writer(out)
        writer.writerow(["rule", "keyword", "n_docs", "n_total_occurrences"])
        for key in sorted(n_docs, key=lambda k: -n_docs[k]):
            writer.writerow([key[0], key[1], n_docs[key], n_hits[key]])