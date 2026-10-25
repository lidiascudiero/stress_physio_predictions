import re
"""Episode-level descriptive questionnaires. Never broadcast answers to windows."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .wesad_io import read_questionnaire

CONDITIONS={'Base':1,'TSST':2,'Fun':3,'Medi 1':4,'Medi 2':4}
POSITIVE=[0,2,3,5,9,10,12,14,16,17]
NEGATIVE=[1,4,6,7,8,11,13,15,18,19]

def parse_self_reports(path):
    path = Path(path)
    q = read_questionnaire(path)

    def normalize_tag(value):
        return value.strip().rstrip(":").strip().upper()

    def values(tag):
        arrays = []
        for row in q["rows"]:
            if not row or normalize_tag(row[0]) != tag:
                continue

            # Elimina solo i campi vuoti finali del CSV.
            # I vuoti interni conservano la posizione dell'item.
            cells = list(row[1:])
            while cells and not cells[-1].strip():
                cells.pop()

            array = []
            for cell in cells:
                text = cell.strip()
                if text.lower() in {"", "nan", "na", "n/a"}:
                    array.append(np.nan)
                else:
                    array.append(float(text))
            arrays.append(array)
        return arrays

    order_rows = [
        row for row in q["rows"]
        if row and normalize_tag(row[0]) == "# ORDER"
    ]
    if len(order_rows) != 1:
        raise ValueError(f"{path.name}: ORDER assente o duplicato.")

    order = [
        value.strip() for value in order_rows[0][1:]
        if value.strip() in CONDITIONS
    ]

    subject_rows = [
        row for row in q["rows"]
        if row and normalize_tag(row[0]) == "# SUBJ"
    ]
    if len(subject_rows) != 1 or len(subject_rows[0]) < 2:
        raise ValueError(f"{path.name}: identificatore assente o ambiguo.")

    subject = subject_rows[0][1].strip()
    if not re.fullmatch(r"S\d+", subject):
        raise ValueError(f"{path.name}: identificatore non valido {subject!r}.")

    expected_subject = path.name.split("_")[0]
    if subject != expected_subject:
        raise ValueError(
            f"{path.name}: ID interno {subject} diverso da {expected_subject}."
        )

    panas = values("# PANAS")
    stai = values("# STAI")
    sam = values("# DIM") or values("# SAM")
    sssq = values("# SSSQ")

    if not len(order) == len(panas) == len(stai) == len(sam):
        raise ValueError(f"{path.name}: numero episodi incoerente.")

    def check_range(items, low, high, scale, condition):
        a = np.asarray(items, dtype=float)
        invalid = np.isinf(a) | (
            np.isfinite(a) & ((a < low) | (a > high))
        )
        if invalid.any():
            raise ValueError(
                f"{path.name}, {condition}, {scale}: "
                f"risposte fuori intervallo: {a[invalid]}"
            )

    def complete_sum(items):
        # Nessuna somma parziale o imputazione.
        a = np.asarray(items, dtype=float)
        return float(a.sum()) if np.isfinite(a).all() else np.nan

    rows = []

    for i, condition in enumerate(order):
        pa, st, sa = panas[i], stai[i], sam[i]

        if len(pa) not in (24, 26) or len(st) != 6 or len(sa) != 2:
            raise ValueError(
                f"{path.name}, {condition}: numero item inatteso "
                f"PANAS={len(pa)}, STAI={len(st)}, SAM={len(sa)}."
            )

        check_range(pa, 1, 5, "PANAS", condition)
        check_range(st, 1, 4, "STAI", condition)
        check_range(sa, 1, 9, "SAM", condition)

        keyed_stai = [
            5 - st[0], st[1], st[2],
            5 - st[3], st[4], 5 - st[5],
        ]

        row = {
            "subject_id": int(subject[1:]),
            "episode_order": i,
            "condition": condition,
            "label": CONDITIONS[condition],
            "panas_positive_sum": complete_sum(
                [pa[j] for j in POSITIVE]
            ),
            "panas_negative_sum": complete_sum(
                [pa[j] for j in NEGATIVE]
            ),
            "stressed_item": pa[20],
            "frustrated_item": pa[21],
            "happy_item": pa[22],
            "sad_item": pa[-1],
            "stai6_keyed_sum": complete_sum(keyed_stai),
            "sam_valence": sa[0],
            "sam_arousal": sa[1],
            "panas_missing_items": int(np.isnan(pa).sum()),
            "stai_missing_items": int(np.isnan(st).sum()),
            "sam_missing_items": int(np.isnan(sa).sum()),
        }

        if condition == "TSST" and sssq:
            if len(sssq) != 1 or len(sssq[0]) != 6:
                raise ValueError(f"{path.name}: numero item SSSQ inatteso.")

            for j, value in enumerate(sssq[0]):
                row[f"sssq_standalone_item_{j + 1}"] = value

            row["sssq_annoyed_from_panas"] = pa[4]

            if len(pa) == 26:
                row["sssq_angry_from_panas"] = pa[23]
                row["sssq_irritated_from_panas"] = pa[24]

        rows.append(row)

    return pd.DataFrame(rows)


def analyze(root,feature_frame=None):
    paths=sorted(Path(root).rglob('S*_quest.csv'))
    if not paths:raise FileNotFoundError('No subject questionnaires available')
    frame=pd.concat([parse_self_reports(p) for p in paths],ignore_index=True)
    if frame.duplicated(['subject_id','condition']).any():raise ValueError('Duplicate questionnaire episodes')
    scores=['panas_positive_sum','panas_negative_sum','stressed_item','stai6_keyed_sum','sam_valence','sam_arousal']
    summary=frame.groupby('condition')[scores].agg(['count','mean','std']).reset_index()
    summary.columns=['_'.join(x for x in col if x) if isinstance(col,tuple) else col for col in summary.columns]
    correlations=[]
    if feature_frame is not None:
        # Only one report for each of labels 1..3. No timing origin assumptions.
        primary=frame[frame.label.isin([1,2,3])]
        cols=[x for x in ['chest__ECG__mean_hr','wrist__BVP__mean_pulse_rate','chest__EDA__mean','wrist__EDA__mean'] if x in feature_frame]
        if cols:
            medians=feature_frame.groupby(['subject_id','label'])[cols].median().reset_index()
            joined=primary.merge(medians,on=['subject_id','label'],how='inner',validate='one_to_one')
            for label,g in joined.groupby('label'):
                for col in cols:
                    valid=g[[col,'stressed_item']].dropna()
                    if len(valid)>=3 and valid[col].nunique()>1 and valid.stressed_item.nunique()>1:
                        correlations.append({'label':int(label),'feature':col,'target':'stressed_item',
                            'n_subjects':len(valid),'spearman_rho':float(spearmanr(valid[col],valid.stressed_item).statistic),
                            'inference':'exploratory effect only; no p-value or causal interpretation'})
    return frame,summary,pd.DataFrame(correlations,columns=['label','feature','target','n_subjects','spearman_rho','inference'])
