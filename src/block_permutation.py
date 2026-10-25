"""Training-only episode-label randomization diagnostic, not an unconditional exact test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from .splits import loso
from .models import tune_fit

def permute_training(frame, rng):
    out = frame.copy()
    for sid, group in frame.groupby('subject_id'):
        episodes = group.groupby('episode_id').agg(label=('label','first'), nlabels=('label','nunique'))
        if not episodes.nlabels.eq(1).all():
            raise ValueError('Episodes must have constant labels')
        labels = rng.permutation(episodes.label.to_numpy())
        mapping = dict(zip(episodes.index, labels))
        mask = out.subject_id.eq(sid)
        out.loc[mask,'label'] = out.loc[mask,'episode_id'].map(mapping).to_numpy()
    return out

def run(frame, c, model='logistic_regression', n_permutations=1000, seed=42, destination=None):
    if n_permutations < 1000:
        raise ValueError('At least 1000 permutations required')
    destination = Path(destination or Path(c['output']['root'])/'results/block_permutation')
    destination.mkdir(parents=True, exist_ok=True)
    signature = __import__('hashlib').sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()).hexdigest()
    from .provenance import code_manifest
    from .config import fingerprint
    spec = dict(model=model,n_permutations=n_permutations,seed=seed,frame_sha256=signature,
                config_sha256=fingerprint(c),code_sha256=code_manifest()[0])
    spec_path = destination/'specification.json'
    if spec_path.exists() and json.loads(spec_path.read_text()) != spec:
        raise ValueError('Checkpoint belongs to another experiment; choose a new directory')
    spec_path.write_text(json.dumps(spec,indent=2),encoding='utf-8')
    cols=[x for x in frame if x.startswith(('chest__','wrist__'))]
    splits=list(loso(frame,(1,2,3)))
    def score(permutation):
        scores=[]
        for fold,(tr,te) in enumerate(splits):
            train,test=frame.iloc[tr].copy(),frame.iloc[te]
            if permutation is not None:
                rng=np.random.default_rng(np.random.SeedSequence([seed,permutation,fold]))
                train=permute_training(train,rng)
            fitted,_,_=tune_fit(train,model,c)
            scores.append(float(balanced_accuracy_score(test.label,fitted.predict(test[cols]))))
        return float(np.mean(scores)),scores
    observed_path=destination/'observed.json'
    if observed_path.exists(): observed=json.loads(observed_path.read_text())
    else:
        mean,folds=score(None);observed=dict(mean=mean,folds=folds)
        observed_path.write_text(json.dumps(observed),encoding='utf-8')
    null_path=destination/'null_distribution.csv'
    rows=pd.read_csv(null_path).to_dict('records') if null_path.exists() else []
    done={int(row['permutation']) for row in rows}
    for b in range(n_permutations):
        if b in done:continue
        mean,_=score(b)
        rows.append(dict(permutation=b,balanced_accuracy=mean))
        pd.DataFrame(rows).to_csv(null_path,index=False)
        print(f'Permutation {b+1}/{n_permutations}: BA={mean:.4f}',flush=True)
    null=pd.DataFrame(rows).sort_values('permutation')
    p=(1+int((null.balanced_accuracy >= observed['mean']).sum()))/(n_permutations+1)
    result=dict(observed_BA=observed['mean'],p_value=p,n_permutations=n_permutations,
        hypothesis='Training episode labels exchangeable within subject, conditional on fixed test labels',
        interpretation='Conditional Monte Carlo randomization diagnostic; not exact population inference',
        limitations=['Episode exchangeability is not established by preserving blocks',
                     'Whole episode labels reassigned; signal order and test labels unchanged',
                     'Episode lengths and resulting class proportions can differ',
                     'Overlapping LOSO training sets remain dependent; order effects can violate exchangeability',
                     'Nested tuning repeated for every permuted training set'])
    (destination/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result,null
