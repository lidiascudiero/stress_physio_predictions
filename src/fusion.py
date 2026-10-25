from .wesad_io import META

def paired(chest,wrist):
    if chest.window_id.duplicated().any() or wrist.window_id.duplicated().any():
        raise ValueError('Duplicate windows')
    a=chest[chest.valid].drop(columns='valid');b=wrist[wrist.valid].drop(columns='valid')
    merged=a.merge(b,on=META,how='inner',validate='one_to_one')
    merged['valid']=True
    return merged

def comparison_tables(tables,paired_only=True):
    fused=paired(tables['chest'],tables['wrist'])
    out={'early_fusion':fused}
    for site in ['chest','wrist']:
        f=tables[site];f=f[f.valid].copy()
        if paired_only:f=f[f.window_id.isin(fused.window_id)]
        out[site]=f.reset_index(drop=True)
    return out
