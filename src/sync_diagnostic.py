"""ACC lag screening only; no automatic realignment or verification claim."""
import numpy as np
from scipy.signal import resample_poly, correlate, correlation_lags
from fractions import Fraction

def estimate(record, target_hz=8, max_lag_seconds=15, anomaly_seconds=2, segment_seconds=300):
    streams=[]
    for site,fs in [('chest',700),('wrist',32)]:
        acc=np.asarray(record['signal'][site]['ACC'],float)
        if acc.ndim != 2 or acc.shape[1]!=3 or not np.isfinite(acc).all():
            return [dict(subject_id=record['subject_id'],status='invalid_ACC')]
        magnitude=np.linalg.norm(acc,axis=1)
        ratio=Fraction(target_hz,fs)
        streams.append(resample_poly(magnitude,ratio.numerator,ratio.denominator))
    n=min(map(len,streams));length=int(segment_seconds*target_hz);rows=[]
    for start in range(0,n-length+1,length):
        x,y=[s[start:start+length] for s in streams]
        x=x-x.mean();y=y-y.mean()
        norm=np.linalg.norm(x)*np.linalg.norm(y)
        if norm<1e-10:
            rows.append(dict(subject_id=record['subject_id'],start_seconds=start/target_hz,status='uninformative_flat'))
            continue
        # correlate(wrist,chest): positive lag means wrist pattern occurs later.
        corr=correlate(y,x,mode='full',method='fft')/norm
        lags=correlation_lags(len(y),len(x),mode='full')
        mask=np.abs(lags)<=max_lag_seconds*target_hz
        cc,ll=corr[mask],lags[mask]
        best=int(np.argmax(cc));lag=float(ll[best]/target_hz);peak=float(cc[best])
        rows.append(dict(subject_id=record['subject_id'],start_seconds=start/target_hz,
            lag_seconds=lag,peak_correlation=peak,status='screening_only',
            anomalous_lag=abs(lag)>anomaly_seconds,weak_peak=peak<.3,
            boundary_peak=abs(lag)>=max_lag_seconds))
    return rows
