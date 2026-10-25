from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

def confusion(matrix,classes,path):
    fig,ax=plt.subplots();im=ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(len(classes)),yticks=range(len(classes)),xticklabels=classes,yticklabels=classes,xlabel='Predicted',ylabel='True')
    for i in range(len(classes)):
        for j in range(len(classes)):ax.text(j,i,str(matrix[i][j]),ha='center',va='center')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def reliability_plot(table,path):
    fig,ax=plt.subplots();ax.plot([0,1],[0,1],'k--')
    for k,g in table.groupby('class'):
        # Aggregate counts, not an unweighted average of sparsely populated bins.
        g=g.copy();g['p_total']=g.mean_probability*g['count'];g['y_total']=g.observed_frequency*g['count']
        b=g.groupby('bin_low')[['count','p_total','y_total']].sum()
        ax.plot(b.p_total/b['count'],b.y_total/b['count'],'o-',label=str(k))
    ax.set(xlabel='Predicted probability',ylabel='Observed frequency',xlim=(0,1),ylim=(0,1));ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def qc_plot(table,path):
    t=table.assign(failure=~table.valid).pivot_table(index='subject_id',columns=['site','channel'],values='failure',aggfunc='mean')
    fig,ax=plt.subplots(figsize=(10,5));im=ax.imshow(t.fillna(1),vmin=0,vmax=1,cmap='Reds',aspect='auto')
    ax.set(yticks=range(len(t)),yticklabels=t.index,xticks=range(len(t.columns)),xticklabels=['/'.join(x) for x in t.columns])
    plt.setp(ax.get_xticklabels(),rotation=70,ha='right');fig.colorbar(im,ax=ax,label='Excluded fraction')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def signal_example(record,path):
    x=record['signal']['chest']['ECG'][:700*60,0]
    f,p=welch(x,fs=700,nperseg=2048)
    fig,axes=plt.subplots(2,1);axes[0].plot(np.arange(len(x))/700,x);axes[0].set(xlabel='Seconds',ylabel='ECG native units')
    axes[1].semilogy(f,p);axes[1].set(xlim=(0,50),xlabel='Hz',ylabel='PSD')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)
