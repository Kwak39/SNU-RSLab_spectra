"""Headless numerical core extracted from RSLab_kNN_Classifier(2).py. See SOURCE_NOTES.md."""
import csv

import os

import queue

import sys

import threading

import traceback

from dataclasses import dataclass

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

VERSION='1.1'

@dataclass
class Dataset:
    axis: np.ndarray
    X: np.ndarray                 # rows=samples, columns=wavenumbers
    sample_ids: list[str]
    sources: list[str]

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        rows=[row for row in csv.reader(stream) if any(cell.strip() for cell in row)]
    if not rows: raise ValueError(f'{Path(path).name}: 빈 CSV입니다.')
    return rows

def floats(values,path,where):
    try: data=np.asarray(values,dtype=float)
    except ValueError as exc: raise ValueError(f'{Path(path).name}: {where}에 숫자가 아닌 값이 있습니다.') from exc
    if not np.isfinite(data).all(): raise ValueError(f'{Path(path).name}: {where}에 NaN/Inf가 있습니다.')
    return data

def load_feature_csv(path):
    """Read shared-axis single or mapping exports. Native paired-axis single AsLS/Range is rejected."""
    path=Path(path);rows=read_csv(path)
    first=rows[0]
    if first[0]=='Raman shift (cm^-1)':
        if len(first)<2: raise ValueError(f'{path.name}: 스펙트럼 열이 없습니다.')
        axis=floats([row[0] for row in rows[1:]],path,'파수축')
        if len(axis)<2 or not np.all(np.diff(axis)>0):
            raise ValueError(f'{path.name}: 파수축은 증가하는 두 점 이상이어야 합니다.')
        values=floats([[row[i] for row in rows[1:]] for i in range(1,len(first))],path,'intensity')
        return Dataset(axis,values,[f'{path.name} | {name}' for name in first[1:]],[str(path)]*len(first[1:]))
    if len(first)>=2 and first[0].endswith(' | wavenumber'):
        raise ValueError(f'{path.name}: 파일별 파수축이 다른 native AsLS/Range CSV입니다. '
                         'ML 비교에는 동일 축이 필요합니다. BLaPE CSV 또는 동일 축으로 별도 정렬한 CSV를 사용하세요.')
    if len(rows)<4 or first[0]!='x coordinate' or rows[1][0]!='y coordinate' or rows[2][0]!='Raman shift (cm^-1)':
        raise ValueError(f'{path.name}: RSLab single/mapping 결과 CSV 형식이 아닙니다.')
    labels=rows[2][1:];axis=floats([row[0] for row in rows[3:]],path,'mapping 파수축')
    if len(axis)<2 or not np.all(np.diff(axis)>0): raise ValueError(f'{path.name}: mapping 파수축이 올바르지 않습니다.')
    values=floats([[row[i] for row in rows[3:]] for i in range(1,len(labels)+1)],path,'mapping intensity')
    return Dataset(axis,values,[f'{path.name} | {label}' for label in labels],[str(path)]*len(labels))

def combine(paths):
    if not paths: raise ValueError('CSV를 하나 이상 선택하세요.')
    datasets=[load_feature_csv(path) for path in paths]
    reference=datasets[0].axis
    for d in datasets[1:]:
        if len(d.axis)!=len(reference) or not np.allclose(d.axis,reference,rtol=1e-9,atol=1e-9):
            raise ValueError('선택한 CSV들의 파수축이 일치하지 않습니다. 같은 범위/edge trim/output points로 전처리하세요.')
    ids=sum((d.sample_ids for d in datasets),[])
    if len(set(ids))!=len(ids): raise ValueError('동일한 sample_id가 중복됩니다. 같은 CSV를 두 번 선택하지 마세요.')
    return Dataset(reference,np.vstack([d.X for d in datasets]),ids,sum((d.sources for d in datasets),[]))

def read_labels(path):
    rows=read_csv(path)
    if rows[0][:2]!=['sample_id','label']:
        raise ValueError('Label CSV의 첫 행은 정확히 sample_id,label 이어야 합니다.')
    result={}
    for line,row in enumerate(rows[1:],2):
        if len(row)<2 or not row[0].strip(): raise ValueError(f'Label CSV {line}행을 확인하세요.')
        if row[0] in result: raise ValueError(f'Label CSV에 중복 sample_id: {row[0]}')
        if row[1].strip(): result[row[0]]=row[1].strip()
    return result

def template(path,datasets):
    ids=[]
    for name,data in datasets:
        ids.extend((name,sample_id) for sample_id in data.sample_ids)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_id','label','dataset'])
        writer.writerows((sample_id,'',name) for name,sample_id in ids)

def labels_for(dataset,label_map,required):
    missing=[sample_id for sample_id in dataset.sample_ids if sample_id not in label_map]
    if required and missing:
        preview='\n'.join(missing[:5])
        suffix='' if len(missing)<=5 else f'\n... 외 {len(missing)-5}개'
        raise ValueError(f'학습 label이 없는 sample_id가 {len(missing)}개입니다:\n{preview}{suffix}')
    if missing:return None
    return np.asarray([label_map[sample_id] for sample_id in dataset.sample_ids])

def make_model(k,preprocess):
    if type(k) is not int or k<1: raise ValueError('k는 1 이상의 정수여야 합니다.')
    steps=[]
    if preprocess in ('L2 normalize','L2 normalize + z-score'):steps.append(('l2',Normalizer(norm='l2')))
    if preprocess in ('z-score per wavenumber','L2 normalize + z-score'):steps.append(('scale',StandardScaler()))
    steps.append(('knn',KNeighborsClassifier(n_neighbors=k,weights='distance',metric='minkowski',p=2)))
    return Pipeline(steps)

def train_predict(train,test,label_map,k,preprocess):
    y_train=labels_for(train,label_map,True)
    if len(np.unique(y_train))<2: raise ValueError('학습 label은 최소 두 클래스가 필요합니다.')
    if k>len(train.X): raise ValueError(f'k={k}는 학습 스펙트럼 수 {len(train.X)}보다 클 수 없습니다.')
    overlap=set(train.sample_ids)&set(test.sample_ids)
    if overlap: raise ValueError('TRAIN과 TEST에 같은 sample_id가 있습니다.')
    model=make_model(k,preprocess);model.fit(train.X,y_train);prediction=model.predict(test.X)
    probabilities=model.predict_proba(test.X)
    classes=model.named_steps['knn'].classes_
    rank_order=np.argsort(-probabilities,axis=1,kind='stable')
    rankings=[]
    for spectrum_index,row in enumerate(rank_order):
        ranked=[(str(classes[index]),float(probabilities[spectrum_index,index])) for index in row[:3]]
        ranked.extend([('',float('nan'))]*(3-len(ranked)))
        rankings.append(ranked)
    y_test=labels_for(test,label_map,False)
    summary=dict(train_n=len(train.X),test_n=len(test.X),features=train.X.shape[1],classes=sorted(np.unique(y_train).tolist()),
                 k=k,preprocess=preprocess,axis=[float(train.axis[0]),float(train.axis[-1]),len(train.axis)])
    if y_test is not None:
        labels=sorted(np.unique(np.concatenate([y_train,y_test,prediction])).tolist())
        summary.update(accuracy=float(accuracy_score(y_test,prediction)),balanced_accuracy=float(balanced_accuracy_score(y_test,prediction)),
                       confusion=confusion_matrix(y_test,prediction,labels=labels).tolist(),labels=labels,
                       report=classification_report(y_test,prediction,labels=labels,zero_division=0))
    return prediction,y_test,summary,rankings

def make_confusion_figure(y_true,prediction,rankings,labels):
    """Return a True-column / Predicted-row percentage confusion-matrix figure."""
    cm_true_pred=confusion_matrix(y_true,prediction,labels=labels)
    true_counts=cm_true_pred.sum(axis=1,keepdims=True)
    percentage=np.divide(cm_true_pred,true_counts,out=np.zeros_like(cm_true_pred,dtype=float),where=true_counts!=0).T
    top_accuracy=[]
    for rank in (1,2,3):
        hits=[true_label in [label for label,_ in rankings[index][:rank]] for index,true_label in enumerate(y_true)]
        top_accuracy.append(100*float(np.mean(hits)))
    n=len(labels);size=max(8,0.55*n+3);fig,ax=plt.subplots(figsize=(size,size))
    ax.set_xlim(0,n);ax.set_ylim(n,0);ax.set_aspect('equal')
    fontsize=max(4,min(10,130/max(n,1)))
    for row in range(n):
        for col in range(n):
            value=percentage[row,col]
            if row==col: color=plt.cm.Greens(0.12+0.78*value/100)
            elif value>0: color=plt.cm.Reds(0.08+0.75*value/100)
            else: color='white'
            ax.add_patch(Rectangle((col,row),1,1,facecolor=color,edgecolor='#D9D9D9',linewidth=0.6))
            if true_counts[col,0]>0:
                text_color='white' if value>=55 else 'black'
                ax.text(col+.5,row+.5,f'{value:.1f}',ha='center',va='center',fontsize=fontsize,color=text_color)
    ax.set_xticks(np.arange(n)+.5);ax.set_yticks(np.arange(n)+.5)
    ax.set_xticklabels(labels,rotation=45,ha='left',fontsize=fontsize)
    ax.set_yticklabels(labels,fontsize=fontsize)
    ax.xaxis.tick_top();ax.xaxis.set_label_position('top')
    ax.set_xlabel('True class');ax.set_ylabel('Predicted class')
    ax.set_title(f'Top-1 accuracy: {top_accuracy[0]:.1f}%   |   Top-2 accuracy: {top_accuracy[1]:.1f}%   |   Top-3 accuracy: {top_accuracy[2]:.1f}%',pad=45)
    fig.tight_layout()
    return fig
