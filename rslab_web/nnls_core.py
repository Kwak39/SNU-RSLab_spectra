"""Headless numerical core extracted from RSLab_Sparse_NNLS(2).py. See SOURCE_NOTES.md."""
import csv

import json

import queue

import sys

import threading

import traceback

from pathlib import Path

from dataclasses import dataclass

import numpy as np
import scipy
from scipy.optimize import minimize, nnls

@dataclass
class Spectrum:
    axis: np.ndarray
    y: np.ndarray
    name: str
    source: str
    x: float = float('nan')
    coord_y: float = float('nan')
    mapping: bool = False

def checked(axis, signal, name, source, x=float('nan'), y=float('nan'), mapping=False):
    axis, signal = np.asarray(axis, dtype=float), np.asarray(signal, dtype=float)
    if len(axis) < 3 or len(axis) != len(signal) or not np.isfinite(axis).all() or not np.isfinite(signal).all():
        raise ValueError(f'{name}: 3점 이상의 유한한 파수/intensity가 필요합니다.')
    order = np.argsort(axis)
    if np.any(np.diff(axis[order]) <= 0):
        raise ValueError(f'{name}: 중복 파수가 있습니다.')
    return Spectrum(axis[order], signal[order], name, source, x, y, mapping)

def read_csv(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = [[v.strip() for v in row] for row in csv.reader(f) if any(v.strip() for v in row)]
    if not rows:
        raise ValueError(f'{path.name}: 빈 파일')
    first = rows[0]; source = str(path)
    if first[0].endswith(' | wavenumber'):
        if len(first) % 2:
            raise ValueError(f'{path.name}: 파수/intensity 열 쌍이 맞지 않습니다.')
        result = []
        for col in range(0, len(first), 2):
            pairs = []
            for row in rows[1:]:
                if len(row) != len(first):
                    raise ValueError(f'{path.name}: CSV 열 수가 일치하지 않습니다.')
                a,b = row[col:col+2]
                if not a and not b: continue
                if not a or not b: raise ValueError(f'{path.name}: 불완전한 파수/intensity 쌍')
                pairs.append((float(a),float(b)))
            if not pairs: raise ValueError(f'{path.name}: 빈 spectrum')
            a,b = zip(*pairs)
            result.append(checked(a,b,first[col].rsplit(' | ',1)[0],source))
        return result
    mapping = first[0] == 'x coordinate'
    if mapping:
        if len(rows)<4 or rows[1][0]!='y coordinate' or rows[2][0]!='Raman shift (cm^-1)':
            raise ValueError(f'{path.name}: mapping header를 확인하세요.')
        header, body = rows[2], rows[3:]
        coords = np.array([first[1:],rows[1][1:]],dtype=float).T
        if coords.shape != (len(header)-1,2) or not np.isfinite(coords).all():
            raise ValueError(f'{path.name}: 좌표와 spectrum 수가 맞지 않습니다.')
    elif first[0] == 'Raman shift (cm^-1)':
        header, body = first, rows[1:]
        coords = np.full((len(header)-1,2),np.nan)
    elif len(first)==2:
        try: [float(v) for v in first]; body=rows
        except ValueError: body=rows[1:]
        header=['wavenumber',path.stem]; coords=np.full((1,2),np.nan)
    else:
        raise ValueError(f'{path.name}: 지원하지 않는 CSV. parameters/labels가 아닌 spectrum CSV를 선택하세요.')
    if len(header)<2 or any(len(row)!=len(header) for row in body):
        raise ValueError(f'{path.name}: intensity 열 수를 확인하세요.')
    matrix=np.array(body,dtype=float)
    if matrix.ndim!=2: raise ValueError(f'{path.name}: 데이터가 없습니다.')
    return [checked(matrix[:,0],matrix[:,j],header[j],source,*coords[j-1],mapping)
            for j in range(1,len(header))]

def load_paths(paths):
    result=[]
    for p in dict.fromkeys(str(Path(p).resolve()) for p in paths): result.extend(read_csv(p))
    if not result: raise ValueError('데이터가 없습니다.')
    return result

def reference_templates(spectra, policy):
    result=[]
    for source in dict.fromkeys(s.source for s in spectra):
        group=[s for s in spectra if s.source==source]
        average=policy=='파일별 평균' or (policy=='자동 (mapping만 파일별 평균)' and group[0].mapping)
        if average:
            axis=group[0].axis
            if any(not np.array_equal(s.axis,axis) for s in group):
                raise ValueError('파일별 평균은 파일 내부 파수축이 같아야 합니다. 각 spectrum 모드를 선택하세요.')
            mean=np.zeros_like(axis)
            for s in group: mean+=s.y/len(group)
            result.append(checked(axis,mean,Path(source).stem+' [mean]',source))
        else: result.extend(group)
    # Unique visible reference IDs even with identical filenames/headers.
    names=[f'R{i+1:02d} | {s.name}' for i,s in enumerate(result)]
    if len(result)>200:
        raise ValueError('Reference가 200개를 초과합니다. 단일 물질 mapping은 파일별 평균 모드를 사용하세요.')
    return result,names

def solve(A,b,alpha):
    if alpha==0:
        c,_=nnls(A,b,maxiter=max(300,10*A.shape[1]));return c
    # Positive L1 penalty; intercept is fixed at zero, dictionary is fixed.
    def objective(c):
        residual=A@c-b
        return .5*np.dot(residual,residual)+alpha*c.sum(), A.T@residual+alpha
    fit=minimize(objective,np.zeros(A.shape[1]),jac=True,method='L-BFGS-B',
                 bounds=[(0,None)]*A.shape[1],options={'maxiter':3000,'ftol':1e-13,'gtol':1e-8})
    if not fit.success: raise ValueError(f'Sparse NNLS 수렴 실패: {fit.message}')
    c=fit.x
    gradient=A.T@(A@c-b)+alpha
    kkt=np.where(c>1e-8,np.abs(gradient),np.maximum(-gradient,0)).max()
    if kkt>1e-5: raise ValueError('Sparse NNLS 수렴 정밀도가 부족합니다. 유사한 reference들을 확인하세요.')
    return c

def fit_all(refs, tests, alpha=.01, policy='자동 (mapping만 파일별 평균)', progress=lambda s:None):
    if not np.isfinite(alpha) or alpha<0: raise ValueError('alpha는 0 이상의 유한한 수입니다.')
    templates,names=reference_templates(refs,policy)
    if not templates or not tests: raise ValueError('Reference와 Test를 모두 선택하세요.')
    all_spectra=templates+tests
    lo=max(s.axis[0] for s in all_spectra);hi=min(s.axis[-1] for s in all_spectra)
    grid=templates[0].axis; grid=grid[(grid>=lo)&(grid<=hi)]
    if len(grid)<3: raise ValueError('모든 입력 사이의 공통 파수 범위에 최소 3점이 필요합니다.')
    R=np.column_stack([np.interp(grid,s.axis,s.y) for s in templates])
    scales=np.linalg.norm(R,axis=0)
    if np.any(scales==0): raise ValueError('공통 범위에서 모두 0인 reference가 있습니다.')
    A=R/scales
    coeff=np.zeros((len(tests),len(templates))); norms=np.zeros(len(tests))
    metrics=[]
    for i,s in enumerate(tests):
        y=np.interp(grid,s.axis,s.y);norm=np.linalg.norm(y); norms[i]=norm
        if norm==0:
            metrics.append((float('nan'),0.,'zero_signal'));continue
        b=y/norm;c=solve(A,b,alpha);coeff[i]=c
        rel=float(np.linalg.norm(b-A@c))
        metrics.append((rel,float(np.sqrt(np.mean((y-norm*(A@c))**2))),
                        'no_active_reference' if c.sum()<=1e-12 else 'fit'))
        if i%20==0 or i==len(tests)-1: progress(f'Fitting {i+1}/{len(tests)}')
    similarity=A.T@A
    pairs=[(names[i],names[j],float(similarity[i,j])) for i in range(len(names))
           for j in range(i+1,len(names)) if similarity[i,j]>.98]
    return dict(grid=grid,A=A,coeff=coeff,norms=norms,metrics=metrics,names=names,
                refs=templates,tests=tests,alpha=alpha,policy=policy,similar_pairs=pairs,
                aligned=sum(not np.array_equal(s.axis,grid) for s in all_spectra))

def contributions(c):
    return 100*c/c.sum() if c.sum()>1e-12 else np.zeros_like(c)

def export_results(path,r):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f)
        w.writerow(['test_index','sample','source','x','y','status','relative_residual','RMSE']+
                   [f'contribution_pct | {n}' for n in r['names']]+[f'coefficient_normalized | {n}' for n in r['names']])
        for i,s in enumerate(r['tests']):
            rel,rmse,status=r['metrics'][i];c=r['coeff'][i]
            w.writerow([i+1,s.name,s.source,*[v if np.isfinite(v) else '' for v in (s.x,s.coord_y)],status,
                        rel if np.isfinite(rel) else '',rmse,*contributions(c),*c])
