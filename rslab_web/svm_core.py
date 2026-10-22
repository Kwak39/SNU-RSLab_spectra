"""Headless numerical core extracted from RSLab_SVM_Classifier(2).py. See SOURCE_NOTES.md."""
import csv

import json

import queue

import sys

import threading

import traceback

import warnings

from collections import Counter

from dataclasses import dataclass

from datetime import datetime

from pathlib import Path

import numpy as np
import sklearn
import joblib
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

VERSION = '1.0'

SEED = 42

AXIS_ATOL = 1e-6

@dataclass
class Dataset:
    axis: np.ndarray
    X: np.ndarray
    sample_ids: list
    sources: list
    names: list
    groups: list

def csv_rows(path):
    for encoding in ('utf-8-sig', 'cp949'):
        try:
            with Path(path).open(encoding=encoding, newline='') as f:
                return [r for r in csv.reader(f) if any(c.strip() for c in r)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f'{Path(path).name}: UTF-8 또는 CP949 CSV로 저장하세요.')

def match_axis(a, b):
    if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=AXIS_ATOL):
        raise ValueError('파수축/제외 구간이 일치하지 않습니다. 같은 범위·Output points·제외 설정으로 준비하세요. '
                         'AsLS의 서로 다른 축은 별도 정렬이 필요합니다. 이 프로그램은 보간하지 않습니다.')

def axis_and_values(axis, values, context):
    axis = np.asarray(axis, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(axis) < 2 or values.ndim != 2 or values.shape[1] != len(axis):
        raise ValueError(f'{context}: 파수/intensity 크기가 맞지 않습니다.')
    if not np.isfinite(axis).all() or not np.isfinite(values).all():
        raise ValueError(f'{context}: NaN/Inf 또는 비어 있는 값이 있습니다.')
    if np.all(np.diff(axis) < 0):
        axis, values = axis[::-1].copy(), values[:, ::-1].copy()
    if not np.all(np.diff(axis) > 0):
        raise ValueError(f'{context}: 파수축은 중복 없는 증가/감소 순서여야 합니다.')
    return axis, values

def load_csv(path):
    path = Path(path).resolve()
    rows = csv_rows(path)
    if len(rows) < 3:
        raise ValueError(f'{path.name}: 데이터가 부족합니다.')
    header = rows[0]
    mapping = header[0].strip() == 'x coordinate'
    paired = header[0].endswith(' | wavenumber')
    if paired:
        if len(header) % 2:
            raise ValueError('paired-axis CSV 열 수는 짝수여야 합니다.')
        axes, spectra, names = [], [], []
        for j in range(0, len(header), 2):
            if not header[j].endswith(' | wavenumber'):
                raise ValueError('paired-axis 파수 열 이름을 확인하세요.')
            pairs = []; padding = False
            for row in rows[1:]:
                if len(row) != len(header): raise ValueError('CSV 열 수가 일정하지 않습니다.')
                a, b = row[j].strip(), row[j+1].strip()
                if not a and not b: padding = True; continue
                if padding or not a or not b: raise ValueError('중간 빈칸 또는 불완전한 파수/intensity 쌍이 있습니다.')
                pairs.append((float(a), float(b)))
            if not pairs: raise ValueError('빈 spectrum입니다.')
            array = np.array(pairs)
            axis, value = axis_and_values(array[:, 0], array[:, 1][None, :], path.name)
            axes.append(axis); spectra.append(value[0]); names.append(header[j].rsplit(' | ', 1)[0])
        for axis in axes[1:]: match_axis(axes[0], axis)
        axis, X = axes[0], np.vstack(spectra)
    else:
        start = 3 if mapping else 1
        if mapping:
            if len(rows) < 5 or rows[1][0].strip() != 'y coordinate' or rows[2][0].strip() != 'Raman shift (cm^-1)':
                raise ValueError('mapping CSV의 x/y/파수 header를 확인하세요.')
            if len(rows[1]) != len(header) or len(rows[2]) != len(header):
                raise ValueError('mapping 좌표 열 수가 맞지 않습니다.')
            coords = np.asarray([header[1:], rows[1][1:]], dtype=float)
            if not np.isfinite(coords).all(): raise ValueError('mapping 좌표가 유효하지 않습니다.')
            names = rows[2][1:]
        elif header[0].strip() == 'Raman shift (cm^-1)':
            names = header[1:]
        else:
            raise ValueError(f'{path.name}: Unified에서 출력한 spectrum CSV를 선택하세요. Label/parameters CSV는 입력 대상이 아닙니다.')
        if len(header) < 2 or any(len(row) != len(header) for row in rows[start:]):
            raise ValueError(f'{path.name}: CSV 열 수가 맞지 않습니다.')
        raw = np.asarray(rows[start:], dtype=float)
        axis, X = axis_and_values(raw[:, 0], raw[:, 1:].T, path.name)
    ids = [f'{path.name} | {name}' for name in names]  # compatible with earlier k-NN templates
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path.name}: 중복 spectrum 이름/좌표가 있습니다.')
    groups = [path.name]*len(names) if mapping else list(names)
    return Dataset(axis, X, ids, [str(path)]*len(names), list(names), groups)

def combine(paths):
    data = [load_csv(path) for path in paths]
    if not data: raise ValueError('CSV 파일을 선택하세요.')
    for d in data[1:]: match_axis(data[0].axis, d.axis)
    def flatten(key): return [v for d in data for v in getattr(d, key)]
    ids = flatten('sample_ids')
    if len(set(ids)) != len(ids): raise ValueError('중복 sample_id가 있습니다. 같은 이름의 파일이나 같은 CSV를 중복 선택하지 마세요.')
    return Dataset(data[0].axis.copy(), np.vstack([d.X for d in data]), ids,
                   flatten('sources'), flatten('names'), flatten('groups'))

def write_template(path, data, labels, groups):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f); w.writerow(['sample_id', 'label', 'group'])
        w.writerows(zip(data.sample_ids, labels, groups))

def read_labels(path, data, labels, groups):
    rows = csv_rows(path)
    if not rows or rows[0][:2] != ['sample_id', 'label']:
        raise ValueError('Label 양식 첫 두 열은 sample_id,label 이어야 합니다.')
    gi = rows[0].index('group') if 'group' in rows[0] else None
    lookup = {}; duplicates = set()
    for row in rows[1:]:
        if len(row) < 2: raise ValueError('Label CSV 행이 불완전합니다.')
        if row[0] in lookup: duplicates.add(row[0])
        lookup[row[0]] = row
    if duplicates: raise ValueError('Label CSV에 중복 sample_id가 있습니다.')
    result, new_groups = list(labels), list(groups); count = 0
    for i, sid in enumerate(data.sample_ids):
        if sid not in lookup: continue
        row = lookup[sid]; result[i] = row[1].strip(); count += 1
        if gi is not None and len(row) > gi and row[gi].strip(): new_groups[i] = row[gi].strip()
    if not count: raise ValueError('일치하는 sample_id가 없습니다. 현재 데이터에서 label 양식을 다시 생성하세요.')
    return result, new_groups, count

def calibration_splits(y, groups):
    classes, counts = np.unique(y, return_counts=True)
    if counts.min() < 2: raise ValueError('확률 보정을 위해 각 class에 최소 2개 spectrum이 필요합니다. 독립 반복 측정을 권장합니다.')
    ng = min(len(set(groups[y == c])) for c in classes)
    for k in range(min(3, ng), 1, -1):
        splits = list(StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups))
        if all(set(y[tr]) == set(classes) and len(te) for tr, te in splits):
            return splits, f'Group-disjoint {k}-fold probability calibration'
    k = int(min(3, counts.min()))
    return list(StratifiedKFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)), (
        f'Spectrum-level {k}-fold calibration: independent groups 부족. 같은 mapping의 pixel이 fold에 섞일 수 있어 확률을 보수적으로 해석하세요.')

def train_model(data, labels, groups, config):
    if any(not x.strip() for x in labels): raise ValueError('TRAIN의 모든 spectrum에 label을 지정하세요.')
    y = np.asarray(labels, dtype=str); group = np.asarray(groups, dtype=str)
    if len(np.unique(y)) < 2: raise ValueError('최소 2종의 class가 필요합니다.')
    C = float(config['C']); gamma = config['gamma']
    if not np.isfinite(C) or C <= 0: raise ValueError('C는 유한한 양수입니다.')
    if gamma not in ('scale', 'auto'):
        gamma = float(gamma)
        if not np.isfinite(gamma) or gamma <= 0: raise ValueError('gamma는 scale/auto 또는 양수입니다.')
    cap = int(config['max_per_class'])
    if cap < 0 or cap == 1: raise ValueError('Class당 학습 수는 0(전체) 또는 2 이상입니다.')
    rng = np.random.default_rng(SEED); selected = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        if cap and len(indices) > cap: indices = rng.choice(indices, cap, replace=False)
        selected.extend(indices)
    selected = np.sort(selected)
    X = data.X[selected]; ys = y[selected]; gs = group[selected]
    if np.any(np.linalg.norm(X, axis=1) == 0): raise ValueError('TRAIN에 완전히 0인 spectrum이 있습니다.')
    steps = []
    if config['normalization'] in ('L2', 'L2 + StandardScaler'): steps.append(('l2', Normalizer()))
    if config['normalization'] in ('StandardScaler', 'L2 + StandardScaler'): steps.append(('scale', StandardScaler()))
    steps.append(('svm', SVC(C=C, kernel=config['kernel'], gamma=gamma,
        class_weight='balanced' if config['balanced'] else None, cache_size=256, max_iter=100000)))
    splits, mode = calibration_splits(ys, gs)
    calibrated = CalibratedClassifierCV(Pipeline(steps), method='sigmoid', cv=splits, ensemble=False, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        calibrated.fit(X, ys)
    metadata = dict(app_version=VERSION, sklearn_version=sklearn.__version__, created=datetime.now().isoformat(),
        config=dict(config), calibration=mode, train_total=len(y), train_used=len(ys),
        counts_before=dict(Counter(labels)), counts_used=dict(Counter(ys.tolist())),
        classes=calibrated.classes_.tolist(), preprocessing='TRAIN-fold-only fit; no spectral interpolation',
        selected_training_ids=[data.sample_ids[i] for i in selected])
    return dict(format='RSLabSVM1', model=calibrated, axis=data.axis.copy(), metadata=metadata,
                train_ids=list(data.sample_ids), train_sources=sorted(set(data.sources)), train_groups=sorted(set(groups)))

def predict(bundle, data, labels, groups):
    match_axis(bundle['axis'], data.axis)
    if set(bundle['train_ids']) & set(data.sample_ids) or set(bundle['train_sources']) & set(data.sources):
        raise ValueError('TRAIN과 TEST에 같은 입력이 있습니다. 독립적으로 측정한 TEST CSV를 선택하세요.')
    model = bundle['model']; classes = model.classes_
    probabilities = np.vstack([model.predict_proba(data.X[i:i+512]) for i in range(0, len(data.X), 512)])
    ranks = np.argsort(-probabilities, axis=1, kind='stable')
    predicted = classes[ranks[:, 0]]
    notes = ['Probability는 학습 class 사이의 추정값이며, 성분 기여율/농도/미지물질 검출 점수가 아닙니다.']
    overlap = set(groups) & set(bundle['train_groups'])
    if overlap: notes.append(f'TRAIN/TEST group 이름 {len(overlap)}개가 겹칩니다. 같은 실제 시료이면 독립 평가가 아닙니다.')
    zero = np.linalg.norm(data.X, axis=1) == 0
    if zero.any(): notes.append(f'0 spectrum {int(zero.sum())}개: 예측 결과를 신뢰하지 마세요.')
    known = np.array([bool(label.strip()) for label in labels]); y = np.asarray(labels)
    report = None; matrix = None; matrix_labels = None
    scores = dict(test_count=len(labels), labeled_count=int(known.sum()), notes=notes)
    if known.any():
        truth = y[known]; guesses = predicted[known]
        report = classification_report(truth, guesses, output_dict=True, zero_division=0)
        matrix_labels = sorted(set(classes) | set(truth))
        matrix = confusion_matrix(truth, guesses, labels=matrix_labels)
        scores.update(accuracy=float(accuracy_score(truth, guesses)),
                      macro_f1=float(f1_score(truth, guesses, average='macro', zero_division=0)))
        for k in range(1, min(3, len(classes))+1):
            scores[f'top_{k}_accuracy'] = float(np.mean([t in classes[ranks[i, :k]] for i,t in zip(np.flatnonzero(known), truth)]))
        unknown = sorted(set(truth)-set(classes))
        if unknown: scores['untrained_test_labels'] = unknown
    return dict(probabilities=probabilities, ranks=ranks, predicted=predicted, classes=classes,
                labels=list(labels), groups=list(groups), zero=zero, scores=scores,
                report=report, matrix=matrix, matrix_labels=matrix_labels, data=data,
                model_metadata=bundle['metadata'])

def save_results(root, result):
    folder = Path(root)/datetime.now().strftime('SVM_%Y%m%d_%H%M%S_%f'); folder.mkdir(parents=True)
    data = result['data']; classes = result['classes']
    with (folder/'predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sample_id', 'source', 'spectrum', 'group', 'true_label', 'predicted_label', 'zero_signal']+
            [s for k in range(1,4) for s in (f'top{k}_class', f'top{k}_probability')]+[f'probability::{c}' for c in classes])
        for i, sid in enumerate(data.sample_ids):
            row = [sid, data.sources[i], data.names[i], result['groups'][i], result['labels'][i],
                   result['predicted'][i], bool(result['zero'][i])]
            for k in range(3):
                if k < len(classes):
                    j = result['ranks'][i,k]; row += [classes[j], float(result['probabilities'][i,j])]
                else: row += ['', '']
            w.writerow(row+result['probabilities'][i].tolist())
    with (folder/'settings_and_scores.json').open('w', encoding='utf-8') as f:
        json.dump(dict(model=result['model_metadata'], scores=result['scores'], axis=data.axis.tolist()), f, ensure_ascii=False, indent=2)
    if result['report'] is not None:
        with (folder/'classification_report.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow(['class_or_average', 'precision', 'recall', 'f1-score', 'support'])
            for key, value in result['report'].items():
                if isinstance(value, dict): w.writerow([key]+[value.get(k,'') for k in ('precision','recall','f1-score','support')])
        with (folder/'confusion_matrix_counts.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow(['true / predicted']+result['matrix_labels'])
            for name, row in zip(result['matrix_labels'], result['matrix']): w.writerow([name]+row.tolist())
    return folder
