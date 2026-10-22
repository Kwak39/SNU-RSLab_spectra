"""Headless v2.4.0 core: native BLaPE followed by fixed-grid Final extraction.
Numerical/export code shared with RSLab_Spectra_Unified_v2_4_0.py.
"""
import csv
import gc
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from itertools import chain
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import solveh_banded
from scipy.special import expit
from scipy.signal import savgol_filter

VERSION = '2.5.3 A1/column-layout detection + Savitzky-Golay + native BLaPE + Final'

@dataclass(frozen=True)
class InputSettings:
    encoding: str = 'auto'
    delimiter: str = 'auto'
    skiprows: int = 0

@dataclass(frozen=True)
class AsLSSettings:
    lam: float = 1e6
    p: float = .001
    niter: int = 10
    anomaly: bool = False
    anomaly_window: int = 5
    anomaly_threshold: float = 2.0
    method: str = 'AsLS'
    tol: float = 1e-3

@dataclass(frozen=True)
class RangeSettings:
    lower: float | None = None
    upper: float | None = None

@dataclass(frozen=True)
class BLaPESettings:
    sigma: float = 30.0
    eps: float = .25
    edge_trim: int = 10
    target_points: int | None = None  # None: retain trusted native coordinates
    exclude_ranges: tuple = ()
    exclusion_margin: float = 0.0  # cm^-1; added to both sides of each interval
    sg_enabled: bool = False
    sg_window: int = 11  # native data points, no resampling
    sg_order: int = 3

@dataclass(frozen=True)
class FinalRangeSettings:
    lower: float = 400.0
    upper: float = 1500.0
    step: float = 1.0

def bands_for(result, fi):
    per_file = result.get('bands_by_file')
    if per_file is not None:
        return per_file[fi]
    bands = result.get('bands')
    if bands:
        return bands
    axis = result['targets'][fi]
    return [dict(lower=float(axis[0]), upper=float(axis[-1]),
                 start_index=0, stop_index=len(axis))]

def intersect_supports(supports, current):
    return sorted((max(a,c),min(b,d)) for a,b in supports for c,d in current
                  if max(a,c) <= min(b,d))

def interpolate_bands(target, axis, row, bands):
    """Never extrapolate or interpolate across removed/masked intervals."""
    output = np.full(len(target), np.nan)
    for band in bands:
        a,b = band['start_index'],band['stop_index']
        x = axis[a:b]
        selected = (target >= x[0]) & (target <= x[-1])
        if np.any(selected):
            output[selected] = np.interp(target[selected], x, row[a:b])
    if not np.isfinite(output).all():
        raise ValueError('요청 축이 유효 범위를 벗어나거나 제외/보호 구간을 가로지릅니다.')
    return output

def drawing_axis(axis, row, bands):
    ends = [band['stop_index'] for band in bands[:-1]]
    return np.insert(axis,ends,np.nan),np.insert(row,ends,np.nan)

def parse_exclusions(text):
    """Examples: 500,540; 950,1000. Inclusive physical wavenumber limits."""
    intervals = []
    for part in re.split(r'[;\n]+', text.strip()):
        if not part.strip(): continue
        values = part.split(',')
        if len(values) != 2:
            raise ValueError('제외 구간은 하한,상한; 하한,상한 형식입니다. 예: 500,540; 950,1000')
        lo, hi = map(float, values)
        if not np.isfinite([lo,hi]).all() or lo >= hi:
            raise ValueError('제외 구간은 유한한 하한 < 상한이어야 합니다.')
        intervals.append((lo,hi))
    merged = []
    for lo,hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0],max(hi,merged[-1][1]))
        else: merged.append((lo,hi))
    return tuple(merged)

def modified_segments(axis, cfg):
    """Return raw segment and trusted core indices, never spanning a mask."""
    keep = np.ones(len(axis),dtype=bool)
    for lo,hi in cfg.exclude_ranges:
        keep &= ~((axis >= lo-cfg.exclusion_margin) & (axis <= hi+cfg.exclusion_margin))
    indices = np.flatnonzero(keep)
    if not len(indices): return []
    groups = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    # Support of the optional SG, Gaussian and nine-point Laplacian in series.
    sg_radius = cfg.sg_window // 2 if cfg.sg_enabled else 0
    guard = max(cfg.edge_trim, sg_radius + int(4*cfg.sigma+0.5)+4)
    result = []
    for group in groups:
        start,stop = int(group[0]),int(group[-1])+1
        # Split also across an excluded interval lying entirely between samples.
        cuts = [start]
        for lo,hi in cfg.exclude_ranges:
            boundary = int(np.searchsorted(axis,lo-cfg.exclusion_margin))
            if start < boundary < stop: cuts.append(boundary)
        cuts = sorted(set(cuts+[stop]))
        for a,b in zip(cuts[:-1],cuts[1:]):
            left = guard if a>0 else max(cfg.edge_trim,4)
            right = guard if b<len(axis) else max(cfg.edge_trim,4)
            if b-a >= 9 and b-right-(a+left) >= 2:
                result.append((a,b,a+left,b-right))
    return result

def modified_targets(axes, cfg):
    """Common continuous supports; exact total target_points across all bands."""
    supports = None
    for axis in axes:
        current = [(float(axis[c]),float(axis[d-1])) for a,b,c,d in modified_segments(axis,cfg)]
        if supports is None: supports = current
        else:
            supports = [(max(a,c),min(b,d)) for a,b in supports for c,d in current
                        if max(a,c)<min(b,d)]
        if not supports:
            raise ValueError('제외/경계 처리 후 공통 유효 구간이 없습니다. sigma, 제외 폭 또는 입력 범위를 확인하세요.')
    supports.sort()
    if cfg.target_points < 2*len(supports):
        raise ValueError(f'유효 구간 {len(supports)}개: Output points는 최소 {2*len(supports)}이어야 합니다.')
    widths = np.array([b-a for a,b in supports])
    extra = (cfg.target_points-2*len(supports))*widths/widths.sum()
    counts = 2+np.floor(extra).astype(int)
    for i in np.argsort(-(extra-np.floor(extra)))[:cfg.target_points-int(counts.sum())]:
        counts[i] += 1
    target = np.concatenate([np.linspace(a,b,int(n)) for (a,b),n in zip(supports,counts)])
    bands = []; offset = 0
    for (lo,hi),n in zip(supports,counts):
        bands.append(dict(lower=lo,upper=hi,start_index=offset,stop_index=offset+int(n)))
        offset += int(n)
    return [target]*len(axes), bands

def modified_blape(corrected, axis, target, cfg):
    segments = modified_segments(axis,cfg)
    if not segments: raise ValueError('modified_BLaPE 계산에 충분한 유효 구간이 없습니다.')
    corrected = smooth_blape_input(corrected, axis, cfg)
    # Scalar reference computed only from trusted, unmasked samples.
    total = sum(d-c for a,b,c,d in segments)
    mean_abs = sum(float(np.sum(np.abs(corrected[c:d]))) for a,b,c,d in segments)/total
    output = np.full(len(target),np.nan)
    smooth_full = np.full(len(axis),np.nan)
    for a,b,c,d in segments:
        smooth = gaussian_filter1d(corrected[a:b],cfg.sigma)
        numerator = -np.convolve(smooth,LAPLACIAN,'same')
        # Suppress floating-point stencil residue on a constant segment.
        noise_floor = 64*np.finfo(float).eps*max(float(np.max(np.abs(smooth))),np.finfo(float).tiny)
        numerator[np.abs(numerator)<=noise_floor] = 0
        denominator = smooth+cfg.eps*mean_abs
        sl = slice(c-a,d-a)
        den = denominator[sl]
        scale = max(float(np.max(np.abs(den))),np.finfo(float).tiny)
        feature = np.divide(numerator[sl],den,out=np.zeros(d-c),
                            where=np.abs(den)>np.finfo(float).eps*scale)
        feature = np.maximum(feature,0)
        smooth_full[c:d] = smooth[sl]
        selected = (target>=axis[c]) & (target<=axis[d-1])
        output[selected] = (feature if np.array_equal(target[selected],axis[c:d])
                            else np.interp(target[selected],axis[c:d],feature))
    if not np.isfinite(output).all():
        raise ValueError('출력축에 제외 구간이 포함되어 있습니다. 구간별 공통 출력축을 사용하세요.')
    maximum = float(output.max())
    if maximum>0: output /= maximum
    return output,smooth_full

class Cancelled(Exception):
    pass

def checkpoint(cancel):
    if cancel is not None and cancel.is_set():
        raise Cancelled('작업을 취소했습니다.')

def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name}: 유한한 양수를 입력하세요.')

def integer(value, minimum, name):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name}: {minimum} 이상 정수를 입력하세요.')

# %% 2. 입력 파일 읽기 / single, 좌표 mapping, 좌표 없는 matrix 자동 판별

def input_rows(path, settings, cancel=None):
    integer(settings.skiprows, 0, 'skiprows')
    delimiters = {'auto': None, 'tab': '\t', 'comma': ',', 'semicolon': ';', 'space': None}
    if settings.delimiter not in delimiters:
        raise ValueError('구분자 설정을 확인하세요.')
    encoding = settings.encoding
    if encoding == 'auto':
        # Validate in bounded chunks; never keep a second copy of a mapping file.
        for candidate in ('utf-8-sig', 'cp949'):
            try:
                with Path(path).open(encoding=candidate) as probe:
                    while probe.read(65536):
                        checkpoint(cancel)
                encoding = candidate
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f'{Path(path).name}: UTF-8 또는 CP949 텍스트가 아닙니다.')
    with Path(path).open(encoding=encoding, newline='') as stream:
        for number, line in enumerate(stream, 1):
            checkpoint(cancel)
            if number <= settings.skiprows or not line.strip() or line.lstrip().startswith(('#', '%', '//')):
                continue
            delimiter = delimiters[settings.delimiter]
            if settings.delimiter == 'auto':
                delimiter = next((d for d in ('\t', ',', ';') if d in line), None)
            fields = next(csv.reader([line], delimiter=delimiter)) if delimiter else line.split()
            fields = [v.strip() for v in fields]
            while fields and not fields[-1]:
                fields.pop()
            yield number, fields

def numeric(fields, path, number):
    try:
        values = np.asarray(fields, dtype=float)
    except ValueError as exc:
        raise ValueError(f'{Path(path).name}, {number}행: 숫자가 아닌 값이 있습니다.') from exc
    if not np.isfinite(values).all():
        raise ValueError(f'{Path(path).name}, {number}행: NaN/Inf는 지원하지 않습니다.')
    return values

def read_auto(path, settings=InputSettings(), cancel=None):
    """Read point-major XY/matrices or row-major mapping with A=y, B=x.

    A numeric A1 in a rectangular table fixes the axis to column A: two columns
    are single, more columns are a matrix. Intensities never vote on orientation.
    Empty/named coordinate corners identify mapping. A shorter axis-only header
    also supports legacy whitespace mapping exports with omitted placeholders.
    """
    path = Path(path).resolve()
    iterator = iter(input_rows(path, settings, cancel))
    first = next(iterator, None); second = next(iterator, None)
    if first is None or second is None:
        raise ValueError(f'{path.name}: 데이터가 부족합니다.')

    def numbers(fields):
        try:
            return np.asarray(fields, dtype=float)
        except ValueError:
            return None

    width = len(second[1])
    omitted_placeholders = (width >= 5 and len(first[1]) == width - 2
                            and numbers(first[1]) is not None)
    y_labels = {'', 'y', 'y coordinate', 'y position', 'position y', 'pos y'}
    x_labels = {'', 'x', 'x coordinate', 'x position', 'position x', 'pos x'}
    named_placeholders = (width >= 5 and len(first[1]) == width
                          and first[1][0].casefold().replace('_', ' ') in y_labels
                          and first[1][1].casefold().replace('_', ' ') in x_labels
                          and numbers(first[1][2:]) is not None)
    if omitted_placeholders or named_placeholders:
        header = first[1] if omitted_placeholders else first[1][2:]
        axis = numeric(header, path, first[0]); locations = []; spectra = []
        for number, fields in chain([second], iterator):
            if len(fields) != len(axis) + 2:
                raise ValueError(f'{path.name}, {number}행: mapping 열 수가 일치하지 않습니다.')
            values = numeric(fields, path, number)
            locations.append((values[1], values[0])); spectra.append(values[2:])
        coords = np.asarray(locations); signals = np.asarray(spectra); kind = 'mapping'
    else:
        rows = []; width = None
        for number, fields in chain([first, second], iterator):
            values = numbers(fields)
            if values is None and not rows and numbers(fields[:1]) is None:
                continue  # Optional leading text labels; A1 numeric is never skipped.
            values = numeric(values if values is not None else fields, path, number)
            if width is None:
                width = len(values)
            if len(values) != width or width < 2:
                raise ValueError(f'{path.name}, {number}행: 파수축/intensity 열 수가 일치하지 않습니다.')
            rows.append(values)
        values = np.asarray(rows)
        del rows
        if values.ndim != 2:
            raise ValueError(f'{path.name}: 숫자 데이터가 없습니다.')
        # A1=0 is a real point, never an empty coordinate placeholder. Keep
        # row 1 and every intensity column, even if intensities are monotonic.
        axis = values[:, 0]; signals = values[:, 1:].T
        coords = None; kind = 'single' if values.shape[1] == 2 else 'matrix'
    order = np.argsort(axis, kind='stable')
    axis = axis[order]
    if len(axis) < 3 or np.any(np.diff(axis) <= 0):
        raise ValueError(f'{path.name}: 파수축은 중복 없이 최소 3점이어야 합니다.')
    if not np.array_equal(order, np.arange(len(axis))):
        for row in signals:
            row[:] = row[order]
    return dict(path=path, kind=kind, axis=axis, raw=signals, coords=coords,
                input_format={'single': 'single_xy', 'mapping': 'mapping_yx_rows',
                              'matrix': 'spectra_columns'}[kind],
                export_order=np.argsort(order))


def spectrum_name(item, index):
    """Stable names without inventing spatial coordinates for a numeric matrix."""
    if item['coords'] is not None:
        x, y = item['coords'][index]
        return f'x={x:g}, y={y:g}'
    return item['path'].name if item['kind'] == 'single' else f'Spectrum {index+1}'

# %% 3. AsLS baseline correction / 선택적 사전 이상치 보정
import warnings
from functools import lru_cache
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from scipy.ndimage import gaussian_filter1d

LAPLACIAN = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]

@lru_cache(maxsize=8)
def _asls_penalty(m, lam):
    d = sparse.diags([np.ones(m-2), -2*np.ones(m-2), np.ones(m-2)],
                     [0,1,2], shape=(m-2,m), format='csc')
    return lam*(d.T@d)

def asls(y, lam=1e6, p=0.001, niter=10):
    """AsLS.m translation; sparse solve replaces Cholesky. Return baseline, residual."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y)<3 or not np.isfinite(y).all():
        raise ValueError('AsLS requires finite 1D data with >=3 points.')
    if not np.isfinite(lam) or lam <= 0 or not 0<p<1:
        raise ValueError('Require lambda>0 and 0<p<1.')
    if not isinstance(niter, (int, np.integer)) or niter < 1:
        raise ValueError('niter must be a positive integer.')
    if np.all(y==y[0]):
        return y.copy(), np.zeros_like(y)
    m = len(y)
    penalty = _asls_penalty(m, float(lam))
    w = np.ones(m)
    for _ in range(niter):
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            baseline = spsolve(sparse.diags(w, format='csc')+penalty, w*y)
        if not np.isfinite(baseline).all():
            raise ValueError('Non-finite AsLS baseline.')
        w = p*(y>baseline)+(1-p)*(y<baseline)
        if np.count_nonzero(w)<2:
            break  # safeguard for an exact fit
    return baseline, y-baseline


@lru_cache(maxsize=8)
def _baseline_bands(size, lam):
    """Lower bands of lambda D.T D; the same penalty as the original AsLS."""
    penalty = _asls_penalty(size, lam)
    bands = np.zeros((3, size))
    for offset in range(3):
        diagonal = penalty.diagonal(-offset)
        bands[offset, :len(diagonal)] = diagonal
    bands.setflags(write=False)
    return bands


def arpls(y, lam=1e5, niter=50, tol=1e-3):
    """Baek et al. (2015), DOI 10.1039/C4AN01061B, second-difference arPLS.

    Independently implemented from the weighting equation documented by
    pybaselines (github.com/derb12/pybaselines, docs/algorithms/whittaker.rst).
    expit avoids exponential overflow; a banded solver avoids repeated sparse
    factorization. Returns baseline and residual on the untouched native axis.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError('arPLS requires finite 1D data with >=3 points.')
    positive(lam, 'arPLS lambda')
    positive(tol, 'arPLS tolerance')
    if not isinstance(niter, (int, np.integer)) or niter < 1:
        raise ValueError('arPLS iterations must be a positive integer.')
    if np.all(y == y[0]):
        return y.copy(), np.zeros_like(y)
    penalty = _baseline_bands(len(y), float(lam))
    weights = np.ones(len(y))
    for _ in range(niter):
        lhs = penalty.copy()
        lhs[0] += weights
        baseline = solveh_banded(lhs, weights * y, lower=True,
                                overwrite_ab=True, check_finite=False)
        if not np.isfinite(baseline).all():
            raise ValueError('Non-finite arPLS baseline.')
        residual = y - baseline
        negative = residual[residual < 0]
        if len(negative) < 2:
            break
        sigma = float(negative.std(ddof=1))
        floor = np.finfo(float).eps * max(1., float(np.max(np.abs(y))))
        if sigma <= floor:
            break
        updated = expit(-2. * (residual - (2. * sigma - negative.mean())) / sigma)
        updated = np.maximum(updated, 1e-12)
        change = np.linalg.norm(updated - weights) / np.linalg.norm(weights)
        if change < tol:
            break
        weights = updated
    return baseline, y - baseline

def anomaly_impute(y, window_size=5, threshold=2.0):
    if not isinstance(window_size, int) or window_size<1 or threshold<=0:
        raise ValueError('Invalid anomaly settings.')
    result = np.asarray(y, dtype=float).copy()
    std = np.std(result)
    for j in range(window_size, len(result)-window_size):
        median = np.median(result[j-window_size:j+window_size+1])
        if result[j]>median+threshold*std:
            result[j]=median
    return result



def validate_asls(cfg):
    positive(cfg.lam, 'lambda')
    if cfg.method not in ('AsLS', 'arPLS'):
        raise ValueError('Baseline 방법은 AsLS 또는 arPLS입니다.')
    if cfg.method == 'AsLS' and (not np.isfinite(cfg.p) or not 0 < cfg.p < 1):
        raise ValueError('AsLS p는 0과 1 사이여야 합니다.')
    integer(cfg.niter, 1, 'AsLS 반복 횟수')
    integer(cfg.anomaly_window, 1, 'anomaly window')
    positive(cfg.anomaly_threshold, 'anomaly threshold')
    positive(cfg.tol, 'arPLS tolerance')

def correct_one(raw, cfg):
    clean = anomaly_impute(raw, cfg.anomaly_window, cfg.anomaly_threshold) if cfg.anomaly else raw
    if cfg.method == 'arPLS':
        return arpls(clean, cfg.lam, cfg.niter, cfg.tol)[1]
    return asls(clean, cfg.lam, cfg.p, cfg.niter)[1]

# %% 4. BLaPE — 이미 AsLS 처리한 신호만 사용 (baseline 중복 제거 없음)

def validate_blape(cfg):
    positive(cfg.sigma, 'sigma'); positive(cfg.eps, 'eps')
    integer(cfg.edge_trim, 0, 'edge trim')
    if cfg.target_points is not None:
        integer(cfg.target_points, 2, 'BLaPE output points')
    if not np.isfinite(cfg.exclusion_margin) or cfg.exclusion_margin<0:
        raise ValueError('Exclusion margin은 0 이상 cm^-1 값입니다.')
    for lo,hi in cfg.exclude_ranges:
        if not np.isfinite([lo,hi]).all() or lo>=hi:
            raise ValueError('제외 범위를 확인하세요.')
    if cfg.sg_enabled:
        integer(cfg.sg_window, 3, 'Savitzky-Golay window')
        integer(cfg.sg_order, 0, 'Savitzky-Golay polynomial order')
        if cfg.sg_window % 2 == 0:
            raise ValueError('Savitzky-Golay window는 3 이상의 홀수 point 수여야 합니다.')
        if cfg.sg_order >= cfg.sg_window:
            raise ValueError('Savitzky-Golay 다항식 차수는 window보다 작아야 합니다.')


def smooth_blape_input(corrected, axis, cfg):
    """Baseline -> optional SG (deriv=0, interp) -> BLaPE, without axis changes.

    Excluded intervals are split before filtering: a strong removed Si peak
    cannot leak into retained values through the additional smoothing filter.
    The original baseline-corrected array is never changed in place.
    """
    if not cfg.sg_enabled:
        return corrected
    validate_blape(cfg)
    segments = ([(a, b) for a, b, _, _ in modified_segments(axis, cfg)]
                if cfg.exclude_ranges else [(0, len(axis))])
    if not segments:
        raise ValueError('제외/경계 처리 후 smoothing에 사용할 구간이 없습니다.')
    output = np.full_like(corrected, np.nan, dtype=float)
    for start, stop in segments:
        if cfg.sg_window > stop - start:
            raise ValueError(f'Savitzky-Golay window={cfg.sg_window}가 입력 구간의 '
                             f'{stop-start} points보다 큽니다. window를 줄이세요.')
        output[start:stop] = savgol_filter(corrected[start:stop], cfg.sg_window,
                                         cfg.sg_order, deriv=0, mode='interp')
    return output

def blape_from_corrected(corrected, axis, target, cfg):
    validate_blape(cfg)
    if cfg.exclude_ranges:
        return modified_blape(corrected,axis,target,cfg)
    if len(axis) < max(9, 2*cfg.edge_trim+2):
        raise ValueError('BLaPE 또는 edge trim에 필요한 입력 point가 부족합니다.')
    corrected = smooth_blape_input(corrected, axis, cfg)
    smooth = gaussian_filter1d(corrected, cfg.sigma)
    numerator = -np.convolve(smooth, LAPLACIAN, 'same')
    denominator = smooth + cfg.eps*np.mean(np.abs(corrected))
    scale = max(float(np.max(np.abs(denominator))), np.finfo(float).tiny)
    peaks = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                      where=np.abs(denominator) > np.finfo(float).eps*scale)
    peaks = np.maximum(peaks, 0)
    if cfg.edge_trim:
        peaks = peaks[cfg.edge_trim:-cfg.edge_trim]
        valid_axis = axis[cfg.edge_trim:-cfg.edge_trim]
    else:
        valid_axis = axis
    if target[0] < valid_axis[0] or target[-1] > valid_axis[-1]:
        raise ValueError('출력 파수축이 유효한 입력 범위를 벗어났습니다.')
    if np.max(peaks) > 0:
        peaks /= np.max(peaks)
    return (peaks.copy() if np.array_equal(target,valid_axis)
            else np.interp(target, valid_axis, peaks)), smooth

# %% 5. AsLS 이후 원래 파수 범위 추출 / 단계별 결과 관리

def validate_range(cfg):
    for value in (cfg.lower, cfg.upper):
        if value is not None and not np.isfinite(value):
            raise ValueError('범위는 유한한 파수 또는 빈칸이어야 합니다.')
    if cfg.lower is not None and cfg.upper is not None and cfg.lower >= cfg.upper:
        raise ValueError('하한은 상한보다 작아야 합니다.')

def snap_range(axis, cfg):
    # Strictly greater, even when a requested value is an exact input point.
    first = 0 if cfg.lower is None else int(np.searchsorted(axis, cfg.lower, side='right'))
    last = len(axis)-1 if cfg.upper is None else int(np.searchsorted(axis, cfg.upper, side='right'))
    if first >= len(axis) or last >= len(axis):
        raise ValueError('입력 경계보다 큰 파수가 없습니다. 범위를 줄이거나 끝 경계를 빈칸으로 두세요.')
    if last-first < 1:
        raise ValueError('선택한 범위에 서로 다른 파수 2점 이상이 필요합니다.')
    return first, last  # Both selected endpoint samples are included.


def make_targets(items, kind, points, edge=0):
    bounds = []
    for item in items:
        axis = item['axis']
        if len(axis) < max(3, 2*edge+2):
            raise ValueError(f'{item["path"].name}: edge trim에 비해 데이터가 짧습니다.')
        bounds.append((axis[edge], axis[-edge-1]))
    if kind == 'single':
        lo, hi = max(v[0] for v in bounds), min(v[1] for v in bounds)
        if lo >= hi:
            raise ValueError('입력 파일들 사이에 공통 파수 범위가 없습니다.')
        common = np.linspace(lo, hi, points)
        return [common]*len(items)
    return [np.linspace(lo, hi, points) for lo, hi in bounds]

class SpectraSession:
    def __init__(self):
        self.clear()

    def clear(self):
        self.items = []; self.kind = None; self.input_settings = None
        self.excluded_pixels = set()  # (file index, spectrum index), export only
        self.asls_result = None; self.range_result = None; self.blape_result = None; self.final_result = None

    def load(self, paths, settings=InputSettings(), emit=lambda *a: None, cancel=None):
        self.clear()
        paths = [Path(p).resolve() for p in paths]
        if not paths:
            raise ValueError('파일을 선택하세요.')
        if len({p.name.casefold() for p in paths}) != len(paths):
            raise ValueError('동일한 파일명이 있습니다. 이름을 구분하세요.')
        items = []
        for i, path in enumerate(paths, 1):
            checkpoint(cancel)
            item = read_auto(path, settings, cancel)
            items.append(item)
            emit('log', f'{i}. {path.name}: {item["kind"]}, {len(item["raw"])} spectra × {len(item["axis"])} points')
            emit('progress', (100*i/len(paths), f'파일 읽기 {i}/{len(paths)}'))
        checkpoint(cancel)
        kinds = {item['kind'] for item in items}
        self.items = items; self.kind = next(iter(kinds)) if len(kinds) == 1 else 'mixed'
        self.input_settings = replace(settings)
        return self.kind

    def run_asls(self, cfg=AsLSSettings(), emit=lambda *a: None, cancel=None):
        validate_asls(cfg)
        if not self.items:
            raise ValueError('먼저 파일을 불러오세요.')
        self.asls_result = None; self.range_result = None; self.blape_result = None; self.final_result = None
        targets = [item['axis'][item['export_order']] for item in self.items]
        arrays = []; total = sum(len(v['raw']) for v in self.items); done = 0; last = 0
        for item in self.items:
            corrected = np.empty_like(item['raw'])
            for i, raw in enumerate(item['raw']):
                checkpoint(cancel)
                try:
                    corrected[i] = correct_one(raw, cfg)
                except Exception as exc:
                    raise ValueError(f'{item["path"].name}, spectrum {i+1}: {exc}') from exc
                done += 1; now = time.monotonic()
                if now-last >= .15 or done == total:
                    emit('progress', (100*done/total, f'{cfg.method} {done}/{total}')); last = now; time.sleep(0)
            arrays.append(corrected)
        checkpoint(cancel)
        self.asls_result = dict(params=replace(cfg), arrays=arrays, targets=targets,
                                computed_at=datetime.now().isoformat())
        return self.asls_result

    def run_range(self, cfg=RangeSettings(), emit=lambda *a: None, cancel=None):
        validate_range(cfg)
        if self.asls_result is None:
            raise ValueError('AsLS를 먼저 실행하세요.')
        self.range_result = None; self.blape_result = None; self.final_result = None
        bounds = []
        for item in self.items:
            checkpoint(cancel)
            try: first, last = snap_range(item['axis'], cfg)
            except ValueError as exc: raise ValueError(f'{item["path"].name}: {exc}') from exc
            axis = item['axis']
            bounds.append(dict(file=item['path'].name, first=first, last=last,
                               lower=float(axis[first]), upper=float(axis[last]), native_points=last-first+1))
            emit('log', f'{item["path"].name}: 선택 경계 {axis[first]:.12g} ~ {axis[last]:.12g} cm^-1 ({last-first+1} native points)')
        arrays = []; targets = []; export_orders = []
        for item, corrected, bound in zip(self.items, self.asls_result['arrays'], bounds):
            checkpoint(cancel)
            first, last = bound['first'], bound['last']
            cut = slice(first, last+1)
            # Views only: preserve every selected native sample without interpolation.
            targets.append(item['axis'][cut])
            arrays.append(corrected[:,cut])
            order = item['export_order']
            export_orders.append(order[(order>=first) & (order<=last)]-first)
        checkpoint(cancel)
        emit('log', '범위 추출 완료: 파일별 원래 파수/point 수 유지. 보간이나 공통 범위 자르기 없음.')
        self.range_result = dict(params=replace(cfg), arrays=arrays, targets=targets,
                                 export_orders=export_orders, snapped_bounds=bounds,
                                 computed_at=datetime.now().isoformat(),
                                 asls_computed_at=self.asls_result['computed_at'])
        return self.range_result

    def run_blape(self, cfg=BLaPESettings(), emit=lambda *a: None, cancel=None, use_range=None):
        validate_blape(cfg)
        if self.asls_result is None:
            raise ValueError('AsLS 단계를 먼저 실행하세요.')
        self.blape_result = None; self.final_result = None
        if use_range is None: use_range = False
        if use_range and self.range_result is None:
            raise ValueError('범위 추출 단계를 먼저 실행하세요.')
        source = self.range_result if use_range else self.asls_result
        input_axes = source['targets'] if use_range else [v['axis'] for v in self.items]
        if cfg.sg_enabled:
            emit('log', f'Savitzky-Golay: baseline 후, BLaPE 전에 적용 · '
                        f'window={cfg.sg_window} native points · order={cfg.sg_order}')
        bands = None; bands_by_file = None
        if cfg.target_points is None:
            targets = []; bands_by_file = []
            for axis in input_axes:
                if cfg.exclude_ranges:
                    segments = modified_segments(axis,cfg)
                    if not segments:
                        raise ValueError('제외/경계 처리 후 유효한 원본 point가 없습니다.')
                    chunks = [axis[c:d] for a,b,c,d in segments]
                else:
                    if len(axis) < max(9,2*cfg.edge_trim+2):
                        raise ValueError('edge trim에 비해 원본 point가 부족합니다.')
                    chunks = [axis[cfg.edge_trim:len(axis)-cfg.edge_trim]]
                target = np.concatenate(chunks) if len(chunks)>1 else chunks[0]
                file_bands = []; offset = 0
                for chunk in chunks:
                    file_bands.append(dict(lower=float(chunk[0]),upper=float(chunk[-1]),
                                           start_index=offset,stop_index=offset+len(chunk)))
                    offset += len(chunk)
                targets.append(target); bands_by_file.append(file_bands)
            bands = bands_by_file[0] if cfg.exclude_ranges else None
            for item,target in zip(self.items,targets):
                emit('log',f'{item["path"].name}: BLaPE 원본 축 유지, {len(target)} points; 출력 보간 없음.')
        elif cfg.exclude_ranges:
            targets,bands = modified_targets(input_axes,cfg)
            emit('log', f'modified_BLaPE: 공통 유효 구간 {len(bands)}개, 최종 {len(targets[0])} points. 제외 경계 양쪽 Gaussian support + 4 points 보호.')
            for band in bands:
                emit('log', f'유효 출력: {band["lower"]:.8g} ~ {band["upper"]:.8g} cm^-1, {band["stop_index"]-band["start_index"]} points')
        else:
            targets = make_targets([dict(axis=x) for x in input_axes], 'single' if use_range else self.kind, cfg.target_points, cfg.edge_trim)
        arrays = []; total = sum(len(v['raw']) for v in self.items); done = 0; last = 0
        for item, corrected, axis, target in zip(self.items, source['arrays'], input_axes, targets):
            features = np.empty((len(corrected), len(target)))
            for i, row in enumerate(corrected):
                checkpoint(cancel)
                try:
                    features[i], _ = blape_from_corrected(row, axis, target, cfg)
                except Exception as exc:
                    raise ValueError(f'{item["path"].name}, spectrum {i+1}: {exc}') from exc
                done += 1; now = time.monotonic()
                if now-last >= .15 or done == total:
                    emit('progress', (100*done/total, f'BLaPE {done}/{total}')); last = now; time.sleep(0)
            arrays.append(features)
        checkpoint(cancel)
        self.blape_result = dict(params=replace(cfg), arrays=arrays, targets=targets,
                                 bands=bands, bands_by_file=bands_by_file, native_output=cfg.target_points is None, method='modified_BLaPE' if cfg.exclude_ranges else 'BLaPE',
                                 computed_at=datetime.now().isoformat(),
                                 asls_computed_at=self.asls_result['computed_at'],
                                 input_stage='Range' if use_range else 'AsLS',
                                 input_axes=input_axes,
                                 range_computed_at=self.range_result['computed_at'] if use_range else None)
        return self.blape_result

    def run_final(self, cfg=FinalRangeSettings(), emit=lambda *a: None, cancel=None):
        """Crop/resample completed BLaPE only; keep its normalization unchanged."""
        self.final_result = None
        if self.blape_result is None:
            raise ValueError('BLaPE를 먼저 실행하세요.')
        if not np.isfinite([cfg.lower,cfg.upper,cfg.step]).all() or cfg.lower>=cfg.upper or cfg.step<=0:
            raise ValueError('Final 범위는 유한한 하한 < 상한, 간격은 양수여야 합니다.')
        intervals = (cfg.upper-cfg.lower)/cfg.step
        if not np.isfinite(intervals) or intervals>200000 or intervals<1:
            raise ValueError('Final은 2~200001 point 범위로 설정하세요.')
        n = int(round(intervals))
        if not np.isclose(intervals,n,rtol=0,atol=1e-8):
            raise ValueError('상한-하한이 간격의 정수배여야 양끝을 포함할 수 있습니다. GUI 간격: 1 cm^-1.')
        requested = cfg.lower + np.arange(n+1,dtype=float)*cfg.step
        requested[-1] = cfg.upper
        result = self.blape_result; supports = [(cfg.lower,cfg.upper)]
        for fi,axis in enumerate(result['targets']):
            checkpoint(cancel)
            if cfg.lower < axis[0] or cfg.upper > axis[-1]:
                raise ValueError(f'{self.items[fi]["path"].name}: edge-trim 후 BLaPE 범위 '
                                 f'{axis[0]:.12g}~{axis[-1]:.12g} cm^-1. '
                                 'Final 범위를 줄이거나 더 넓게 측정하세요. 외삽하지 않습니다.')
            supports = intersect_supports(supports,[(v['lower'],v['upper']) for v in bands_for(result,fi)])
        keep = np.zeros(len(requested),dtype=bool)
        for lo,hi in supports:
            keep |= (requested>=lo) & (requested<=hi)
        indices = np.flatnonzero(keep)
        if len(indices)<2:
            raise ValueError('공통 유효 구간에 최종 격자점이 2개 미만입니다.')
        target = requested[indices]; bands=[]; offset=0
        for lo,hi in supports:
            chunk=requested[(requested>=lo)&(requested<=hi)]
            if not len(chunk): continue
            bands.append(dict(lower=float(chunk[0]),upper=float(chunk[-1]),
                              start_index=offset,stop_index=offset+len(chunk)))
            offset += len(chunk)
        arrays=[]; total=sum(len(a) for a in result['arrays']); done=0; last=0
        for fi,(axis,rows) in enumerate(zip(result['targets'],result['arrays'])):
            features=np.empty((len(rows),len(target)))
            for pi,row in enumerate(rows):
                checkpoint(cancel)
                features[pi]=interpolate_bands(target,axis,row,bands_for(result,fi))
                done+=1; now=time.monotonic()
                if now-last>=.15 or done==total:
                    emit('progress',(100*done/total,f'Final {done}/{total}'));last=now;time.sleep(0)
            arrays.append(features)
        checkpoint(cancel)
        self.final_result=dict(params=replace(cfg),arrays=arrays,targets=[target]*len(arrays),
                               bands=bands,method='BLaPE final range',
                               requested_points=len(requested),omitted_points=len(requested)-len(target),
                               computed_at=datetime.now().isoformat(),
                               blape_computed_at=result['computed_at'])
        emit('log',f'Final: 요청 {cfg.lower:g}~{cfg.upper:g}, 간격 {cfg.step:g} cm^-1; '
                   f'{len(target)} points. 제외/경계 보호로 생략: {len(requested)-len(target)} points. '
                   'BLaPE 재계산/재정규화 없음; 외삽/빈 구간 연결 없음.')
        return self.final_result

    def select(self, text, tolerance=1e-6):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError('좌표 tolerance는 0 이상의 유한한 수여야 합니다.')
        if not text.strip():
            return []
        if self.kind == 'single':
            requests = [(int(v),) for v in re.split(r'[\s,;]+', text.strip())]
        else:
            requests = []
            for line in text.splitlines():
                if not line.strip(): continue
                fields = line.split(',')
                if len(fields) not in (1, 2, 3):
                    raise ValueError('파일번호 또는 파일번호,spectrum번호 또는 파일번호,x,y를 입력하세요.')
                req = (int(fields[0]),) + tuple(float(v) for v in fields[1:])
                if not np.isfinite(req[1:]).all():
                    raise ValueError('좌표는 유한한 숫자여야 합니다.')
                requests.append(req)
        selected = []
        for req in requests:
            fi = req[0]-1
            if not 0 <= fi < len(self.items):
                raise ValueError(f'파일 번호는 1~{len(self.items)}입니다.')
            pi = 0; item = self.items[fi]
            if item['coords'] is not None:
                if len(req) != 3:
                    raise ValueError('좌표 mapping은 파일번호,x,y를 입력하세요.')
                distance = np.hypot(item['coords'][:, 0]-req[1], item['coords'][:, 1]-req[2])
                pi = int(np.argmin(distance))
                if distance[pi] > tolerance:
                    nearest = tuple(item['coords'][pi])
                    raise ValueError(f'{req}: 해당 좌표 없음. 가장 가까운 좌표: {nearest}')
            elif len(req) == 2:
                pi = int(req[1])-1
                if req[1] != pi+1 or not 0 <= pi < len(item['raw']):
                    raise ValueError(f'spectrum 번호는 1~{len(item["raw"])} 정수입니다.')
            elif len(req) != 1 or len(item['raw']) != 1:
                raise ValueError('좌표 없는 다중 spectrum은 파일번호,spectrum번호를 입력하세요.')
            selected.append((fi, pi))
        return selected

    def plot_data(self, stage, selected, cancel=None):
        checkpoint(cancel)
        if stage == 'Final':
            if self.final_result is None:
                raise ValueError('BLaPE 후 범위 추출(Final)을 먼저 실행하세요.')
            data=self.plot_data('BLaPE',selected,cancel)
            fi,pi=selected; result=self.final_result
            x,y=drawing_axis(result['targets'][fi],result['arrays'][fi][pi],bands_for(result,fi))
            data['series']=[v for v in data['series'] if v[0]!=2]+[(2,x,y,'Final BLaPE (uniform grid)')]
            data['stage']='Final'
            return data
        if self.asls_result is None or (stage == 'BLaPE' and self.blape_result is None) or (stage == 'Range' and self.range_result is None):
            raise ValueError(f'{stage} 단계를 먼저 실행하세요.')
        fi, pi = selected; item = self.items[fi]; axis = item['axis']
        corrected = self.asls_result['arrays'][fi][pi]; raw = item['raw'][pi]
        cfg = self.asls_result['params']
        clean = anomaly_impute(raw, cfg.anomaly_window, cfg.anomaly_threshold) if cfg.anomaly else raw
        baseline = clean-corrected
        series = [(0, axis, raw, 'Raw'), (0, axis, baseline, f'{cfg.method} baseline'),
                  (1, axis, corrected, f'{cfg.method} corrected')]
        if cfg.anomaly:
            series.insert(1, (0, axis, clean, 'Anomaly corrected'))
        if stage == 'Range':
            series = [(0, axis, corrected, 'AsLS full range'),
                      (1, self.range_result['targets'][fi], self.range_result['arrays'][fi][pi], 'Extracted AsLS (native points)')]
        if stage == 'BLaPE':
            result = self.blape_result
            source = self.range_result if result['input_stage']=='Range' else self.asls_result
            input_axis = result['input_axes'][fi]; input_row = source['arrays'][fi][pi]
            series = [v for v in series if v[0] != 1]
            series.append((1, input_axis, input_row, f'BLaPE input ({cfg.method} corrected)'))
            if result['params'].sg_enabled:
                sg_input = smooth_blape_input(input_row, input_axis, result['params'])
                series.append((1, input_axis, sg_input,
                               f'Savitzky-Golay (w={result["params"].sg_window}, n={result["params"].sg_order})'))
            smooth = (blape_from_corrected(input_row,input_axis,result['targets'][fi],result['params'])[1]
                      if result.get('bands') or result['params'].sg_enabled
                      else gaussian_filter1d(input_row, result['params'].sigma))
            plot_x = result['targets'][fi]; plot_y = result['arrays'][fi][pi]
            if result.get('bands'):
                # NaNs only for drawing: never draw a line across a masked gap.
                ends = [v['stop_index'] for v in bands_for(result,fi)[:-1]]
                plot_x = np.insert(plot_x,ends,np.nan)
                plot_y = np.insert(plot_y,ends,np.nan)
            series += [(1, input_axis, smooth, 'Gaussian smoothed'),
                       (2, plot_x, plot_y, result.get('method','BLaPE'))]
        checkpoint(cancel)
        excluded = []
        if stage == 'BLaPE':
            p = self.blape_result['params']
            excluded = [(lo-p.exclusion_margin,hi+p.exclusion_margin) for lo,hi in p.exclude_ranges]
        return dict(series=series, stage=stage, label=self.label(selected), excluded_ranges=excluded)

    def label(self, selected):
        fi, pi = selected; item = self.items[fi]
        suffix = '' if item['kind'] == 'single' else f' / {spectrum_name(item, pi)}'
        return f'{fi+1}. {item["path"].name}{suffix}'

    def similarity_data(self, stage, selected, cancel=None):
        """Cosine similarity for manually selected spectra.

        AsLS keeps its native axes, so spectra on unequal axes are compared only
        inside their common measured range, after linear interpolation onto the
        first selected spectrum's native points.  No extrapolation is used.
        """
        if stage not in ('AsLS', 'Range', 'BLaPE', 'Final'):
            raise ValueError('유사도 단계는 AsLS, Range 또는 BLaPE이어야 합니다.')
        if self.asls_result is None or (stage == 'BLaPE' and self.blape_result is None) or (stage == 'Range' and self.range_result is None):
            raise ValueError(f'{stage} 단계를 먼저 실행하세요.')
        if stage=='Final' and self.final_result is None:
            raise ValueError('Final을 먼저 실행하세요.')
        selected = list(dict.fromkeys(selected))
        if len(selected) < 2:
            raise ValueError('유사도를 보려면 서로 다른 spectrum을 2개 이상 선택하세요.')
        if len(selected) > 100:
            raise ValueError('한 번에 비교할 spectrum은 100개 이하로 선택하세요.')
        snapshot = {'AsLS': self.asls_result, 'Range': self.range_result, 'BLaPE': self.blape_result, 'Final': self.final_result}[stage]
        axes = [np.asarray(self.items[fi]['axis'] if stage == 'AsLS' else snapshot['targets'][fi], dtype=float)
                for fi, _ in selected]
        rows = [np.asarray(snapshot['arrays'][fi][pi], dtype=float) for fi, pi in selected]
        lo = max(float(axis[0]) for axis in axes)
        hi = min(float(axis[-1]) for axis in axes)
        if not lo < hi:
            raise ValueError('선택한 spectrum들 사이에 공통 파수 범위가 없습니다.')
        reference_axis = axes[0]
        mask = (reference_axis >= lo) & (reference_axis <= hi)
        target = reference_axis[mask]
        if len(target) < 2:
            raise ValueError('공통 파수 범위에 비교 가능한 point가 부족합니다.')
        if stage in ('BLaPE','Final'):
            supports=[(lo,hi)]
            for fi,_ in selected:
                supports=intersect_supports(supports,[(v['lower'],v['upper']) for v in bands_for(snapshot,fi)])
            valid=np.zeros(len(target),dtype=bool)
            for a,b in supports: valid |= (target>=a)&(target<=b)
            target=target[valid]
            if len(target)<2: raise ValueError('공통 유효 구간에 비교 point가 부족합니다.')
            aligned=np.vstack([interpolate_bands(target,axis,row,bands_for(snapshot,fi))
                               for (fi,_),axis,row in zip(selected,axes,rows)])
        else:
            aligned = np.vstack([np.interp(target, axis, row) for axis, row in zip(axes, rows)])
        norms = np.linalg.norm(aligned, axis=1)
        if np.any(norms <= np.finfo(float).eps):
            bad = self.label(selected[int(np.flatnonzero(norms <= np.finfo(float).eps)[0])])
            raise ValueError(f'0에 가까운 spectrum은 cosine similarity를 계산할 수 없습니다: {bad}')
        similarity = aligned @ aligned.T / np.outer(norms, norms)
        similarity = np.clip(similarity, -1.0, 1.0)
        checkpoint(cancel)
        return dict(stage=stage, selected=selected, labels=[self.label(v) for v in selected],
                    matrix=similarity, common_range=(float(target[0]), float(target[-1])),
                    points=len(target), resampled=not all(np.array_equal(axis, axes[0]) for axis in axes))

    def save(self, choice, output_root, emit=lambda *a: None, cancel=None):
        return save_session(self, choice, output_root, emit, cancel)

# %% 6. CSV 저장 — AsLS / BLaPE / Both, 계산/그림과 독립

def save_session(session, choice, output_root, emit=lambda *a: None, cancel=None):
    if choice not in ('AsLS', 'Range', 'BLaPE', 'Final', 'Both', 'All'):
        raise ValueError('저장 대상은 AsLS/Range/BLaPE/Final/Both/All 중 하나입니다.')
    stages = ({'Both':['AsLS','BLaPE'],'All':[k for k,v in [('AsLS',session.asls_result),('Range',session.range_result),('BLaPE',session.blape_result),('Final',session.final_result)] if v is not None]}.get(choice,[choice]))
    for stage in stages:
        if session.asls_result is None or (stage == 'BLaPE' and session.blape_result is None) or (stage == 'Range' and session.range_result is None):
            raise ValueError(f'{stage} 단계를 먼저 실행하세요. 자동 계산하지 않습니다.')
    if 'Final' in stages and session.final_result is None:
        raise ValueError('Final을 먼저 실행하세요. 자동 계산하지 않습니다.')
    checkpoint(cancel)
    kept = {fi: [pi for pi in range(len(item['raw']))
                 if (fi, pi) not in session.excluded_pixels]
            for fi, item in enumerate(session.items)}
    included_files = [fi for fi, indices in kept.items() if indices]
    if not included_files:
        raise ValueError('모든 파일이 저장 제외되었습니다. 최소 1개 파일을 복원하세요.')
    folder = Path(output_root).expanduser()/f'{session.kind}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    folder.mkdir(parents=True, exist_ok=False)
    emit('log', f'저장 폴더: {folder}')
    metadata = dict(app_version=VERSION, status='running', save_choice=choice,
                    input_type=session.kind, input_files=[str(v['path']) for v in session.items],
                    input_settings=asdict(session.input_settings), mapping_input='A=y; B=x',
                    asls_parameters=asdict(session.asls_result['params']),
                    asls_computed_at=session.asls_result['computed_at'],
                    baseline_method=session.asls_result['params'].method,
                    baseline_parameters=asdict(session.asls_result['params']),
                    processing_order=f"optional anomaly -> {session.asls_result['params'].method} -> native Gaussian/BLaPE + edge trim -> optional Final crop/interpolation",
                    sigma_unit='native input-axis points; Range does not resample', output_ranges={},
                    asls_export='Original wavenumber values/order/count; no interpolation',
                    asls_single_layout=('not applicable; one CSV per input file' if session.kind != 'single'
                        else 'shared axis' if all(np.array_equal(x,session.asls_result['targets'][0])
                        for x in session.asls_result['targets']) else 'per-file wavenumber/intensity pairs; blank padding'))
    metadata['detected_inputs'] = [dict(file=item['path'].name, kind=item['kind'],
        input_format=item.get('input_format'), spectra=len(item['raw']), points=len(item['axis']),
        coordinates_available=item['coords'] is not None) for item in session.items]
    if 'Range' in stages or (any(v in stages for v in ('BLaPE','Final')) and session.blape_result['input_stage']=='Range'):
        metadata.update(range_parameters=asdict(session.range_result['params']),
                        range_computed_at=session.range_result['computed_at'],
                        range_snapped_bounds=session.range_result['snapped_bounds'],
                        range_output=[[float(x[0]),float(x[-1]),len(x)] for x in session.range_result['targets']],
                        range_rule='strictly greater for both bounds, inclusive endpoints; native values/count/order in CSV; no interpolation or common-range truncation')
    if any(v in stages for v in ('BLaPE','Final')):
        sg = session.blape_result['params']
        metadata['smoothing_parameters'] = dict(method='Savitzky-Golay', enabled=sg.sg_enabled,
            window_length=sg.sg_window, polyorder=sg.sg_order, deriv=0, mode='interp',
            stage='after baseline correction, before BLaPE', axis='native data points',
            excluded_intervals='filtered as separate continuous segments')
        if sg.sg_enabled:
            metadata['processing_order'] = metadata['processing_order'].replace(
                ' -> native Gaussian/BLaPE', ' -> Savitzky-Golay -> native Gaussian/BLaPE')
        metadata['blape_input_stage'] = session.blape_result['input_stage']
        metadata.update(blape_parameters=asdict(session.blape_result['params']),
                        blape_computed_at=session.blape_result['computed_at'])
        metadata['blape_method'] = session.blape_result.get('method','BLaPE')
        metadata['blape_valid_bands'] = [bands_for(session.blape_result,fi) for fi in range(len(session.items))]
        metadata['blape_native_output'] = session.blape_result.get('native_output',False)
        metadata['masked_axis_instruction'] = 'Use the stored wavenumbers verbatim. Do not interpolate across gaps. Match axes and masks between reference/train/test.'
        metadata['modified_blape_normalization'] = 'Mean abs from trusted native samples; one maximum across all retained output bands; per spectrum.'
    if 'Final' in stages:
        metadata.update(final_parameters=asdict(session.final_result['params']),
                        final_computed_at=session.final_result['computed_at'],
                        final_bands=session.final_result['bands'],
                        final_omitted_points=session.final_result['omitted_points'],
                        final_rule='Linear interpolation of completed BLaPE; inclusive fixed grid; no extrapolation or interpolation across gaps; no renormalization')
    metadata['excluded_mapping_pixels'] = [
        dict(file=session.items[fi]['path'].name, spectrum_index_1based=pi+1,
             x=float(session.items[fi]['coords'][pi,0]), y=float(session.items[fi]['coords'][pi,1]))
        for fi, pi in sorted(session.excluded_pixels) if session.items[fi]['coords'] is not None]
    metadata['excluded_single_files'] = [session.items[fi]['path'].name
        for fi, pi in sorted(session.excluded_pixels) if session.items[fi]['kind'] == 'single']
    metadata['excluded_matrix_spectra'] = [dict(file=session.items[fi]['path'].name,
        spectrum_index_1based=pi+1) for fi, pi in sorted(session.excluded_pixels)
        if session.items[fi]['kind'] == 'matrix']
    metadata['exported_spectra_per_file'] = {session.items[fi]['path'].name: len(v) for fi,v in kept.items()}
    def settings():
        with (folder/'parameters.csv').open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f); w.writerow(['parameter', 'value'])
            for k, v in metadata.items():
                w.writerow([k, json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v])

    def write_header(writer, indices):
        item = session.items[indices[0]]
        if len(indices) == 1 and item['coords'] is not None:
            coords = item['coords'][kept[indices[0]]]
            writer.writerow(['x coordinate']+coords[:,0].tolist())
            writer.writerow(['y coordinate']+coords[:,1].tolist())
        writer.writerow(['Raman shift (cm^-1)']+[spectrum_name(session.items[fi], pi)
            for fi in indices for pi in kept[fi]])

    def write_native(path, indices, stage):
        snapshot = {'AsLS':session.asls_result,'Range':session.range_result,'BLaPE':session.blape_result}[stage]
        orders = ([item['export_order'] for item in session.items] if stage=='AsLS'
                  else snapshot['export_orders'] if stage=='Range'
                  else [np.arange(len(x)) for x in snapshot['targets']])
        axes = [snapshot['targets'][i] if stage=='AsLS' else snapshot['targets'][i][orders[i]] for i in indices]
        shared = all(np.array_equal(axis, axes[0]) for axis in axes)
        partial = path.with_suffix('.csv.partial')
        with partial.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            if shared:
                write_header(writer, indices)
            else:
                writer.writerow([label for i in indices for label in
                    (f'{session.items[i]["path"].name} | wavenumber', f'{session.items[i]["path"].name} | {session.asls_result["params"].method if stage == "AsLS" else stage}')])
            count = max(len(axis) for axis in axes)
            for j in range(count):
                if j % 32 == 0: checkpoint(cancel)
                if shared:
                    row = [axes[0][j]]+[snapshot['arrays'][fi][pi,orders[fi][j]]
                                       for fi in indices for pi in kept[fi]]
                else:
                    row = []
                    for fi, axis in zip(indices, axes):
                        row.extend([axis[j],snapshot['arrays'][fi][0,orders[fi][j]]]
                                   if j < len(axis) else ['', ''])
                writer.writerow(row)
                if j % 128 == 0 or j == count-1:
                    emit('progress',(100*(j+1)/count, f'{path.name}: {j+1}/{count} rows'))
        checkpoint(cancel); partial.rename(path)

    def write_result(path, stage, indices, target):
        if stage in ('AsLS','Range','BLaPE'):
            return write_native(path, indices, stage)
        partial = path.with_suffix('.csv.partial')
        snapshot = {'AsLS':session.asls_result,'Range':session.range_result,'BLaPE':session.blape_result,'Final':session.final_result}[stage]
        with partial.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            write_header(writer, indices)
            # Final is already sampled on the common physical output axis.
            count = sum(len(kept[i]) for i in indices)
            for start in range(0, len(target), 64):
                checkpoint(cancel)
                stop = min(start+64, len(target))
                block = np.empty((stop-start, count)); col = 0
                for fi in indices:
                    for pi in kept[fi]:
                        row = snapshot['arrays'][fi][pi]
                        if col % 128 == 0: checkpoint(cancel)
                        block[:, col] = row[start:stop]
                        col += 1
                for offset, values in enumerate(block):
                    writer.writerow([target[start+offset]]+values.tolist())
                emit('progress', (100*stop/len(target), f'{path.name}: {stop}/{len(target)} rows'))
        checkpoint(cancel); partial.rename(path)
    try:
        settings()
        for name in ('BLaPE','Final'):
            if name not in stages: continue
            snapshot=session.blape_result if name=='BLaPE' else session.final_result
            manifest=dict(stage=name,parameters=asdict(snapshot['params']),
                          no_interpolation_across_gaps=True,
                          files=[dict(file=item['path'].name,
                                      wavenumbers=snapshot['targets'][fi].tolist(),
                                      bands=bands_for(snapshot,fi))
                                 for fi,item in enumerate(session.items)],
                          note='Use the exact stored axis and mask for train/test/reference.')
            if name=='Final':
                manifest.update(wavenumbers=snapshot['targets'][0].tolist(),bands=snapshot['bands'],
                                omitted_points=snapshot['omitted_points'])
            with (folder/f'{name}_axis_manifest.json').open('w',encoding='utf-8') as stream:
                json.dump(manifest,stream,indent=2,ensure_ascii=False)
        for stage in stages:
            snapshot = {'AsLS':session.asls_result,'Range':session.range_result,'BLaPE':session.blape_result,'Final':session.final_result}[stage]
            metadata['output_ranges'][stage] = [[float(x[0]), float(x[-1]), len(x)] for x in snapshot['targets']]
            export_name = session.asls_result['params'].method if stage == 'AsLS' else stage
            if session.kind == 'single':
                write_result(folder/f'single_{export_name}_spectra.csv', stage, included_files, snapshot['targets'][included_files[0]])
            else:
                for fi in included_files:
                    item = session.items[fi]
                    # Full input filename avoids same-stem TXT/CSV collisions.
                    write_result(folder/f'{item["path"].name}_{item["kind"]}_{export_name}.csv', stage, [fi], snapshot['targets'][fi])
        checkpoint(cancel); metadata['status'] = 'complete'; settings()
        return folder
    except BaseException as exc:
        metadata.update(status='cancelled' if isinstance(exc, Cancelled) else 'failed', error=str(exc))
        settings(); raise

# Original BLaPE project license (retained for included numerical code):
# MIT License
# 
# Copyright (c) 2024 [Your Name]
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE. 
