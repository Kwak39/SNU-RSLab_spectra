"""Web adapters around the uploaded numerical algorithms."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import sklearn

from . import knn_core, nnls_core, svm_core


def train_knn(data, labels, groups, config):
    labels = [label.strip() for label in labels]
    if len(labels) != len(data.X) or any(not label for label in labels):
        raise ValueError('모든 TRAIN spectrum에 label을 지정하세요.')
    if len(set(labels)) < 2:
        raise ValueError('최소 두 class가 필요합니다.')
    if config['k'] > len(data.X):
        raise ValueError('k는 TRAIN spectrum 수보다 클 수 없습니다.')
    if np.any(np.linalg.norm(data.X, axis=1) == 0):
        raise ValueError('TRAIN에 완전히 0인 spectrum이 있습니다.')
    model = knn_core.make_model(config['k'], config['normalization'])
    model.fit(data.X, labels)
    unique, counts = np.unique(labels, return_counts=True)
    metadata = dict(config=config, app_version='WebSuite 1.0',
                    sklearn_version=sklearn.__version__,
                    created=datetime.now(timezone.utc).isoformat(),
                    train_total=len(labels), train_used=len(labels),
                    classes=unique.tolist(), counts_used=dict(zip(unique.tolist(), counts.tolist())),
                    calibration='Distance-weighted neighbor votes; not calibrated probabilities',
                    selected_training_ids=list(data.sample_ids))
    return dict(format='RSLabKNN1', model=model, axis=data.axis.copy(), metadata=metadata,
                train_ids=list(data.sample_ids), train_sources=sorted(set(data.sources)),
                train_groups=sorted(set(groups)))


def predict(bundle, data, labels, groups, hashes=()):
    if set(bundle.get('content_hashes', [])) & set(hashes):
        raise ValueError('TRAIN과 TEST에 내용이 동일한 파일이 있습니다. 독립 TEST를 선택하세요.')
    # Unified exports reuse generic filenames for independent TRAIN and TEST.
    # Keep exact sample-ID and content checks; a basename alone is not identity.
    result = svm_core.predict(dict(bundle, train_sources=[]), data, labels, groups)
    if bundle['format'] == 'RSLabKNN1':
        result['scores']['notes'].append('k-NN 점수는 거리 가중 투표 비율입니다. 보정된 확률이 아닙니다.')
    return result


def fit_nnls(refs, tests, alpha=0.01, policy='자동 (mapping만 파일별 평균)',
             selected=None, strict=True, progress=lambda text: None):
    templates, names = nnls_core.reference_templates(refs, policy)
    if selected is not None:
        indices = list(selected)
        templates = [templates[i] for i in indices]
        names = [names[i] for i in indices]
    if not templates:
        raise ValueError('후보 reference를 하나 이상 선택하세요.')
    if not tests:
        raise ValueError('TEST spectrum을 올리세요.')
    spectra = templates + tests
    if strict:
        for spectrum in spectra[1:]:
            svm_core.match_axis(spectra[0].axis, spectrum.axis)
        # Canonicalize only <=1e-6 round-off, preserving the entire masked grid.
        aligned = [nnls_core.checked(spectra[0].axis.copy(), s.y.copy(), s.name,
                                    s.source, s.x, s.coord_y, s.mapping) for s in spectra]
        result = nnls_core.fit_all(aligned[:len(templates)], aligned[len(templates):],
                                   alpha, '각 spectrum', progress)
    else:
        # Interpolation is opt-in for continuous axes only. Never bridge BLaPE gaps.
        for spectrum in spectra:
            delta = np.diff(spectrum.axis)
            if np.any(delta > 5 * np.median(delta)):
                raise ValueError('불연속 파수축에서는 보간을 사용할 수 없습니다. 동일 축 모드를 사용하세요.')
        result = nnls_core.fit_all(templates, tests, alpha, '각 spectrum', progress)
    result['names'] = names
    result['policy'] = policy
    result['strict_axis'] = strict
    # Rebuild pair names after optional candidate selection.
    similarity = result['A'].T @ result['A']
    result['similar_pairs'] = [(names[i], names[j], float(similarity[i, j]))
                              for i in range(len(names)) for j in range(i + 1, len(names))
                              if similarity[i, j] > .98]
    return result
