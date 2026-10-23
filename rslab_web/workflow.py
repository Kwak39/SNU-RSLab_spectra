"""Persistent session inputs and array-based preprocessing/ML handoff."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import nnls_core, svm_core
from . import preprocess_core

INPUT_PATCH = 'PIPE-2.6.1'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class UploadStore:
    """Keep raw files on session-local disk; store only metadata in memory."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.revision = 0

    def add(self, files):
        added, duplicates = [], 0
        used = {entry['name'].casefold() for entry in self.entries.values()}
        for filename, payload in files:
            original = Path(filename.replace('\\', '/')).name
            if original in ('', '.', '..'):
                raise ValueError('유효하지 않은 파일명입니다.')
            digest = hashlib.sha256(payload).hexdigest()
            uid = hashlib.sha256((original.casefold() + '\0' + digest).encode()).hexdigest()
            if uid in self.entries:
                duplicates += 1
                continue
            path = Path(original)
            name, suffix = original, 2
            while name.casefold() in used:
                name = f'{path.stem}__{suffix}{path.suffix}'
                suffix += 1
            destination = self.directory / name
            destination.write_bytes(payload)
            self.entries[uid] = dict(name=name, original=original, path=destination,
                                     size=len(payload), sha256=digest)
            used.add(name.casefold())
            added.append(uid)
        if added:
            self.revision += 1
        return added, duplicates

    def remove(self, ids):
        removed = 0
        for uid in set(ids):
            entry = self.entries.pop(uid, None)
            if entry is not None:
                entry['path'].unlink(missing_ok=True)
                removed += 1
        if removed:
            self.revision += 1
        return removed

    def clear(self):
        return self.remove(list(self.entries))

    @property
    def paths(self):
        return [entry['path'] for entry in self.entries.values()]

    @property
    def size(self):
        return sum(entry['size'] for entry in self.entries.values())


@dataclass
class SpectralPool:
    data: svm_core.Dataset
    signature: str
    provenance: dict
    row_meta: list


def _readonly(array):
    view = np.asarray(array).view()
    view.setflags(write=False)
    return view


def make_pool(session, stage):
    result = {'Raw': dict(arrays=[v['raw'] for v in session.items]),
              'AsLS': session.asls_result, 'Smoothing': session.smoothing_result, 'Range': session.range_result,
              'BLaPE': session.blape_result, 'Final': getattr(session, 'final_result', None)}.get(stage)
    if result is None:
        raise ValueError(f'{stage} 결과가 없습니다.')
    axes, blocks, ids, sources, names, groups, row_meta = [], [], [], [], [], [], []
    for fi, item in enumerate(session.items):
        kept = [pi for pi in range(len(item['raw'])) if (fi, pi) not in session.excluded_pixels]
        if not kept:
            continue
        axis = item['axis'] if stage in ('Raw', 'AsLS', 'Smoothing') else result['targets'][fi]
        axes.append(axis)
        block = result['arrays'][fi]
        blocks.append(block if len(kept) == len(block) else block[kept])
        source = item['path'].name
        for pi in kept:
            mapping = item['coords'] is not None
            x, y = item['coords'][pi] if mapping else (float('nan'), float('nan'))
            name = preprocess_core.spectrum_name(item, pi)
            ids.append(f'{source} | {name}')
            sources.append(source)
            names.append(name)
            # File defaults preserve independent preparation units during splitting.
            groups.append(source)
            row_meta.append(dict(x=float(x), y=float(y), mapping=mapping,
                                 input_kind=item['kind'], spectrum_index_1based=pi+1))
    if not blocks:
        raise ValueError('전달할 spectrum이 없습니다.')
    for axis in axes[1:]:
        try:
            svm_core.match_axis(axes[0], axis)
        except ValueError as exc:
            raise ValueError('전달할 파일들의 파수축이 다릅니다. Final 범위 추출로 공통 범위·간격을 맞춘 뒤 Final 결과를 전달하세요. '
                             f'상세: {exc}') from exc
    matrix = blocks[0] if len(blocks) == 1 else np.vstack(blocks)
    axis = _readonly(axes[0])
    data = svm_core.Dataset(axis, _readonly(matrix), ids, sources, names, groups)
    blape = session.blape_result if stage in ('BLaPE', 'Final') else None
    lineage = preprocess_core.stage_lineage(session, stage)
    provenance = dict(stage=stage, processing_steps=lineage, asls=asdict(session.asls_result['params']),
                      smoothing=asdict(session.smoothing_result['params']) if session.smoothing_result and
                      'Smoothing' in lineage else None,
                      range=asdict(session.range_result['params']) if session.range_result and
                      (stage == 'Range' or (blape and blape.get('input_stage') == 'Range')) else None,
                      blape=asdict(blape['params']) if blape else None,
                      final=asdict(result['params']) if stage == 'Final' else None,
                      final_source=result.get('input_stage') if stage == 'Final' else None,
                      bands=result.get('bands') if stage in ('BLaPE','Final') else None,
                      inputs=[dict(file=item['path'].name, kind=item['kind'],
                                   spectra=len(item['raw']), points=len(item['axis']),
                                   data_start_line=item.get('data_start_line'),
                                   leading_metadata=item.get('leading_metadata', []),
                                   coordinates_available=item['coords'] is not None)
                              for item in session.items])
    digest = hashlib.sha256()
    for array in (axis, matrix):
        # No second, full matrix-sized bytes object when the array is contiguous.
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    digest.update('\0'.join(ids).encode())
    digest.update(repr(provenance).encode())
    return SpectralPool(data, digest.hexdigest(), provenance, row_meta)


def subset_packet(pool, indices):
    indices = np.asarray(indices, dtype=int)
    if not len(indices):
        return None
    if len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= len(pool.data.X):
        raise ValueError('Spectrum 선택 인덱스가 유효하지 않습니다.')
    if np.all(np.diff(indices) == 1):
        matrix = pool.data.X[indices[0]:indices[-1]+1]
    else:
        matrix = _readonly(pool.data.X[indices])
    take = lambda values: [values[i] for i in indices]
    data = svm_core.Dataset(pool.data.axis, matrix, take(pool.data.sample_ids),
                            take(pool.data.sources), take(pool.data.names), take(pool.data.groups))
    sig = hashlib.sha256((pool.signature + ':' + ','.join(map(str, indices))).encode()).hexdigest()
    # A shared source file can legitimately contain disjoint TRAIN/TEST rows.
    # Exact sample IDs are still checked by the classifier; groups are reported.
    return dict(data=data, signature=sig, hashes=[], provenance=pool.provenance,
                pool_signature=pool.signature, indices=indices.tolist())


def nnls_packet(session, stage):
    """Direct transfer also supports native, unequal per-file axes for NNLS."""
    result = {'Raw': dict(arrays=[v['raw'] for v in session.items]),
              'AsLS': session.asls_result, 'Smoothing': session.smoothing_result, 'Range': session.range_result,
              'BLaPE': session.blape_result, 'Final': getattr(session, 'final_result', None)}[stage]
    if result is None:
        raise ValueError(f'{stage} 결과가 없습니다.')
    spectra = []
    digest = hashlib.sha256(stage.encode())
    for fi, item in enumerate(session.items):
        axis = item['axis'] if stage in ('Raw', 'AsLS', 'Smoothing') else result['targets'][fi]
        for pi, signal in enumerate(result['arrays'][fi]):
            if (fi, pi) in session.excluded_pixels:
                continue
            mapping = item['coords'] is not None
            x, y = item['coords'][pi] if mapping else (float('nan'), float('nan'))
            name = preprocess_core.spectrum_name(item, pi)
            spectrum = nnls_core.Spectrum(_readonly(axis), _readonly(signal), name,
                                          item['path'].name, float(x), float(y), mapping)
            spectra.append(spectrum)
            digest.update(memoryview(np.ascontiguousarray(signal)).cast('B'))
            digest.update(memoryview(np.ascontiguousarray(axis)).cast('B'))
            digest.update((spectrum.source + name).encode())
    return dict(data=spectra, signature=digest.hexdigest(), hashes=[])


def nnls_from_subset(pool, packet):
    if packet is None:
        return None
    if 'nnls_packet' in packet:
        return packet['nnls_packet']
    data = packet['data']
    spectra = []
    for i, original in enumerate(packet['indices']):
        meta = pool.row_meta[original]
        spectra.append(nnls_core.Spectrum(data.axis, data.X[i], data.names[i], data.sources[i],
                                          meta['x'], meta['y'], meta['mapping']))
    result = dict(packet, data=spectra)
    packet['nnls_packet'] = result
    return result


def new_assignments(pool):
    return dict(pool_signature=pool.signature, roles=['미지정'] * len(pool.data.X),
                labels=[''] * len(pool.data.X), groups=list(pool.data.groups), revision=0,
                packets={})


def apply_file_assignments(pool, state, records):
    by_source = {}
    for i, source in enumerate(pool.data.sources):
        by_source.setdefault(source, []).append(i)
    for row in records:
        role = str(row.get('set', '변경 안 함')).strip()
        if role not in ('변경 안 함', 'TRAIN', 'TEST', '제외', '미지정'):
            raise ValueError('분배 값은 TRAIN / TEST / 제외입니다.')
        for i in by_source.get(row['file'], []):
            if role != '변경 안 함':
                state['roles'][i] = role
            for field in ('label', 'group'):
                value = row.get(field, '')
                if isinstance(value, str) and value.strip():
                    state['labels' if field == 'label' else 'groups'][i] = value.strip()
    state['revision'] += 1


def assignment_packets(pool, state):
    result = []
    for role in ('TRAIN', 'TEST'):
        indices = [i for i, value in enumerate(state['roles']) if value == role]
        cached = state['packets'].get(role)
        if cached is None or cached.get('indices') != indices:
            packet = subset_packet(pool, indices)
            state['packets'][role] = packet
        else:
            packet = cached
        labels = dict(labels=[state['labels'][i] for i in indices],
                      groups=[state['groups'][i] for i in indices])
        result.extend([packet, labels])
    return tuple(result)


def clear_results(state, tool):
    for suffix in ('result', 'export'):
        state.pop(f'{tool}_{suffix}', None)


def clear_ml(state):
    for key in list(state):
        if key == 'ml_pool' or key.startswith(('svm_', 'knn_', 'nnls_', 'bridge_')):
            del state[key]


def import_assignments(pool, state, rows):
    ids = {sid: i for i, sid in enumerate(pool.data.sample_ids)}
    seen, changes = set(), []
    for row in rows:
        sid = row.get('sample_id', '')
        if sid in seen:
            raise ValueError('CSV에 중복 sample_id가 있습니다.')
        seen.add(sid)
        if sid not in ids:
            continue
        role = row.get('set', '').strip()
        if role.lower() in ('reference', 'refs'):
            role = 'TRAIN'
        elif role.lower() in ('train', 'test'):
            role = role.upper()
        if role and role not in ('TRAIN', 'TEST', '제외', '미지정'):
            raise ValueError('set 열은 TRAIN / TEST / 제외 / 미지정이어야 합니다.')
        changes.append((ids[sid], row, role))
    if not changes:
        raise ValueError('현재 데이터와 일치하는 sample_id가 없습니다.')
    for i, row, role in changes:
        if role:
            state['roles'][i] = role
        if 'label' in row:
            state['labels'][i] = row['label'].strip()
        if row.get('group', '').strip():
            state['groups'][i] = row['group'].strip()
    state['revision'] += 1
    return len(changes)


def assignment_rows(pool, state, unit='파일', role='전체', query=''):
    """Metadata-only table; indices remain tied to the immutable pool order."""
    if state['pool_signature'] != pool.signature:
        raise ValueError('현재 전처리 데이터와 Labeling 설정이 일치하지 않습니다.')
    if unit not in ('파일', 'Spectrum'):
        raise ValueError('선택 단위가 유효하지 않습니다.')
    if role not in ('전체', '미지정', 'TRAIN', 'TEST', '제외'):
        raise ValueError('Set 필터가 유효하지 않습니다.')
    query = query.strip().casefold()
    rows = []
    if unit == '파일':
        by_source = {}
        for i, name in enumerate(pool.data.sources):
            by_source.setdefault(name, []).append(i)
        groups = by_source.values()
    else:
        groups = ([i] for i in range(len(pool.data.X)))
    for indices in groups:
        roles = {state['roles'][i] for i in indices}
        labels = {state['labels'][i] for i in indices}
        # A file filter selects the entire matching file, never a hidden subset.
        if role != '전체' and role not in roles:
            continue
        name = pool.data.sources[indices[0]]
        spectrum = pool.data.names[indices[0]] if unit == 'Spectrum' else ''
        if query and query not in ' '.join([name, spectrum, *labels]).casefold():
            continue
        row = dict(file=name, spectra=len(indices),
                   set=next(iter(roles)) if len(roles) == 1 else '혼합',
                   label=next(iter(labels)) if len(labels) == 1 else '여러 label',
                   _indices=indices)
        if unit == 'Spectrum':
            row.update(sample_id=pool.data.sample_ids[indices[0]], spectrum=spectrum)
        rows.append(row)
    return rows


def apply_selection(pool, state, indices, role, label=''):
    """Validate a complete bulk move before mutating roles or labels."""
    if state['pool_signature'] != pool.signature:
        raise ValueError('현재 전처리 데이터와 Labeling 설정이 일치하지 않습니다.')
    if role not in ('TRAIN', 'TEST', '미지정', '제외'):
        raise ValueError('Set은 TRAIN / TEST / 미지정 / 제외입니다.')
    if not isinstance(label, str):
        raise ValueError('Label에는 물질명을 입력하세요.')
    chosen = list(indices)
    if not chosen or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
                         or i < 0 or i >= len(pool.data.X) for i in chosen):
        raise ValueError('적용할 파일 또는 spectrum을 선택하세요.')
    chosen = sorted(set(chosen))
    label = label.strip()
    for i in chosen:
        state['roles'][i] = role
        if label:
            state['labels'][i] = label
    state['revision'] += 1
    return len(chosen)

