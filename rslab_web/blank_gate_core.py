"""Reference similarity screening with explicit human confirmation, per spectrum."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import preprocess_core as core

INPUT_PATCH = 'GATE-2.7.1'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
REFERENCE_EXTENSIONS = {'.txt', '.csv', '.dat', '.tsv'}


def discover_references(folder):
    """Discover arbitrary filenames; labels live separately in labels.json."""
    folder = Path(folder)
    settings_path = folder/'labels.json'
    settings = {}
    if settings_path.is_file():
        try:
            document = json.loads(settings_path.read_text(encoding='utf-8-sig'))
            if document.get('schema_version') != 1 or not isinstance(document.get('references'), dict):
                raise ValueError('schema_version=1 / references 객체가 필요합니다.')
            settings = document['references']
        except (OSError, ValueError, AttributeError) as exc:
            raise ValueError(f'labels.json 읽기 실패: {exc}') from exc
    records = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()) if folder.is_dir() else []:
        if not path.is_file() or path.suffix.casefold() not in REFERENCE_EXTENSIONS:
            continue
        entry = settings.get(path.name, {})
        if not isinstance(entry, dict) or not isinstance(entry.get('label', ''), str) or not isinstance(entry.get('enabled', False), bool):
            raise ValueError(f'{path.name}: Label / 사용 설정 형식을 확인하세요.')
        records.append(dict(key='folder:'+path.name, filename=path.name, payload=path.read_bytes(),
                            label=entry.get('label', '').strip(), enabled=entry.get('enabled', False), source='폴더'))
    return records


def add_uploaded_references(records, uploads):
    """Preserve folder entries; identify uploads by filename and content."""
    result = [dict(record) for record in records]
    existing = {(record['filename'].casefold(), hashlib.sha256(record['payload']).hexdigest()) for record in records}
    for filename, payload in uploads:
        name = Path(filename.replace('\\', '/')).name
        if not name or Path(name).suffix.casefold() not in REFERENCE_EXTENSIONS:
            raise ValueError('Reference는 TXT / CSV / DAT / TSV 파일이어야 합니다.')
        digest = hashlib.sha256(payload).hexdigest()
        token = (name.casefold(), digest)
        if token in existing:
            continue
        existing.add(token)
        result.append(dict(key='upload:'+name+':'+digest, filename=name, payload=payload,
                           label='', enabled=False, source='이번 세션 업로드'))
    return result


def reference_bundle(records):
    """One portable folder with original spectra and a separate label manifest."""
    import io
    import zipfile
    used, settings = set(), {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            name = Path(record['filename'].replace('\\', '/')).name
            if Path(name).suffix.casefold() not in REFERENCE_EXTENSIONS:
                raise ValueError('유효하지 않은 reference 확장자입니다.')
            candidate = name
            index = 2
            while candidate.casefold() in used:
                candidate = f'{Path(name).stem}__{index}{Path(name).suffix}'
                index += 1
            used.add(candidate.casefold())
            label = record['label'].strip()
            enabled = bool(record['enabled'])
            if enabled and not label:
                raise ValueError('사용할 reference의 Label을 입력하세요.')
            archive.writestr('blank_references/'+candidate, record['payload'])
            settings[candidate] = dict(label=label, enabled=enabled)
        archive.writestr('blank_references/labels.json',
            json.dumps(dict(schema_version=1, references=settings), ensure_ascii=False, indent=2)+'\n')
    return buffer.getvalue()


@dataclass(frozen=True)
class GateConfig:
    stage: str = 'Raw'
    metric: str = 'Pearson'
    threshold: float = .98
    min_overlap: float = .9
    min_points: int = 30


def validate(cfg):
    if cfg.stage not in ('Raw', 'AsLS', 'Smoothing'):
        raise ValueError('Blank 비교 단계는 Raw, Baseline, SG 중에서 선택하세요.')
    if cfg.metric not in ('Pearson', 'Cosine') or not 0 <= cfg.threshold <= 1:
        raise ValueError('유사도 기준은 0–1이어야 합니다.')
    if not 0 < cfg.min_overlap <= 1 or cfg.min_points < 3:
        raise ValueError('파수 범위 겹침 / 최소 point 설정이 유효하지 않습니다.')


def stage_arrays(session, stage):
    if stage == 'Raw':
        return [item['raw'] for item in session.items]
    result = session.asls_result if stage == 'AsLS' else session.smoothing_result
    if result is None or (stage == 'AsLS' and not result['params'].enabled):
        raise ValueError(f'{stage} 결과가 없습니다. 전처리를 먼저 실행하세요.')
    return result['arrays']


def prepare_reference(reference, session, stage):
    """Use the sample's applied settings, never the unsaved widget values."""
    item = reference['item']
    rows = item['raw']
    if stage != 'Raw':
        rows = np.asarray([core.correct_one(row, session.asls_result['params']) for row in rows])
    if stage == 'Smoothing':
        rows = np.asarray([core.smooth_spectrum(row, session.smoothing_result['params']) for row in rows])
    return dict(label=reference['label'], key=reference['key'], axis=item['axis'], rows=rows,
                sha256=reference['sha256'], filename=reference['filename'])


def comparison(block, axis, reference, ref_axis, cfg):
    """Interpolate only inside measured overlap; score all sample rows in chunks."""
    axis = np.asarray(axis, dtype=float)
    ref_axis = np.asarray(ref_axis, dtype=float)
    if len(axis) < 3 or len(ref_axis) < 3 or np.any(np.diff(axis) <= 0) or np.any(np.diff(ref_axis) <= 0):
        raise ValueError('비교 파수축은 중복 없이 오름차순이어야 합니다.')
    lo, hi = max(axis[0], ref_axis[0]), min(axis[-1], ref_axis[-1])
    mask = (axis >= lo) & (axis <= hi)
    overlap = max(0., hi-lo)/(axis[-1]-axis[0])
    if overlap < cfg.min_overlap or mask.sum() < cfg.min_points:
        return None
    x = axis[mask]
    ref = np.interp(x, ref_axis, reference)
    normalized_ref = ref-ref.mean() if cfg.metric == 'Pearson' else ref.copy()
    norm = np.linalg.norm(normalized_ref)
    if not np.isfinite(normalized_ref).all() or norm <= np.finfo(float).eps*max(1., np.linalg.norm(ref)):
        return None
    normalized_ref /= norm
    scores = np.full(len(block), np.nan)
    for start in range(0, len(block), 512):
        values = np.array(block[start:start+512, mask], dtype=float, copy=True)
        finite = np.isfinite(values).all(axis=1)
        original_norms = np.linalg.norm(values, axis=1)
        if cfg.metric == 'Pearson':
            values -= values.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(values, axis=1)
        good = finite & (norms > np.finfo(float).eps*np.maximum(1., original_norms))
        out = np.full(len(values), np.nan)
        out[good] = np.clip((values[good] @ normalized_ref)/norms[good], -1., 1.)
        scores[start:start+len(values)] = out
    return dict(scores=scores, axis=x, reference=ref, mask=mask, overlap=overlap)


def scan_key(session, references, cfg):
    payload = dict(config=asdict(cfg), references=[(r['key'], r['sha256'], r['label']) for r in references],
                   core_version=core.VERSION,
                   baseline=session.asls_result['computed_at'],
                   smoothing=session.smoothing_result['asls_computed_at'] if session.smoothing_result else None,
                   smoothing_params=asdict(session.smoothing_result['params']) if session.smoothing_result else None,
                   files=[(str(item['path']), item['raw'].shape) for item in session.items])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def screen(session, references, cfg):
    validate(cfg)
    if not references:
        raise ValueError('활성 blank reference를 선택하세요.')
    arrays = stage_arrays(session, cfg.stage)
    prepared = [prepare_reference(ref, session, cfg.stage) for ref in references]
    records = []
    for fi, (item, block) in enumerate(zip(session.items, arrays)):
        best = np.full(len(block), -np.inf)
        matches = [None]*len(block)
        for ri, ref in enumerate(prepared):
            for pi_ref, row in enumerate(ref['rows']):
                result = comparison(block, item['axis'], row, ref['axis'], cfg)
                if result is None:
                    continue
                scores = result['scores']
                better = np.isfinite(scores) & (scores > best)
                for pi in np.flatnonzero(better):
                    matches[pi] = (ri, pi_ref, result['overlap'])
                best[better] = scores[better]
        for pi, score in enumerate(best):
            match = matches[pi]
            coords = item['coords'][pi] if item['coords'] is not None else (None, None)
            records.append(dict(fi=fi, pi=pi, file=item['path'].name,
                spectrum=core.spectrum_name(item, pi), x=None if coords[0] is None else float(coords[0]),
                y=None if coords[1] is None else float(coords[1]),
                reference=prepared[match[0]]['label'] if match else '',
                reference_index=match[0] if match else None,
                reference_spectrum=match[1] if match else None,
                overlap=match[2] if match else None, score=float(score) if match else None,
                candidate=bool(match is not None and score >= cfg.threshold)))
    return dict(signature=scan_key(session, references, cfg), config=asdict(cfg), records=records,
                references=prepared)


def gate_state(session):
    return getattr(session, 'blank_gate', {})


def current_result(session):
    state = gate_state(session)
    result = state.get('result')
    return result if result and result['signature'] == state.get('scan_key') else None


def reviewed_blanks(session):
    state = gate_state(session)
    return (set(state.get('confirmed', set()))-set(state.get('kept', set()))) if current_result(session) else set()


def reviewed_kept(session):
    return set(gate_state(session).get('kept', set())) if current_result(session) else set()


def confirmed(session):
    return reviewed_blanks(session) if gate_state(session).get('enabled') else set()


def state_signature(session):
    state = gate_state(session)
    payload = dict(enabled=bool(state.get('enabled')), scan=state.get('scan_key') if state.get('enabled') else None,
                   ready=current_result(session) is not None if state.get('enabled') else False,
                   confirmed=sorted(confirmed(session)),
                   kept=sorted(reviewed_kept(session)) if state.get('enabled') else [])
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def report(session):
    result = current_result(session)
    chosen = reviewed_blanks(session)
    kept = reviewed_kept(session)
    return [dict(file=row['file'], spectrum=row['spectrum'], spectrum_index_1based=row['pi']+1,
                 x=row['x'], y=row['y'], closest_reference=row['reference'], similarity=row['score'],
                 reference_sha256=(result['references'][row['reference_index']]['sha256']
                                   if row['reference_index'] is not None else ''),
                 comparison_stage=result['config']['stage'], metric=result['config']['metric'],
                 threshold=result['config']['threshold'], gate_enabled=bool(gate_state(session).get('enabled')),
                 overlap=row['overlap'], candidate=row['candidate'], confirmed_blank=(row['fi'],row['pi']) in chosen,
                 manually_kept=(row['fi'],row['pi']) in kept)
            for row in result['records']] if result else []


def export_session(session, policy='exclude_confirmed'):
    """Immutable membership snapshot. 'all' restores blanks AND manual exclusions."""
    if policy not in ('all', 'exclude_confirmed'):
        raise ValueError('유효하지 않은 blank 출력 범위입니다.')
    state = gate_state(session)
    result = current_result(session)
    if policy == 'exclude_confirmed' and state.get('enabled') and result is None:
        raise ValueError('Blank-gate 설정/입력/reference가 변경되었습니다. 후보 검사를 다시 실행하세요.')
    output = copy.copy(session)
    output.items = list(session.items)
    chosen = confirmed(session)
    output.excluded_pixels = (set(session.excluded_pixels) | chosen) if policy == 'exclude_confirmed' else set()
    output.blank_gate_info = dict(enabled=bool(state.get('enabled')), policy=policy,
        screened=result is not None,
        state_signature=state_signature(session), confirmed_count=len(reviewed_blanks(session)),
        applied_blank_count=len(chosen) if policy == 'exclude_confirmed' else 0,
        removed_count=len(output.excluded_pixels), manual_exclusions=[list(v) for v in sorted(session.excluded_pixels)],
        config=result['config'] if result else None,
        references=[dict(label=r['label'], sha256=r['sha256'], filename=r['filename'])
                    for r in result['references']] if result else [],
        rule='Only manually confirmed blanks are removed; similarity alone never removes a spectrum.')
    output.blank_gate_report = report(session)
    return output
