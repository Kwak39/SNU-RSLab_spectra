"""Shared blank review before CSV export and SVM/kNN/NNLS handoff."""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import blank_gate_core as gate
from . import preprocess_core as core

INPUT_PATCH = 'GATE-2.7.1'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def reference_files():
    folder = Path(__file__).resolve().parents[1]/'blank_references'
    with st.expander('Reference 폴더 · Label / 사용 선택', expanded=True):
        st.caption('blank_references 폴더의 TXT / CSV / DAT / TSV를 자동으로 찾습니다. '
                   '파일명과 별개로 Label을 지정하고 사용할 항목을 체크하세요. 같은 Label을 여러 파일에 지정해도 됩니다.')
        uploads = st.file_uploader('이번 세션에 reference 추가 (선택)',
            type=['txt', 'csv', 'dat', 'tsv'], accept_multiple_files=True, key='bg_catalog_uploads')
        try:
            records = gate.discover_references(folder)
        except ValueError as exc:
            st.warning(str(exc))
            return []
        try:
            records = gate.add_uploaded_references(records, [(file.name, file.getvalue()) for file in uploads or []])
        except ValueError as exc:
            st.warning(str(exc))
            return []
        if not records:
            st.info('폴더에 reference 파일을 넣거나 이번 세션에 업로드하세요. 파일명은 자유롭게 정할 수 있습니다.')
            return []
        edits = st.session_state.setdefault('bg_catalog_edits', {})
        loaded = []
        for record in records:
            digest = hashlib.sha256(record['payload']).hexdigest()
            cached_key = (record['key'], digest)
            cache = st.session_state.setdefault('bg_reference_cache', {})
            try:
                if cached_key not in cache:
                    temp = Path(st.session_state.workdir)/'blank_references'
                    temp.mkdir(exist_ok=True)
                    path = temp/(digest+Path(record['filename']).suffix)
                    path.write_bytes(record['payload'])
                    cache[cached_key] = core.read_auto(path, core.InputSettings(), None)
                record['item'], record['sha256'] = cache[cached_key], digest
                entry = edits.setdefault(record['key'], dict(label=record['label'], enabled=record['enabled']))
                record.update(label=entry['label'], enabled=entry['enabled'])
                loaded.append(record)
            except Exception as exc:
                st.warning(f'{record["filename"]} 읽기 실패 · 판정에서 제외: {exc}')
        if not loaded:
            return []
        table = pd.DataFrame([dict(사용=record['enabled'], 파일=record['filename'], 출처=record['source'],
            Label=record['label'], Spectra=len(record['item']['raw']),
            파수범위=f'{record["item"]["axis"][0]:.2f}–{record["item"]["axis"][-1]:.2f}') for record in loaded])
        identity = hashlib.sha256(str([(record['key'], record['sha256']) for record in loaded]).encode()).hexdigest()[:12]
        revision = st.session_state.get('bg_catalog_revision', 0)
        edited = st.data_editor(table, hide_index=True, width='stretch',
            disabled=['파일', '출처', 'Spectra', '파수범위'], key=f'bg_catalog_editor_{identity}_{revision}')
        if st.button('Label · 사용 선택 적용', key='bg_catalog_apply'):
            changes = edited.fillna('').to_dict('records')
            if any(bool(row['사용']) and not str(row['Label']).strip() for row in changes):
                st.warning('사용할 reference에는 Label을 입력하세요. 파일명은 Label로 자동 사용하지 않습니다.')
            else:
                for record, row in zip(loaded, changes):
                    edits[record['key']] = dict(label=str(row['Label']).strip(), enabled=bool(row['사용']))
                st.session_state.bg_catalog_revision = revision+1
                st.session_state.bg_catalog_notice = 'Label과 사용 선택을 적용했습니다. Gate ON 상태이면 후보 검사를 다시 실행하세요.'
                st.rerun()
        if st.session_state.get('bg_catalog_notice'):
            st.success(st.session_state.pop('bg_catalog_notice'))
        selected = [record for record in loaded if record['enabled'] and record['label']]
        invalid_labels = any(record['enabled'] and not record['label'] for record in loaded)
        pending = any(str(row['Label']).strip() != record['label'] or bool(row['사용']) != record['enabled']
                      for record, row in zip(loaded, edited.fillna('').to_dict('records')))
        if pending:
            st.caption('수정한 Label · 사용 선택은 적용 버튼을 누른 뒤 검사하거나 저장하세요.')
        st.caption(f'등록 {len(loaded)}개 · 사용 {len(selected)}개 · '
                   f'Label 미지정 {sum(not record["label"] for record in loaded)}개')
        st.download_button('Reference + Label 설정 ZIP 저장', data=lambda records=[dict(r) for r in loaded]: gate.reference_bundle(records),
            file_name='RSLab_blank_references_labelled.zip', mime='application/zip',
            disabled=invalid_labels or pending, on_click='ignore', key='bg_reference_download')
        st.caption('적용은 현재 세션에 반영됩니다. 영구 저장하려면 ZIP을 풀어 GitHub 최상위 '
                   'blank_references 폴더에 spectrum 파일과 labels.json을 함께 올리세요. '
                   '다음 세션에서 Label과 사용 선택을 다시 불러옵니다.')
    return selected


def report_csv(records):
    buffer = io.StringIO(newline='')
    if records:
        writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    return buffer.getvalue().encode('utf-8-sig')


def comparison_figure(session, result, row):
    cfg = gate.GateConfig(**result['config'])
    fi, pi = row['fi'], row['pi']
    item = session.items[fi]
    values = gate.stage_arrays(session, cfg.stage)[fi][pi]
    ref = result['references'][row['reference_index']]
    ref_values = ref['rows'][row['reference_spectrum']]
    comparison = gate.comparison(values[None, :], item['axis'], ref_values, ref['axis'], cfg)
    figure, axes = plt.subplots(1, 3, figsize=(12.3, 2.9), dpi=300)
    axes[0].plot(item['axis'], values, lw=.8)
    axes[0].set_title(f'Input · {cfg.stage}', fontsize=9)
    axes[1].plot(ref['axis'], ref_values, lw=.8, color='#e38a1d')
    axes[1].set_title('Blank reference', fontsize=9)
    left, right = values[comparison['mask']].copy(), comparison['reference'].copy()
    if cfg.metric == 'Pearson':
        left -= left.mean(); right -= right.mean()
    left /= np.linalg.norm(left); right /= np.linalg.norm(right)
    axes[2].plot(comparison['axis'], left, lw=.9, label='Input')
    axes[2].plot(comparison['axis'], right, lw=.9, alpha=.8, label='Reference')
    axes[2].set_title(f'{cfg.metric} = {row["score"]:.5f}', fontsize=9)
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.set(xlabel='Raman shift (cm$^{-1}$)', ylabel='Intensity' if axis is not axes[2] else 'Normalized shape')
        axis.tick_params(labelsize=8)
    figure.suptitle(session.label((fi, pi)), fontsize=9)
    figure.tight_layout()
    return figure


def render(session):
    state = getattr(session, 'blank_gate', None)
    if state is None:
        session.blank_gate = state = dict(enabled=False, result=None, confirmed=set())
    state.setdefault('kept', set())
    previous_enabled = state['enabled']
    state['enabled'] = st.checkbox('Blank-gate 사용', False, key='bg_enabled')
    if state['enabled'] != previous_enabled:
        st.rerun()
    references = reference_files()
    options = ['Raw']
    if session.asls_result['params'].enabled:
        options.append('AsLS')
    if session.smoothing_result is not None:
        options.append('Smoothing')
    a, b, c, d = st.columns([1.2, 1, 1, 1])
    stage = a.selectbox('비교 단계', options, key='bg_stage',
                        format_func=lambda value: 'Baseline' if value == 'AsLS' else value)
    metric = b.selectbox('유사도', ['Pearson', 'Cosine'], key='bg_metric')
    threshold = c.number_input('후보 기준 (0–1)', min_value=0., max_value=1., value=.98,
                               step=.005, format='%.3f', key='bg_threshold')
    overlap = d.number_input('최소 범위 겹침', min_value=.1, max_value=1., value=.9,
                              step=.05, key='bg_overlap')
    cfg = gate.GateConfig(stage=stage, metric=metric, threshold=float(threshold), min_overlap=float(overlap))
    state['scan_key'] = gate.scan_key(session, references, cfg)
    st.caption('Raw는 원본 신호 모양을 비교합니다. Baseline/SG는 reference에도 동일한 적용 설정을 계산합니다. '
               '공통 측정 범위에서만 보간하며 외삽하지 않습니다. 최소 30 points가 필요합니다.')
    st.caption('Pearson은 평균값을 뺀 모양, Cosine은 평균값을 유지한 모양을 비교합니다. '
               '0.98은 시작값이며 검증된 blank 판정 기준이 아닙니다. '
               '레이저·용매·기판 등 측정 조건이 맞는 reference를 선택하세요.')
    if not state['enabled']:
        st.info('Blank-gate OFF: blank 판정에 따른 제외는 적용하지 않습니다. 수동 지점 제외는 별도입니다.')
        return
    if st.button('Blank 후보 검사', disabled=not references, key='bg_scan', type='primary'):
        try:
            with st.spinner('각 spectrum / mapping point를 reference와 비교 중…'):
                result = gate.screen(session, references, cfg)
            if not state.get('result') or state['result']['signature'] != result['signature']:
                state['confirmed'] = set()
                state['kept'] = set()
            state['result'] = result
            st.session_state.bg_notice = '후보 검사 완료. 아직 blank 확정 전이며 자동 제외하지 않습니다.'
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get('bg_notice'):
        st.success(st.session_state.pop('bg_notice'))
    result = gate.current_result(session)
    if result is None:
        st.info('Reference / 설정 / 전처리 결과가 바뀌면 검사를 다시 실행하세요. 이전 판정은 적용되지 않습니다.')
        return
    candidates = {(row['fi'], row['pi']) for row in result['records'] if row['candidate']}
    chosen = state['confirmed']
    kept = state['kept']
    unscorable = sum(row['score'] is None for row in result['records'])
    st.caption(f'전체 {len(result["records"]):,} · 후보 {len(candidates):,} · 확정 blank {len(chosen):,} · '
               f'유사도 계산 불가 {unscorable:,} (짧은 겹침·평탄한 신호 등)')
    st.caption('후보는 자동 제외하지 않습니다. 비교 후 선택 확정하거나, 후보 전체를 확정하고 예외를 복원하세요. '
               '정상으로 복원한 항목은 이후 후보 일괄 확정에서도 보호합니다.')
    a, b = st.columns([1, 2])
    view = a.selectbox('검토 범위', ['후보', '전체', '확정 blank', '계산 불가'], key='bg_review_view')
    query = b.text_input('파일 / reference 검색', '', key='bg_query').casefold().strip()
    records = [row for row in result['records'] if
               (view == '전체' or view == '후보' and row['candidate'] or
                view == '확정 blank' and (row['fi'], row['pi']) in chosen or
                view == '계산 불가' and row['score'] is None) and
               (not query or query in (row['file']+' '+row['reference']).casefold())]
    total_pages = max(1, (len(records)+199)//200)
    page_key = 'bg_page_'+hashlib.sha256((result['signature']+view+query).encode()).hexdigest()[:12]
    page = int(st.number_input('검토 페이지 (200개씩)', min_value=1, max_value=total_pages, value=1,
                               step=1, key=page_key))
    visible = records[(page-1)*200:page*200]
    table = [dict(파일=r['file'], Spectrum=r['spectrum'], x=r['x'], y=r['y'], Reference=r['reference'],
                  유사도=r['score'], 후보=r['candidate'], 확정_blank=(r['fi'],r['pi']) in chosen,
                  정상_확정=(r['fi'],r['pi']) in kept) for r in visible]
    event = st.dataframe(table, hide_index=True, width='stretch', height=260,
        on_select='rerun', selection_mode='multi-row', key='bg_review_table')
    selected_rows = event.get('selection', {}).get('rows', []) if event else []
    selected_keys = {(visible[i]['fi'], visible[i]['pi']) for i in selected_rows if 0 <= i < len(visible)}
    a, b, c, d = st.columns(4)
    if a.button('선택 → blank 확정', disabled=not selected_keys, key='bg_confirm_selected'):
        chosen.update(selected_keys); kept.difference_update(selected_keys); st.rerun()
    if b.button('선택 → 정상으로 복원', disabled=not selected_keys, key='bg_restore_selected'):
        chosen.difference_update(selected_keys); kept.update(selected_keys); st.rerun()
    if c.button('후보 전체 blank 확정', disabled=not candidates, key='bg_confirm_candidates'):
        chosen.update(candidates-kept); st.rerun()
    if d.button('확정 blank 전체 복원', disabled=not chosen, key='bg_restore_all'):
        kept.update(chosen); chosen.clear(); st.rerun()
    if visible:
        index = st.selectbox('비교할 spectrum', range(len(visible)), key='bg_compare_row',
            format_func=lambda i: f'{visible[i]["file"]} · {visible[i]["spectrum"]}')
        row = visible[index]
        if row['reference_index'] is None:
            st.info('비교 가능한 reference가 없습니다. 자동으로 blank 처리하지 않습니다.')
        else:
            st.caption(f'비교 reference Label: {row["reference"]} · '
                       f'파일: {result["references"][row["reference_index"]]["filename"]}')
            figure = comparison_figure(session, result, row)
            st.pyplot(figure, dpi=300); plt.close(figure)
        a, b = st.columns(2)
        current = (row['fi'],row['pi'])
        if a.button('현재 spectrum → blank 확정', disabled=current in chosen, key='bg_confirm_current'):
            chosen.add(current); kept.discard(current); st.rerun()
        if b.button('현재 spectrum → 정상 확정', disabled=current in kept, key='bg_restore_current'):
            chosen.discard(current); kept.add(current); st.rerun()
    else:
        st.info('이 검토 범위에 해당하는 spectrum이 없습니다. 전체 보기로 다른 spectrum도 확인할 수 있습니다.')
    records_report = gate.report(session)
    st.download_button('Blank 판정 기록 CSV', data=report_csv(records_report),
        file_name='RSLab_blank_gate_review.csv', mime='text/csv', on_click='ignore', key='bg_review_download')
    if st.button('검토 결정 초기화', disabled=not (chosen or kept), key='bg_reset_review'):
        chosen.clear(); kept.clear(); st.rerun()
