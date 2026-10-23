"""Shared blank review before CSV export and SVM/kNN/NNLS handoff."""
from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from . import blank_gate_core as gate
from . import preprocess_core as core

INPUT_PATCH = 'GATE-2.7.0'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def reference_files():
    folder = Path(__file__).resolve().parents[1]/'blank_references'
    result = []
    with st.expander('탑재 reference 확인 · 교체 / Si wafer 추가'):
        for key, label, filename in gate.REFERENCE_SLOTS:
            upload = st.file_uploader(label, type=['txt', 'csv', 'dat', 'tsv'], key=f'bg_upload_{key}')
            disk = folder/filename
            payload = upload.getvalue() if upload is not None else disk.read_bytes() if disk.is_file() else None
            if payload is None:
                st.caption(f'{label}: 미등록 · 판정에 사용하지 않음')
                continue
            digest = hashlib.sha256(payload).hexdigest()
            cache = st.session_state.setdefault('bg_reference_cache', {})
            cache_key = (key, digest)
            try:
                if cache_key not in cache:
                    temp = Path(st.session_state.workdir)/'blank_references'
                    temp.mkdir(exist_ok=True)
                    path = temp/filename
                    path.write_bytes(payload)
                    cache[cache_key] = core.read_auto(path, core.InputSettings(), None)
                item = cache[cache_key]
                result.append(dict(key=key, label=label, filename=filename, payload=payload,
                                   sha256=digest, item=item))
                source = upload.name if upload is not None else filename
                st.caption(f'{label}: {source} · {len(item["raw"]):,} spectra · '
                           f'{item["axis"][0]:.2f}–{item["axis"][-1]:.2f} cm⁻¹ · SHA256 {digest[:12]}')
            except Exception as exc:
                st.warning(f'{label} reference 읽기 실패: {exc}')
        if result:
            saved = [(ref['filename'], ref['payload']) for ref in result]
            st.download_button('Reference 탑재용 ZIP 다운로드', data=lambda: reference_zip(saved),
                file_name='RSLab_blank_references.zip', mime='application/zip',
                on_click='ignore', key='bg_reference_download')
        st.caption('업로드한 교체 파일은 현재 세션에서 사용합니다. 영구 탑재하려면 ZIP을 풀어 '
                   'GitHub 저장소 최상위 blank_references 폴더에 넣으세요. 서버 재시작 후에도 불러옵니다.')
    return result


def reference_zip(saved):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
        for filename, payload in saved:
            archive.writestr('blank_references/'+filename, payload)
    return stream.getvalue()


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
    axes[1].set_title(ref['label'], fontsize=9)
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
    names = {ref['key']: ref['label'] for ref in references}
    selected = st.multiselect('비교할 reference', list(names), default=list(names),
        format_func=lambda key: names[key], key='bg_references')
    references = [ref for ref in references if ref['key'] in selected]
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
               '탑재한 AgNS reference는 660 nm 측정 자료입니다. 다른 레이저·측정 조건에는 맞는 reference가 필요합니다.')
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
