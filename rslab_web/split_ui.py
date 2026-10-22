"""One selection table, one label field, and explicit TRAIN/TEST destinations."""
from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter

import pandas as pd
import streamlit as st

from .common import csv_bytes
from .workflow import (apply_selection, assignment_packets, assignment_rows,
                       clear_results, import_assignments, new_assignments)


def _selection_rows(event):
    if isinstance(event, dict):
        return event.get('selection', {}).get('rows', [])
    return list(event.selection.rows) if event is not None else []


def _csv_settings(tool, pool, state, train_name):
    with st.expander('Labeling 설정 저장 · 불러오기'):
        left, right = st.columns(2)
        with left:
            st.download_button('Labeling CSV 저장', csv_bytes(['sample_id', 'set', 'label'],
                zip(pool.data.sample_ids,
                    [train_name if r == 'TRAIN' else r for r in state['roles']], state['labels'])),
                file_name=f'{tool}_labeling.csv', mime='text/csv', key=f'{tool}_labeling_download',
                on_click='ignore')
            if tool != 'nnls':
                other = 'knn' if tool == 'svm' else 'svm'
                other_state = st.session_state.get(f'{other}_pool_assignments')
                available = other_state is not None and other_state['pool_signature'] == pool.signature
                if st.button(f'{"k-NN" if other == "knn" else "SVM"}의 Labeling 복사',
                             disabled=not available, key=f'{tool}_copy_labeling'):
                    for field in ('roles', 'labels'):
                        state[field] = list(other_state[field])
                    state['revision'] += 1
                    clear_results(st.session_state, tool)
                    st.rerun()
        with right:
            uploaded = st.file_uploader('Labeling CSV', type=['csv'], key=f'{tool}_labeling_upload')
            if st.button('Labeling 불러오기', disabled=uploaded is None, key=f'{tool}_labeling_import'):
                try:
                    payload = uploaded.getvalue()
                    try:
                        text = payload.decode('utf-8-sig')
                    except UnicodeDecodeError:
                        text = payload.decode('cp949')
                    # Older files may contain group; user-edited grouping is no longer used.
                    rows = [{k: v for k, v in row.items() if k in ('sample_id', 'set', 'label')}
                            for row in csv.DictReader(io.StringIO(text))]
                    import_assignments(pool, state, rows)
                    clear_results(st.session_state, tool)
                    st.rerun()
                except (ValueError, UnicodeError, csv.Error, AttributeError) as exc:
                    st.error(str(exc))
        st.caption('sample_id = spectrum 식별자 · set = TRAIN/TEST · label = 물질명. 설정만 저장하며 스펙트럼 값은 포함하지 않습니다.')


def render(tool, pool):
    train_name = 'REFERENCE' if tool == 'nnls' else 'TRAIN'
    state_key = f'{tool}_pool_assignments'
    if state_key not in st.session_state or st.session_state[state_key]['pool_signature'] != pool.signature:
        st.session_state[state_key] = new_assignments(pool)
        clear_results(st.session_state, tool)
    state = st.session_state[state_key]
    st.markdown('**1 · 데이터 선택 → Label 입력 → Set 배정**')
    st.caption('Label은 물질명입니다 (예: alizarin). 빈칸이면 기존 Label을 유지합니다. TEST 정답 Label은 선택 사항입니다.')
    a, b, c, d = st.columns([1, 1, 2, .9])
    unit = a.selectbox('선택 단위', ['파일', 'Spectrum'], key=f'{tool}_assign_unit')
    display_roles = ['전체', '미지정', train_name, 'TEST', '제외']
    view_role = b.selectbox('Set 보기', display_roles, key=f'{tool}_assign_filter')
    role = 'TRAIN' if view_role == 'REFERENCE' else view_role
    query = c.text_input('파일명 · Label 검색', '', key=f'{tool}_assign_query')
    rows = assignment_rows(pool, state, unit, role, query)
    token = hashlib.sha256(repr((pool.signature, unit, role, query, state['revision'])).encode()).hexdigest()[:16]
    pages = max(1, (len(rows) + 499) // 500)
    page = int(d.number_input('페이지', min_value=1, max_value=pages, value=1,
                              step=1, key=f'{tool}_assign_page_{token}'))
    visible = rows[(page-1)*500:page*500]
    columns = ['file', 'spectra', 'set', 'label'] if unit == '파일' else ['file', 'spectrum', 'set', 'label']
    frame = pd.DataFrame([{name: row[name] for name in columns} for row in visible], columns=columns)
    frame['set'] = frame['set'].replace({'TRAIN': train_name})
    event = st.dataframe(frame, hide_index=True, width='stretch', height=min(285, 38+35*max(1, len(frame))),
        column_config={'file': '파일', 'spectra': 'Spectrum 수', 'spectrum': 'Spectrum',
                       'set': '현재 Set', 'label': '현재 Label'},
        selection_mode='multi-row', on_select='rerun', key=f'{tool}_assign_table_{token}_{page}')
    st.caption(f'검색 결과 {len(rows):,} {unit} · {page}/{pages} 페이지 (최대 500행). Ctrl/Shift로 다중선택할 수 있습니다. '
               + ('파일 선택은 그 파일의 모든 spectrum에 적용됩니다.' if unit == '파일' else ''))
    a, b = st.columns([1, 2])
    scope = a.radio('적용 범위', ['선택한 행', '검색 결과 전체'], horizontal=True,
                     key=f'{tool}_assign_scope')
    label = b.text_input('선택 데이터의 Label', '', placeholder='물질명 · 빈칸이면 유지',
                          key=f'{tool}_assign_label')
    selected = _selection_rows(event)
    chosen_rows = rows if scope == '검색 결과 전체' else [visible[i] for i in selected if 0 <= i < len(visible)]
    indices = [i for row in chosen_rows for i in row['_indices']]
    st.caption(f'적용 대상: {len(chosen_rows):,} {unit} / {len(indices):,} spectra'
               + (' · 모든 페이지 포함' if scope == '검색 결과 전체' else ''))
    a, b, c = st.columns(3)
    for column, name, destination in [(a, f'{train_name}에 넣기', 'TRAIN'),
                                      (b, 'TEST에 넣기', 'TEST'), (c, '배정 해제', '미지정')]:
        if column.button(name, key=f'{tool}_assign_{destination}', disabled=not indices, width='stretch'):
            try:
                count = apply_selection(pool, state, indices, destination, label)
                clear_results(st.session_state, tool)
                st.session_state[f'{tool}_assign_notice'] = f'{count:,} spectra → {train_name if destination == "TRAIN" else destination}'
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    if st.session_state.get(f'{tool}_assign_notice'):
        st.success(st.session_state.pop(f'{tool}_assign_notice'))
    with st.expander('개별 Label · Set 수정'):
        if unit != 'Spectrum':
            st.caption('선택 단위를 Spectrum으로 바꾸면 현재 페이지에서 개별 Label과 Set을 수정할 수 있습니다.')
        elif visible:
            details = pd.DataFrame([dict(sample_id=pool.data.sample_ids[row['_indices'][0]],
                set=train_name if row['set'] == 'TRAIN' else row['set'], label=row['label']) for row in visible])
            edited = st.data_editor(details, hide_index=True, width='stretch', height=230,
                disabled=['sample_id'], column_config={
                    'set': st.column_config.SelectboxColumn('Set', options=display_roles[1:], required=True),
                    'label': 'Label'}, key=f'{tool}_detail_{token}_{page}')
            if st.button('개별 수정 저장', key=f'{tool}_detail_apply'):
                try:
                    import_assignments(pool, state, edited.fillna('').to_dict('records'))
                    clear_results(st.session_state, tool)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
    _csv_settings(tool, pool, state, train_name)

    # The committed result is deliberately placed below all editing controls.
    counts = Counter(state['roles'])
    st.markdown('**2 · 배정 현황 확인**')
    st.caption(f'{train_name} {counts["TRAIN"]:,} · TEST {counts["TEST"]:,} · '
               f'미지정 {counts["미지정"]:,} · 제외 {counts["제외"]:,} spectra')
    summaries = Counter((label or '(Label 없음)', role) for label, role in zip(state['labels'], state['roles'])
                        if role in ('TRAIN', 'TEST'))
    if summaries:
        labels = sorted({label for label, _ in summaries})
        summary = pd.DataFrame([{'Label': label, train_name: summaries[(label, 'TRAIN')],
                                'TEST': summaries[(label, 'TEST')]} for label in labels])
        st.dataframe(summary, hide_index=True, width='stretch', height=min(210, 38+35*len(summary)))
    train_packet, train_labels, test_packet, test_labels = assignment_packets(pool, state)
    missing = sum(not label for label in train_labels['labels'])
    if missing and tool != 'nnls':
        st.warning(f'TRAIN {missing:,}개에 Label이 없습니다. 위에서 물질명을 입력해 배정하세요.')
    shared_files = ({pool.data.sources[i] for i, r in enumerate(state['roles']) if r == 'TRAIN'} &
                    {pool.data.sources[i] for i, r in enumerate(state['roles']) if r == 'TEST'})
    if shared_files:
        st.caption(f'같은 파일에서 {train_name}/TEST로 나눈 파일 {len(shared_files)}개. 독립 시편 평가에는 서로 다른 시편을 사용하세요.')
    return train_packet, train_labels, test_packet, test_labels
