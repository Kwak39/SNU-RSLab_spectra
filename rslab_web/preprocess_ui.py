"""Compact input workflow: baseline -> native BLaPE -> final common axis."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import io
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from . import preprocess_core as core
from . import workflow
from .common import file_pairs
from .workflow import UploadStore, clear_ml, clear_results, make_pool

INPUT_PATCH = 'PIPE-2.6.1'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def compatible_input_files():
    return (getattr(core, 'INPUT_PATCH', None) == INPUT_PATCH
            and getattr(workflow, 'INPUT_PATCH', None) == INPUT_PATCH)


def load_core():
    return core, Path(core.__file__)


def init_state():
    if 'workdir' not in st.session_state:
        st.session_state.pre_temp = tempfile.TemporaryDirectory(prefix='rslab_streamlit_')
        st.session_state.workdir = Path(st.session_state.pre_temp.name)
    for key, value in [('session', None), ('processed', False), ('messages', []),
                       ('pre_upload_generation', 0), ('pre_point_generation', 0)]:
        st.session_state.setdefault(key, value)
    if 'pre_store' not in st.session_state:
        st.session_state.pre_store = UploadStore(Path(st.session_state.workdir) / 'inputs')
    if st.session_state.get('pre_engine_version') != core.VERSION:
        if st.session_state.session is not None or st.session_state.get('ml_pool') is not None:
            reset_session()
            if st.session_state.get('ml_pool') is not None:
                clear_ml(st.session_state)
            st.session_state.pre_notice = ('입력 엔진 업데이트: 이전 전처리 결과를 초기화했습니다. '
                '입력 파일 목록은 유지됩니다. 전처리를 다시 실행하고 ML로 보내세요.')
        st.session_state.pre_engine_version = core.VERSION


def reset_session(remove_files=False):
    st.session_state.session = None
    st.session_state.processed = False
    st.session_state.messages = []
    st.session_state.pre_point_generation = st.session_state.get('pre_point_generation', 0) + 1
    for key in ('export_zip', 'export_choice', 'pre_load_token'):
        st.session_state.pop(key, None)
    if remove_files:
        st.session_state.pre_store.clear()
        st.session_state.pre_upload_generation += 1


def export_snapshot(session):
    """Keep completed arrays alive if the page changes during a download."""
    snapshot = copy.copy(session)
    snapshot.items = list(session.items)
    snapshot.excluded_pixels = set(session.excluded_pixels)
    return snapshot


def make_export_zip(engine, session, include_asls, include_range=False,
                    include_blape=False, include_final=False, include_smoothing=False):
    """Pure download worker: no Streamlit/session-state access or shared folders."""
    stages = [name for name, selected in [('AsLS', include_asls), ('Smoothing', include_smoothing), ('Range', include_range),
              ('BLaPE', include_blape), ('Final', include_final)] if selected]
    if not stages:
        raise ValueError('내보낼 처리 단계를 선택하세요.')
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory(prefix='rslab_export_') as temp:
        root = Path(temp)
        for stage in stages:
            session.save(stage, root)
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    return buffer.getvalue()


def plot_current(engine, session, stage, file_index, spectrum_index):
    data = session.plot_data(stage, (file_index, spectrum_index))
    count = max(row[0] for row in data['series'])+1
    figure, axes = plt.subplots(1, count, figsize=(12.3, 2.9), dpi=300, squeeze=False)
    for row, x, y, label in data['series']:
        display_label = label.replace('Savitzky-Golay', 'SG').replace('Final (uniform grid)', 'Final')
        axes[0, row].plot(x, y, lw=.9, label=display_label)
    for i, axis in enumerate(axes[0]):
        for lo, hi in data.get('excluded_ranges', []):
            axis.axvspan(lo, hi, color='#d62728', alpha=.15)
        axis.set(xlabel='Raman shift (cm$^{-1}$)', ylabel='Intensity')
        axis.legend(fontsize=7, loc='upper right')
        if data.get('panel_titles'):
            axis.set_title(data['panel_titles'][i], fontsize=9)
    figure.suptitle(data['label'], fontsize=9)
    figure.tight_layout()
    return figure


def _apply_sg_preview(window, order):
    st.session_state.pre_sg_enabled = True
    st.session_state.pre_sg_window = int(window)
    st.session_state.pre_sg_order = int(order)
    st.session_state.pre_sg_apply_pending = True


def _smoothing_preview(session, settings):
    st.caption('현재 선택한 spectrum만 실시간 미리보기합니다. 적용 버튼은 전체 파일과 이후 활성 단계를 다시 계산합니다.')
    a, b, c, d = st.columns([2, 1, 1, 1])
    fi = a.selectbox('파일', range(len(session.items)), key='sg_preview_file',
                    format_func=lambda i: session.items[i]['path'].name)
    item = session.items[fi]
    pi = int(b.number_input('Spectrum 번호', min_value=1, max_value=len(item['raw']),
                            value=1, step=1, key=f'sg_preview_pixel_{fi}_{len(item["raw"])}'))-1
    window = int(c.number_input('Window (points)', min_value=3, value=settings['sg_window'],
                                 step=2, key='sg_preview_window'))
    order = int(d.number_input('Polynomial order (n)', min_value=0, value=settings['sg_order'],
                                step=1, key='sg_preview_order'))
    before = session.asls_result['arrays'][fi][pi]
    try:
        after = core.smooth_spectrum(before, core.SmoothingSettings(window, order))
    except ValueError as exc:
        st.warning(str(exc))
        return
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 2.9), dpi=300)
    axis = item['axis']
    axes[0].plot(axis, before, lw=.8, label='Before SG')
    axes[0].set_title('Before SG', fontsize=9)
    axes[1].plot(axis, after, lw=1, label=f'SG w={window}, n={order}')
    axes[1].set_title('SG preview', fontsize=9)
    for ax in axes:
        ax.set(xlabel='Raman shift (cm$^{-1}$)', ylabel='Intensity'); ax.legend(fontsize=7)
    figure.tight_layout()
    st.pyplot(figure, dpi=300); plt.close(figure)
    current = session.smoothing_result
    st.caption(f'미리보기: window={window}, n={order} · '
               + (f'전체 적용값: window={current["params"].window_length}, n={current["params"].polyorder}'
                  if current else '전체 Smoothing 단계는 아직 적용되지 않았습니다.'))
    if st.button('이 SG 설정 적용 · 전처리', key='sg_apply_preview',
                 on_click=lambda: _apply_sg_preview(window, order)):
        st.rerun()


def _picked_spectrum(event, count):
    """Read field-projected Vega point selections, never coordinate/row guessing."""
    picked = (event or {}).get('selection', {}).get('mapping_pick', {})
    if isinstance(picked, dict):
        values = picked.get('spectrum', [])
    elif isinstance(picked, list):
        values = [row.get('spectrum') for row in picked if isinstance(row, dict)]
    else:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) != 1 or isinstance(values[0], bool):
        return None
    try:
        number = int(values[0])
        return number if float(values[0]) == number and 1 <= number <= count else None
    except (ValueError, TypeError, OverflowError):
        return None


def _mapping_click(chart_key, number_key, count):
    # Streamlit runs callbacks before the number-input widget is instantiated.
    number = _picked_spectrum(st.session_state.get(chart_key), count)
    if number is not None:
        st.session_state[number_key] = number


def mapping_spec(session, fi, pi):
    item = session.items[fi]
    rows = [dict(spectrum=i+1, x=float(x), y=float(y),
                 status='제외' if (fi, i) in session.excluded_pixels else '저장 대상',
                 current=i == pi)
            for i, (x, y) in enumerate(item['coords'])]
    return dict(data=dict(values=rows), height=250,
        mark=dict(type='circle', cursor='pointer', opacity=1),
        params=[dict(name='mapping_pick', select=dict(type='point', fields=['spectrum'],
                     toggle=False, clear='dblclick'))],
        encoding=dict(
            x=dict(field='x', type='quantitative', title='x position', scale=dict(zero=False)),
            y=dict(field='y', type='quantitative', title='y position', scale=dict(zero=False, reverse=True)),
            color=dict(field='status', type='nominal', title=None,
                       scale=dict(domain=['저장 대상', '제외'], range=['#2878b5', '#d64242']),
                       legend=dict(orient='bottom')),
            size=dict(condition=dict(test='datum.current', value=220), value=90),
            stroke=dict(condition=dict(test='datum.current', value='#f0a000'), value='white'),
            strokeWidth=dict(condition=dict(test='datum.current', value=3), value=1),
            tooltip=[dict(field='spectrum', type='ordinal', title='Spectrum'),
                     dict(field='x', type='quantitative'), dict(field='y', type='quantitative'),
                     dict(field='status', type='nominal', title='상태')]))


def index_spec(session, fi, pi):
    """Numbered tiles follow column order; they carry no spatial coordinates."""
    start = (pi // 100) * 100
    stop = min(start+100, len(session.items[fi]['raw']))
    rows = [dict(spectrum=i+1, column=(i-start) % 10, row=(i-start) // 10,
                 status='제외' if (fi, i) in session.excluded_pixels else '저장 대상',
                 current=i == pi) for i in range(start, stop)]
    return dict(data=dict(values=rows), height=max(110, ((stop-start+9)//10)*28),
        params=[dict(name='mapping_pick', select=dict(type='point', fields=['spectrum'],
                     toggle=False, clear='dblclick'))],
        encoding=dict(
            x=dict(field='column', type='ordinal', axis=None, scale=dict(padding=.12)),
            y=dict(field='row', type='ordinal', axis=None, scale=dict(padding=.12)),
            tooltip=[dict(field='spectrum', type='ordinal', title='Spectrum'),
                     dict(field='status', type='nominal', title='상태')]),
        layer=[dict(mark=dict(type='rect', cursor='pointer'), encoding=dict(
            color=dict(field='status', type='nominal', title=None,
                scale=dict(domain=['저장 대상', '제외'], range=['#2878b5', '#d64242']),
                legend=dict(orient='bottom')),
            stroke=dict(condition=dict(test='datum.current', value='#f0a000'), value='white'),
            strokeWidth=dict(condition=dict(test='datum.current', value=3), value=1))),
            dict(mark=dict(type='text', cursor='pointer', color='white', fontSize=10),
                 encoding=dict(text=dict(field='spectrum', type='ordinal')))])


def _mapping_tab(session, stages):
    files = [fi for fi, item in enumerate(session.items) if item['kind'] in ('mapping', 'matrix')]
    if not files:
        st.info('좌표 mapping 또는 다중 spectrum 파일을 전처리하면 이 탭에서 지점을 클릭할 수 있습니다.')
        return
    a, b, c = st.columns([1, 2, 1])
    stage = a.selectbox('표시 단계', stages, index=len(stages)-1, key='map_view_stage',
        format_func=lambda v: session.asls_result['params'].method if v == 'AsLS' else v)
    fi = b.selectbox('Mapping / 다중 spectrum 파일', files, key='map_view_file',
                     format_func=lambda i: session.items[i]['path'].name)
    item = session.items[fi]
    number_key = f'map_view_pixel_{fi}_{len(item["raw"])}'
    pi = int(c.number_input('Spectrum 번호', min_value=1, max_value=len(item['raw']),
                            value=1, step=1, key=number_key))-1
    location, spectrum = st.columns([1, 3], gap='medium')
    with location:
        if item['coords'] is not None:
            x, y = item['coords'][pi]
            st.caption(f'실제 위치 x={x:g}, y={y:g} · Spectrum {pi+1}')
            spec = mapping_spec(session, fi, pi)
        else:
            first = (pi//100)*100+1
            st.caption(f'원본 열 순서의 번호표 · {first}–{min(first+99,len(item["raw"]))} / '
                       f'{len(item["raw"])} spectra')
            st.caption('번호를 클릭하세요. 공간 좌표는 파일에 없으며, 번호표는 실제 위치 지도가 아닙니다.')
            spec = index_spec(session, fi, pi)
        key = f'map_pick_{st.session_state.pre_point_generation}_{fi}_{pi}'
        st.vega_lite_chart(spec=spec, use_container_width=True, key=key,
            selection_mode='mapping_pick',
            on_select=lambda: _mapping_click(key, number_key, len(item['raw'])))
        st.caption('주황 테두리: 선택 · 파란색: 포함 · 빨간색: 제외')
    with spectrum:
        figure = plot_current(core, session, stage, fi, pi)
        st.pyplot(figure, dpi=300)
        plt.close(figure)
    _exclusion_controls(session, fi, pi, key_prefix='map')


def set_excluded(session, fi, pi, excluded, restore_file=False):
    """Change export membership only; raw/corrected arrays remain available."""
    if not 0 <= fi < len(session.items) or not 0 <= pi < len(session.items[fi]['raw']):
        raise ValueError('유효하지 않은 spectrum 번호입니다.')
    if restore_file:
        session.excluded_pixels.difference_update(
            {(fi, index) for index in range(len(session.items[fi]['raw']))})
    elif excluded:
        session.excluded_pixels.add((fi, pi))
    else:
        session.excluded_pixels.discard((fi, pi))


def _exclusion_controls(session, fi, pi, key_prefix='pre'):
    excluded = (fi, pi) in session.excluded_pixels
    removed = sum((fi, i) in session.excluded_pixels for i in range(len(session.items[fi]['raw'])))
    st.caption(f'현재 spectrum: {"제외" if excluded else "저장 대상"} · '
               f'이 파일 {len(session.items[fi]["raw"])-removed:,}개 포함 / {removed:,}개 제외')
    a, b, c = st.columns(3)
    if a.button('현재 지점 제외', disabled=excluded, key=f'{key_prefix}_point_exclude', width='stretch'):
        set_excluded(session, fi, pi, True)
        st.rerun()
    if b.button('현재 지점 복원', disabled=not excluded, key=f'{key_prefix}_point_restore', width='stretch'):
        set_excluded(session, fi, pi, False)
        st.rerun()
    if c.button('이 파일 전체 복원', disabled=not removed, key=f'{key_prefix}_file_restore', width='stretch'):
        set_excluded(session, fi, pi, False, restore_file=True)
        st.rerun()


def _input_files():
    store = st.session_state.pre_store
    st.markdown('**1 · 입력 파일**')
    uploaded = st.file_uploader('Single / Mapping / 다중 spectrum', type=['txt', 'csv', 'dat', 'tsv'],
        accept_multiple_files=True, key=f'pre_upload_{st.session_state.pre_upload_generation}',
        help='2열 single, 좌표가 있는 mapping, 첫 열이 파수축인 다중 spectrum을 함께 넣을 수 있습니다. '
             '여러 폴더에서 선택 → 목록 추가를 반복하세요. 상단 메타데이터는 줄 수를 지정하지 않고 '
             '내용으로 구분합니다. 인코딩·구분자·입력 형식을 자동 판별합니다.')
    if st.button('목록에 추가', disabled=not uploaded, key='pre_add_files', width='stretch'):
        try:
            added, duplicates = store.add(file_pairs(uploaded))
            if added:
                reset_session()
            st.session_state.pre_upload_generation += 1
            st.session_state.pre_notice = f'{len(added)}개 추가 · 중복 {duplicates}개 제외'
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.session_state.get('pre_notice'):
        st.success(st.session_state.pop('pre_notice'))
    st.caption(f'누적 {len(store.entries):,} files · {store.size / 1024**2:.1f} MiB')
    with st.expander('누적 파일 확인 · 삭제'):
        remove_ids = st.multiselect('파일 목록', list(store.entries),
            format_func=lambda uid: store.entries[uid]['name'], key=f'pre_remove_{store.revision}')
        left, right = st.columns(2)
        if left.button('선택 삭제', disabled=not remove_ids, key='pre_remove_selected'):
            store.remove(remove_ids)
            reset_session()
            st.rerun()
        if right.button('전체 삭제', key='pre_clear_inputs'):
            reset_session(remove_files=True)
            st.rerun()
        if st.button('결과만 비우기 · 입력 유지', key='pre_clear_results'):
            reset_session()
            st.rerun()
        st.caption('원본은 직접 삭제할 때까지 현재 세션에 유지됩니다.')


def _settings():
    with st.expander('입력 · 전처리 설정', expanded=True):
        files, baseline, blape = st.columns([1.05, 1, 1.15], gap='medium')
        with files:
            _input_files()
        with baseline:
            st.markdown('**2 · Baseline correction**')
            baseline_enabled = st.checkbox('Baseline correction 사용', True, key='pre_baseline_enabled')
            method = st.radio('방법', ['AsLS', 'arPLS'], horizontal=True, key='pre_baseline_method', disabled=not baseline_enabled)
            a, b, c = st.columns(3)
            lam = a.number_input('λ', min_value=1., value=1e5,
                format='%.5g', key=f'pre_lambda_{method}', disabled=not baseline_enabled)
            if method == 'AsLS':
                p = b.number_input('ρ (p)', min_value=.000001, max_value=.999999, value=.001,
                                   format='%.6f', key='pre_asls_p', disabled=not baseline_enabled)
                tol = .001
            else:
                p = .001
                tol = b.number_input('Tolerance', min_value=.0000001, value=.001,
                                     format='%.5g', key='pre_arpls_tol', disabled=not baseline_enabled)
            niter = c.number_input('N of iterations', min_value=1, value=50,
                                  step=1, key=f'pre_iterations_{method}', disabled=not baseline_enabled)
            st.caption('원본 전체 범위·point 유지')
            with st.expander('이상치 보정 (선택)'):
                anomaly = st.checkbox('보정 사용', False, key='pre_anomaly', disabled=not baseline_enabled)
                a, b = st.columns(2)
                window = a.number_input('Window', min_value=1, value=5, step=1,
                                        disabled=not anomaly, key='pre_anomaly_window')
                threshold = b.number_input('Threshold', min_value=.01, value=2., step=.1,
                                            disabled=not anomaly, key='pre_anomaly_threshold')
            st.markdown('**3 · Smoothing · Savitzky–Golay**')
            sg_enabled = st.checkbox('Smoothing 사용', False, key='pre_sg_enabled')
            a, b = st.columns(2)
            sg_window = a.number_input('Window (points)', min_value=3, value=11, step=2,
                disabled=not sg_enabled, key='pre_sg_window', help='3 이상의 홀수. 입력 구간의 point 수 이하여야 합니다.')
            sg_order = b.number_input('Polynomial order (n)', min_value=0, value=3, step=1,
                disabled=not sg_enabled, key='pre_sg_order', help='다항식 차수. Window보다 작아야 합니다.')
            st.caption('독립된 SG 결과로 확인·CSV 저장·ML 전달 가능. 원본 파수·point 유지. BLaPE를 끄고 SG만 실행할 수도 있습니다.')
        with blape:
            st.markdown('**4 · BLaPE**')
            enabled = st.checkbox('BLaPE 실행', True, key='pre_use_blape')
            blape_use_sg = st.checkbox('BLaPE 입력에 SG 결과 사용', False,
                disabled=not (enabled and sg_enabled), key='pre_blape_use_sg',
                help='기본 입력은 Baseline 결과입니다. Baseline OFF이면 Raw를 사용합니다. '
                     '이 옵션을 켜면 별도로 계산된 Smoothing 결과를 사용합니다.')
            a, b, c = st.columns(3)
            sigma = a.number_input('σ (points)', min_value=.01, value=30., step=1.,
                                   disabled=not enabled, key='pre_sigma')
            eps = b.number_input('ε', min_value=.000001, value=.25, format='%.5g',
                                 disabled=not enabled, key='pre_eps')
            edge = c.number_input('Edge trim', min_value=0, value=10, step=1,
                                  disabled=not enabled, key='pre_edge')
            with st.expander('제외 구간 (modified BLaPE)'):
                exclusion_text = st.text_input('구간: 500,540; 950,1000', '',
                                               disabled=not enabled, key='pre_exclusions')
                margin = st.number_input('여유 폭 (cm⁻¹)', min_value=0., value=0., step=1.,
                                         disabled=not enabled, key='pre_exclusion_margin')
            st.caption('Edge trim 후 남은 원본 파수·point 유지')
        a, b, c, d = st.columns([1.15, 1, 1, 1.7], vertical_alignment='bottom')
        use_final = a.checkbox('5 · Final 범위 추출', False, key='pre_use_final')
        lower = b.number_input('하한 (cm⁻¹)', value=400., step=1.,
                               disabled=not use_final, key='pre_final_lower')
        upper = c.number_input('상한 (cm⁻¹)', value=1500., step=1.,
                               disabled=not use_final, key='pre_final_upper')
        with d:
            left, right = st.columns(2)
            run = left.button('전처리 실행', type='primary', width='stretch', key='pre_run')
            existing = st.session_state.session
            final_only = right.button('Final만 재추출', key='pre_final_only', width='stretch',
                disabled=not (use_final and existing and existing.asls_result),
                help='완료된 마지막 단계에서 범위를 추출합니다. 다른 단계 설정 변경은 전처리 실행으로 적용하세요.')
        st.caption('Final: 양끝 포함 · 1 cm⁻¹ 간격 (400–1500 → 1,101 points). 외삽하거나 제외 구간을 연결하지 않습니다.')
    return dict(run=run, final_only=final_only, baseline_enabled=baseline_enabled, method=method, lam=lam, p=p, tol=tol,
                niter=int(niter), anomaly=anomaly, window=int(window), threshold=threshold,
                enabled=enabled, sigma=sigma, eps=eps, edge=int(edge), exclusions=exclusion_text,
                margin=margin, use_final=use_final, lower=lower, upper=upper,
                sg_enabled=sg_enabled, blape_use_sg=bool(sg_enabled and blape_use_sg),
                sg_window=int(sg_window), sg_order=int(sg_order))


def _run(settings):
    store = st.session_state.pre_store
    if not store.entries:
        st.warning('파일 선택 후 목록에 추가하세요.')
        return
    try:
        baseline = core.AsLSSettings(lam=float(settings['lam']), p=float(settings['p']),
            niter=settings['niter'], method=settings['method'], tol=float(settings['tol']),
            anomaly=settings['anomaly'], anomaly_window=settings['window'],
            anomaly_threshold=float(settings['threshold']), enabled=settings['baseline_enabled'])
        sg = core.SmoothingSettings(settings['sg_window'], settings['sg_order'])
        blape = core.BLaPESettings(sigma=float(settings['sigma']), eps=float(settings['eps']),
            edge_trim=settings['edge'], target_points=None,
            exclude_ranges=core.parse_exclusions(settings['exclusions']) if settings['enabled'] else (),
            exclusion_margin=float(settings['margin']), sg_enabled=settings['blape_use_sg'],
            sg_window=settings['sg_window'], sg_order=settings['sg_order'])
        input_settings = core.InputSettings()
        load_token = (INPUT_PATCH, core.VERSION, store.revision, input_settings)
        reused = (st.session_state.get('pre_load_token') == load_token
                  and st.session_state.session is not None and bool(st.session_state.session.items))
        session = st.session_state.session if reused else core.SpectraSession()
        logs = []
        def emit(kind, value):
            if kind == 'log':
                logs.append(str(value))
        with st.spinner(f'{settings["method"]} / Smoothing / BLaPE 처리 중…'):
            if not reused:
                session.load(store.paths, input_settings, emit=emit)
            if session.asls_result is None or session.asls_result['params'] != baseline:
                session.run_asls(baseline, emit=emit)
            else:
                logs.append(f'{baseline.method}: 동일 조건의 결과 재사용')
            # This application no longer offers a pre-BLaPE crop.
            session.range_result = None
            if settings['sg_enabled']:
                if (session.smoothing_result is None or session.smoothing_result['params'] != sg
                        or session.smoothing_result['asls_computed_at'] != session.asls_result['computed_at']):
                    session.run_smoothing(sg, emit=emit)
                else:
                    logs.append('Smoothing: 동일 조건의 결과 재사용')
            elif session.smoothing_result is not None:
                session.smoothing_result = None
                session.blape_result = session.final_result = None
            if settings['enabled']:
                if (session.blape_result is None or session.blape_result['params'] != blape
                        or session.blape_result['input_stage'] != ('Smoothing' if settings['blape_use_sg'] else 'AsLS')):
                    session.run_blape(blape, emit=emit, use_range=False)
                else:
                    logs.append('BLaPE: 동일 조건의 결과 재사용')
            else:
                session.blape_result = None
            session.final_result = None
            if settings['use_final']:
                try:
                    session.run_final(core.FinalRangeSettings(settings['lower'], settings['upper']), emit=emit)
                except ValueError as exc:
                    logs.append(f'Final 미완료: {exc}')
                    st.warning(f'이전 단계 완료. Final 범위를 확인하세요: {exc}')
        st.session_state.session = session
        st.session_state.processed = True
        st.session_state.pre_load_token = load_token
        st.session_state.messages = logs
        st.success('전처리 완료')
    except Exception as exc:
        reset_session()
        st.error(str(exc))


def _exports(session, stages):
    names = {'Raw': 'Raw', 'AsLS': f'Baseline ({session.asls_result["params"].method})',
             'Smoothing': 'Smoothing (SG)', 'BLaPE': 'BLaPE', 'Final': 'Final'}
    total = sum(len(item['raw']) for item in session.items)
    kept = total - len(session.excluded_pixels)
    st.caption(f'저장·다음 ML 전달 대상 {kept:,} / {total:,} spectra · 제외 {total-kept:,}개')
    token = (st.session_state.pre_point_generation, id(session))
    if (st.session_state.get('pre_bridge_token') == token and
            st.session_state.get('pre_bridge_exclusions') != frozenset(session.excluded_pixels)):
        st.warning('ML 전달 후 제외 목록이 변경되었습니다. ML로 보내기를 다시 눌러 반영하세요.')
    if not kept:
        st.warning('모든 spectrum이 제외되었습니다. 처리 결과 확인에서 저장할 지점을 복원하세요.')
    if stages == ['Raw']:
        st.info('저장할 처리 단계가 꺼져 있습니다. 원하는 단계를 켜고 전처리를 실행하세요. Raw는 ML로 전달할 수 있습니다.')
    st.caption('각 행의 CSV와 ML은 해당 단계의 결과만 사용합니다. ML 공용 풀에는 마지막으로 보낸 단계가 들어갑니다.')
    snapshot = export_snapshot(session)
    for stage in stages:
        title, csv_col, ml_col = st.columns([2, 1, 1], vertical_alignment='center')
        title.markdown(f'**{names[stage]}**')
        if stage != 'Raw':
            selection = {f'include_{key}': stage == label for key, label in
                         [('asls', 'AsLS'), ('smoothing', 'Smoothing'), ('blape', 'BLaPE'), ('final', 'Final')]}
            csv_col.download_button('CSV 다운로드',
                data=lambda selection=selection: make_export_zip(core, snapshot, **selection),
                file_name=f'RSLab_{stage}_CSV.zip', mime='application/zip',
                disabled=not kept, on_click='ignore', key=f'pre_download_{stage}', width='stretch')
        if ml_col.button('ML로 보내기', key=f'bridge_send_{stage}', disabled=not kept, width='stretch'):
            _send_to_ml(session, stage, names[stage], token)
    with st.expander('여러 단계 CSV를 하나의 ZIP으로 저장'):
        cols = st.columns(4)
        choices = [cols[i].checkbox(names[stage], value=True, key=f'pre_export_{stage}')
                   if stage in stages else False for i, stage in enumerate(('AsLS', 'Smoothing', 'BLaPE', 'Final'))]
        selection = tuple(choices)
        st.download_button('선택 단계 ZIP 다운로드',
            data=lambda: make_export_zip(core, snapshot, selection[0], False, selection[2], selection[3], selection[1]),
            file_name='RSLab_Spectra_results.zip', mime='application/zip',
            disabled=not any(choices) or not kept, on_click='ignore', key='pre_download', width='stretch')
    if st.session_state.get('pre_bridge_stage'):
        st.caption(f'마지막 ML 전달: {st.session_state.pre_bridge_stage} · '
                   f'{st.session_state.get("pre_bridge_count", 0):,} spectra')


def _send_to_ml(session, stage, name, token):
    try:
        pool = make_pool(session, stage)
        if st.session_state.get('ml_pool') is None or st.session_state.ml_pool.signature != pool.signature:
            st.session_state.ml_pool = pool
            for tool in ('svm', 'knn', 'nnls'):
                st.session_state.pop(f'{tool}_pool_assignments', None)
                clear_results(st.session_state, tool)
        for tool in ('svm', 'knn', 'nnls'):
            st.session_state[f'{tool}_input_mode'] = '전처리 데이터'
        st.session_state.pre_bridge_token = token
        st.session_state.pre_bridge_exclusions = frozenset(session.excluded_pixels)
        st.session_state.pre_bridge_stage = name
        st.session_state.pre_bridge_count = len(pool.data.X)
        st.success(f'{name} · {len(pool.data.X):,} spectra 전달. ML 탭에서 Label/TRAIN/TEST를 지정하세요.')
    except Exception as exc:
        st.error(str(exc))


def deployment_status():
    """Compare shipped release, on-disk files and modules actually imported.

    No remote GitHub request: this checks the checkout serving this app.
    Import-time fingerprints stay fixed until the Python modules restart.
    """
    folder = Path(__file__).resolve().parent
    try:
        manifest = json.loads((folder / 'build_manifest.json').read_text(encoding='utf-8'))
        expected = manifest['sha256']
        if not isinstance(expected, dict):
            raise ValueError('Invalid sha256 record')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return dict(ok=False, release=INPUT_PATCH, rows=[], error=f'배포 확인 파일을 읽을 수 없습니다: {exc}')
    rows = []
    for module in (core, workflow):
        path = Path(module.__file__).resolve()
        rows.append(_file_status(path, getattr(module, 'INPUT_PATCH', ''),
                                 getattr(module, 'LOADED_SOURCE_SHA256', ''), expected))
    rows.append(_file_status(Path(__file__).resolve(), INPUT_PATCH, LOADED_SOURCE_SHA256, expected))
    ok = (manifest.get('release') == INPUT_PATCH and compatible_input_files()
          and all(row['일치'] for row in rows))
    return dict(ok=ok, release=manifest.get('release', 'unknown'), rows=rows,
                built_at=manifest.get('built_at_utc', ''), error='')


def _file_status(path, version, loaded_hash, expected):
    try:
        disk_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        disk_hash = ''
    wanted = expected.get(path.name, '')
    return {'파일': path.name, '실행 버전': version,
            '기대 SHA256': wanted, '실행 SHA256': loaded_hash, '디스크 SHA256': disk_hash,
            '일치': bool(wanted and wanted == loaded_hash == disk_hash), '실제 경로': str(path)}


def _deployment_banner():
    status = deployment_status()
    if status['ok']:
        st.success(f"적용 확인 · {INPUT_PATCH} · 실행 코드와 배포 파일 3개 일치")
    else:
        st.warning(f"적용 확인 필요 · 실행 {INPUT_PATCH} / 배포 {status['release']}. "
                   '파일 누락·버전 혼합·재시작 필요 여부를 아래에서 확인하세요.')
    with st.expander('배포 적용 상태 · 파일 확인'):
        st.caption(f"실행 중인 입력 기능: {INPUT_PATCH} · 배포 묶음: {status['release']}")
        commit = checkout_commit()
        st.caption(f'서버 checkout 커밋: {commit or "이 환경에서 확인할 수 없음"}')
        st.caption('커밋이 표시되면 GitHub의 해당 브랜치 최신 커밋과 비교할 수 있습니다. '
                   '파일 일치 표는 이번 패치의 입력·전처리·ML 전달 코드 3개를 검사합니다.')
        if status.get('built_at'):
            st.caption(f"패치 제작 시각 (UTC): {status['built_at']} · 앱 업데이트 시각을 뜻하지 않습니다.")
        if status['error']:
            st.warning(status['error'])
        if status['rows']:
            st.dataframe(status['rows'], hide_index=True, width='stretch')
        st.caption('기대값과 디스크가 다르면 이 ZIP의 4개 파일을 같은 위치에 올리세요. '
                   '디스크와 실행값이 다르면 Streamlit Cloud의 Manage app → Reboot app으로 재시작하세요. '
                   '이 표는 현재 서버의 파일을 확인하며, GitHub 최신 커밋을 직접 조회하지 않습니다.')
        st.caption('이 안내 자체가 보이지 않으면 이전 앱이 실행 중입니다. 배포 대상 저장소·브랜치·메인 파일 경로를 확인하세요.')
        if st.button('적용 상태 다시 확인', key='pre_check_build'):
            st.rerun()


def checkout_commit():
    """Read this server's checkout only; never contact a GitHub remote."""
    try:
        result = subprocess.run(['git', '-C', str(Path(__file__).resolve().parents[1]),
                                 'rev-parse', '--short=12', 'HEAD'],
                                capture_output=True, text=True, timeout=2, check=False)
        value = result.stdout.strip()
        return value if result.returncode == 0 and 7 <= len(value) <= 40 and all(
            c in '0123456789abcdef' for c in value) else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def render():
    _deployment_banner()
    if not compatible_input_files():
        st.error('입력 패치의 파일 버전이 일치하지 않습니다. rslab_web 폴더의 preprocess_core.py, '
                 'preprocess_ui.py, workflow.py를 같은 ZIP의 파일로 교체한 뒤 앱을 재시작하세요.')
        return
    init_state()
    st.caption('Baseline · SG · BLaPE · Final 결과를 따로 확인·내보내기합니다. '
               'BLaPE는 기본으로 Baseline/Raw를 사용하며, SG 결과 사용은 별도로 선택합니다.')
    settings = _settings()
    pending_sg = st.session_state.pop('pre_sg_apply_pending', False)
    if settings['run'] or pending_sg:
        _run(settings)
    elif settings['final_only']:
        try:
            st.session_state.session.run_final(core.FinalRangeSettings(settings['lower'], settings['upper']))
            st.success('Final 재추출 완료')
        except Exception as exc:
            st.error(str(exc))
    session = st.session_state.session
    if not st.session_state.processed or session is None or session.asls_result is None:
        return
    stages = ['AsLS'] if session.asls_result['params'].enabled else []
    if session.smoothing_result is not None:
        stages.append('Smoothing')
    if session.blape_result is not None:
        stages.append('BLaPE')
    if session.final_result is not None:
        stages.append('Final')
        result = session.final_result
        st.caption(f'Final {result["params"].lower:g}–{result["params"].upper:g} cm⁻¹ · '
                   f'{len(result["targets"][0]):,} points · 제외/보호 구간 생략 {result["omitted_points"]:,} points')
    if not stages:
        stages = ['Raw']
    view, smoothing, mapping, export, log = st.tabs(['처리 결과 확인', 'SG 미리보기', 'Mapping 지점 선택',
                                        'CSV · ML 내보내기', '처리 기록'])
    with view:
        a, b, c = st.columns([1, 2, 1])
        stage = a.selectbox('표시 단계', stages, index=len(stages)-1,
            format_func=lambda v: session.asls_result['params'].method if v == 'AsLS' else v, key='pre_view_stage')
        fi = b.selectbox('파일', range(len(session.items)),
                         format_func=lambda i: session.items[i]['path'].name, key='pre_view_file')
        item = session.items[fi]
        number_key = f'pre_view_pixel_{fi}_{len(item["raw"])}'
        pi = c.number_input('Spectrum 번호', min_value=1, max_value=len(item['raw']), value=1,
                            step=1, key=number_key) - 1
        format_names = {'single': 'Single', 'mapping': 'Mapping · 좌표 있음',
                        'matrix': '다중 spectrum · 좌표 없음'}
        st.caption(f'{format_names[item["kind"]]} · {len(item["raw"]):,} spectra × '
                   f'{len(item["axis"]):,} 원본 points · '
                   f'파수 {item["axis"][0]:.3f}–{item["axis"][-1]:.3f} cm⁻¹')
        if item['coords'] is not None:
            x, y = item['coords'][int(pi)]
            st.caption(f'선택 좌표 x={x:g}, y={y:g}')
            location, spectrum = st.columns([1, 3], gap='medium')
            with location:
                # A fresh selection after a preview change also lets a user
                # click a previously selected point after entering a number.
                chart_key = f'pre_map_{st.session_state.pre_point_generation}_{fi}_{int(pi)}'
                st.vega_lite_chart(spec=mapping_spec(session, fi, int(pi)),
                    use_container_width=True, key=chart_key, selection_mode='mapping_pick',
                    on_select=lambda: _mapping_click(chart_key, number_key, len(item['raw'])))
                st.caption('지점을 클릭하면 오른쪽 spectrum이 바뀝니다. 주황 테두리: 현재 선택')
            with spectrum:
                figure = plot_current(core, session, stage, fi, int(pi))
                st.pyplot(figure, dpi=300)
                plt.close(figure)
        elif item['kind'] == 'matrix':
            st.caption('Spectrum 번호는 원본 intensity 열 순서입니다. 파일에 없는 위치 좌표는 생성하지 않습니다.')
        if item['coords'] is None:
            figure = plot_current(core, session, stage, fi, int(pi))
            st.pyplot(figure, dpi=300)
            plt.close(figure)
        _exclusion_controls(session, fi, int(pi))
    with smoothing:
        _smoothing_preview(session, settings)
    with mapping:
        _mapping_tab(session, stages)
    with export:
        _exports(session, stages)
    with log:
        st.caption(f'{session.kind} · {len(session.items)} files · {core.VERSION}')
        st.dataframe([{'파일': item['path'].name, '자동 판별': format_names[item['kind']],
                       'Spectra': len(item['raw']), '원본 points': len(item['axis']),
                       '데이터 시작 행': item.get('data_start_line'),
                       '상단 metadata 행': len(item.get('leading_metadata', []))}
                      for item in session.items], hide_index=True, width='stretch')
        if st.session_state.messages:
            st.code('\n'.join(st.session_state.messages), language='text')
        st.json({name: vars(result['params']) for name, result in
                 [('Baseline', session.asls_result), ('Smoothing', session.smoothing_result), ('BLaPE', session.blape_result),
                  ('Final', session.final_result)] if result is not None})
