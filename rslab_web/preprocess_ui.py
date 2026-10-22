"""Compact input workflow: baseline -> native BLaPE -> final common axis."""
from __future__ import annotations

import copy
import io
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from . import preprocess_core as core
from .common import file_pairs
from .workflow import UploadStore, clear_results, make_pool


def load_core():
    return core, Path(core.__file__)


def init_state():
    if 'workdir' not in st.session_state:
        st.session_state.pre_temp = tempfile.TemporaryDirectory(prefix='rslab_streamlit_')
        st.session_state.workdir = Path(st.session_state.pre_temp.name)
    for key, value in [('session', None), ('processed', False), ('messages', []),
                       ('pre_upload_generation', 0)]:
        st.session_state.setdefault(key, value)
    if 'pre_store' not in st.session_state:
        st.session_state.pre_store = UploadStore(Path(st.session_state.workdir) / 'inputs')


def reset_session(remove_files=False):
    st.session_state.session = None
    st.session_state.processed = False
    st.session_state.messages = []
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
                    include_blape=False, include_final=False):
    """Pure download worker: no Streamlit/session-state access or shared folders."""
    stages = [name for name, selected in [('AsLS', include_asls), ('Range', include_range),
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
    count = 3 if stage in ('BLaPE', 'Final') else 2
    figure, axes = plt.subplots(1, count, figsize=(4.1 * count, 2.9), squeeze=False)
    for row, x, y, label in data['series']:
        axes[0, row].plot(x, y, lw=.9, label=label)
    for axis in axes[0]:
        for lo, hi in data.get('excluded_ranges', []):
            axis.axvspan(lo, hi, color='#d62728', alpha=.15)
        axis.set(xlabel='Raman shift (cm$^{-1}$)', ylabel='Intensity')
        axis.legend(fontsize=7)
    figure.suptitle(data['label'], fontsize=9)
    figure.tight_layout()
    return figure


def _input_files():
    store = st.session_state.pre_store
    st.markdown('**1 · 입력 파일**')
    uploaded = st.file_uploader('Single / Mapping / 다중 spectrum', type=['txt', 'csv', 'dat', 'tsv'],
        accept_multiple_files=True, key=f'pre_upload_{st.session_state.pre_upload_generation}',
        help='2열 single, 좌표가 있는 mapping, 첫 열이 파수축인 다중 spectrum을 함께 넣을 수 있습니다. '
             '여러 폴더에서 선택 → 목록 추가를 반복하세요. 인코딩·구분자·입력 형식을 자동 판별합니다.')
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
            method = st.radio('방법', ['AsLS', 'arPLS'], horizontal=True, key='pre_baseline_method')
            a, b, c = st.columns(3)
            lam = a.number_input('λ', min_value=1., value=1e6 if method == 'AsLS' else 1e5,
                format='%.5g', key=f'pre_lambda_{method}')
            if method == 'AsLS':
                p = b.number_input('p', min_value=.000001, max_value=.999999, value=.001,
                                   format='%.6f', key='pre_asls_p')
                tol = .001
            else:
                p = .001
                tol = b.number_input('Tolerance', min_value=.0000001, value=.001,
                                     format='%.5g', key='pre_arpls_tol')
            niter = c.number_input('반복 횟수', min_value=1, value=10 if method == 'AsLS' else 50,
                                  step=1, key=f'pre_iterations_{method}')
            st.caption('원본 전체 범위·point 유지')
            with st.expander('이상치 보정 (선택)'):
                anomaly = st.checkbox('보정 사용', False, key='pre_anomaly')
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
            st.caption('Baseline 후 → SG smoothing → BLaPE. 원본 파수·point 유지.')
        with blape:
            st.markdown('**4 · BLaPE**')
            enabled = st.checkbox('BLaPE 실행', True, key='pre_use_blape')
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
        use_final = a.checkbox('5 · Final 범위 추출', True, disabled=not enabled, key='pre_use_final')
        lower = b.number_input('하한 (cm⁻¹)', value=400., step=1.,
                               disabled=not (enabled and use_final), key='pre_final_lower')
        upper = c.number_input('상한 (cm⁻¹)', value=1500., step=1.,
                               disabled=not (enabled and use_final), key='pre_final_upper')
        with d:
            left, right = st.columns(2)
            run = left.button('전처리 실행', type='primary', width='stretch', key='pre_run')
            existing = st.session_state.session
            final_only = right.button('Final만 재추출', key='pre_final_only', width='stretch',
                disabled=not (enabled and use_final and existing and existing.blape_result),
                help='기존 BLaPE에 현재 범위만 적용합니다. Baseline/Smoothing/BLaPE 설정 변경은 전처리 실행으로 적용하세요.')
        if sg_enabled and not enabled:
            st.caption('Smoothing은 BLaPE 입력에 적용됩니다. 사용하려면 BLaPE 실행을 켜세요.')
        st.caption('Final: 양끝 포함 · 1 cm⁻¹ 간격 (400–1500 → 1,101 points). 외삽하거나 제외 구간을 연결하지 않습니다.')
    return dict(run=run, final_only=final_only, method=method, lam=lam, p=p, tol=tol,
                niter=int(niter), anomaly=anomaly, window=int(window), threshold=threshold,
                enabled=enabled, sigma=sigma, eps=eps, edge=int(edge), exclusions=exclusion_text,
                margin=margin, use_final=use_final, lower=lower, upper=upper,
                sg_enabled=sg_enabled, sg_window=int(sg_window), sg_order=int(sg_order))


def _run(settings):
    store = st.session_state.pre_store
    if not store.entries:
        st.warning('파일 선택 후 목록에 추가하세요.')
        return
    try:
        baseline = core.AsLSSettings(lam=float(settings['lam']), p=float(settings['p']),
            niter=settings['niter'], method=settings['method'], tol=float(settings['tol']),
            anomaly=settings['anomaly'], anomaly_window=settings['window'],
            anomaly_threshold=float(settings['threshold']))
        blape = core.BLaPESettings(sigma=float(settings['sigma']), eps=float(settings['eps']),
            edge_trim=settings['edge'], target_points=None,
            exclude_ranges=core.parse_exclusions(settings['exclusions']),
            exclusion_margin=float(settings['margin']), sg_enabled=settings['sg_enabled'],
            sg_window=settings['sg_window'], sg_order=settings['sg_order'])
        input_settings = core.InputSettings()
        load_token = (store.revision, input_settings)
        reused = st.session_state.get('pre_load_token') == load_token and st.session_state.session is not None
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
            if settings['enabled']:
                if (session.blape_result is None or session.blape_result['params'] != blape
                        or session.blape_result['input_stage'] != 'AsLS'):
                    session.run_blape(blape, emit=emit, use_range=False)
                else:
                    logs.append('BLaPE: 동일 조건의 결과 재사용')
                session.final_result = None
                if settings['use_final']:
                    try:
                        session.run_final(core.FinalRangeSettings(settings['lower'], settings['upper']), emit=emit)
                    except ValueError as exc:
                        logs.append(f'Final 미완료: {exc}')
                        st.warning(f'Baseline/BLaPE 완료. Final 범위를 확인하세요: {exc}')
            else:
                session.blape_result = session.final_result = None
        st.session_state.session = session
        st.session_state.processed = True
        st.session_state.pre_load_token = load_token
        st.session_state.messages = logs
        st.success('전처리 완료')
    except Exception as exc:
        reset_session()
        st.error(str(exc))


def _exports(session, stages):
    names = {'AsLS': session.asls_result['params'].method, 'BLaPE': 'BLaPE', 'Final': 'Final'}
    left, right = st.columns(2, gap='large')
    with left:
        st.markdown('**CSV ZIP**')
        cols = st.columns(3)
        choices = [cols[i].checkbox(names[stage], value=True, key=f'pre_export_{stage}')
                   if stage in stages else False for i, stage in enumerate(('AsLS', 'BLaPE', 'Final'))]
        snapshot = export_snapshot(session)
        selection = tuple(choices)
        st.download_button('ZIP 만들기 · 다운로드',
            data=lambda: make_export_zip(core, snapshot, selection[0], False, selection[1], selection[2]),
            file_name='RSLab_Spectra_results.zip', mime='application/zip',
            disabled=not any(choices), on_click='ignore', key='pre_download', width='stretch')
    with right:
        st.markdown('**ML 공용 데이터로 전달**')
        stage = st.selectbox('전달 단계', stages, index=len(stages)-1,
                             format_func=lambda value: names[value], key='bridge_stage')
        if st.button('ML로 보내기', key='bridge_send', width='stretch'):
            try:
                pool = make_pool(session, stage)
                if st.session_state.get('ml_pool') is None or st.session_state.ml_pool.signature != pool.signature:
                    st.session_state.ml_pool = pool
                    for tool in ('svm', 'knn', 'nnls'):
                        st.session_state.pop(f'{tool}_pool_assignments', None)
                        clear_results(st.session_state, tool)
                for tool in ('svm', 'knn', 'nnls'):
                    st.session_state[f'{tool}_input_mode'] = '전처리 데이터'
                st.success(f'{names[stage]} · {len(pool.data.X):,} spectra 전달. ML 탭에서 Label/TRAIN/TEST를 지정하세요.')
            except Exception as exc:
                st.error(str(exc))


def render():
    init_state()
    st.caption('파일 누적 → AsLS / arPLS → Savitzky–Golay (선택) → 원본 축 BLaPE → Final 공통축 → ML')
    settings = _settings()
    if settings['run']:
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
    stages = ['AsLS']
    if session.blape_result is not None:
        stages.append('BLaPE')
    if session.final_result is not None:
        stages.append('Final')
        result = session.final_result
        st.caption(f'Final {result["params"].lower:g}–{result["params"].upper:g} cm⁻¹ · '
                   f'{len(result["targets"][0]):,} points · 제외/보호 구간 생략 {result["omitted_points"]:,} points')
    view, export, log = st.tabs(['처리 결과 확인', 'CSV · ML 내보내기', '처리 기록'])
    with view:
        a, b, c = st.columns([1, 2, 1])
        stage = a.selectbox('표시 단계', stages, index=len(stages)-1,
            format_func=lambda v: session.asls_result['params'].method if v == 'AsLS' else v, key='pre_view_stage')
        fi = b.selectbox('파일', range(len(session.items)),
                         format_func=lambda i: session.items[i]['path'].name, key='pre_view_file')
        item = session.items[fi]
        pi = c.number_input('Spectrum 번호', min_value=1, max_value=len(item['raw']), value=1,
                            step=1, key=f'pre_view_pixel_{fi}_{len(item["raw"])}') - 1
        format_names = {'single': 'Single', 'mapping': 'Mapping · 좌표 있음',
                        'matrix': '다중 spectrum · 좌표 없음'}
        st.caption(f'{format_names[item["kind"]]} · {len(item["raw"]):,} spectra × '
                   f'{len(item["axis"]):,} 원본 points')
        if item['coords'] is not None:
            x, y = item['coords'][int(pi)]
            st.caption(f'선택 좌표 x={x:g}, y={y:g}')
            with st.expander('Mapping 위치 확인'):
                fig, ax = plt.subplots(figsize=(3, 2.5))
                ax.scatter(item['coords'][:, 0], item['coords'][:, 1], s=9)
                ax.scatter(x, y, s=60, facecolors='none', edgecolors='orange')
                ax.invert_yaxis()
                ax.set_aspect('equal', adjustable='box')
                st.pyplot(fig)
                plt.close(fig)
        elif item['kind'] == 'matrix':
            st.caption('Spectrum 번호는 원본 intensity 열 순서입니다. 파일에 없는 위치 좌표는 생성하지 않습니다.')
        figure = plot_current(core, session, stage, fi, int(pi))
        st.pyplot(figure)
        plt.close(figure)
    with export:
        _exports(session, stages)
    with log:
        st.caption(f'{session.kind} · {len(session.items)} files · {core.VERSION}')
        st.dataframe([{'파일': item['path'].name, '자동 판별': format_names[item['kind']],
                       'Spectra': len(item['raw']), '원본 points': len(item['axis'])}
                      for item in session.items], hide_index=True, width='stretch')
        if st.session_state.messages:
            st.code('\n'.join(st.session_state.messages), language='text')
        st.json({name: vars(result['params']) for name, result in
                 [('Baseline', session.asls_result), ('BLaPE', session.blape_result),
                  ('Final', session.final_result)] if result is not None})
