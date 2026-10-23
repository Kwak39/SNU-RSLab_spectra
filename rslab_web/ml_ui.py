"""Compact SVM, k-NN and NNLS interfaces with isolated per-session state."""
from __future__ import annotations

import io
import hashlib
import tempfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import __version__, analysis, nnls_core, split_ui, svm_core, blank_gate_core as gate
from .common import (broken_line, content_hashes, csv_bytes, file_pairs, json_bytes,
                     load_files, signature, zip_bytes)
from .workflow import clear_results, nnls_from_subset


INPUT_PATCH = 'GATE-2.7.0'
LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def inputs(tool, role, title, loader):
    key = f'{tool}_{role}'
    uploads = st.file_uploader(title, type=['csv'], accept_multiple_files=True, key=f'{key}_files')
    files = file_pairs(uploads)
    cache_key = f'{key}_parsed'
    if not files:
        if st.session_state.pop(cache_key, None) is not None:
            clear_results(st.session_state, tool)
        return None
    sig = signature(files)
    previous = st.session_state.get(cache_key)
    if previous is None or previous['signature'] != sig:
        try:
            packet = dict(data=load_files(files, loader), signature=sig, hashes=content_hashes(files))
        except Exception as exc:
            st.session_state.pop(cache_key, None)
            clear_results(st.session_state, tool)
            st.error(str(exc))
            return None
        st.session_state[cache_key] = packet
        clear_results(st.session_state, tool)
    packet = st.session_state[cache_key]
    data = packet['data']
    st.caption(f'{len(data) if isinstance(data, list) else len(data.X):,} spectra')
    return packet


def label_controls(tool, role, packet):
    data = packet['data']
    key = f'{tool}_{role}_{packet["signature"][:16]}'
    state_key = key + '_labels'
    if state_key not in st.session_state:
        st.session_state[state_key] = dict(labels=[''] * len(data.X), groups=list(data.groups), revision=0)
    state = st.session_state[state_key]
    with st.expander('TRAIN Label (필수)' if role == 'train' else 'TEST 정답 Label (선택)', expanded=role == 'train'):
        sizes = Counter(data.sources)
        rows = []
        for name, count in sizes.items():
            labels = {state['labels'][i] for i, source in enumerate(data.sources) if source == name}
            rows.append(dict(file=name, spectra=count, label=next(iter(labels)) if len(labels) == 1 else ''))
        table = st.data_editor(pd.DataFrame(rows), hide_index=True, width='stretch', height=min(200, 38+35*len(rows)),
            disabled=['file', 'spectra'], column_config={'file': '파일', 'spectra': 'Spectrum 수', 'label': 'Label'},
            key=f'{key}_files_{state["revision"]}')
        if st.button('파일 Label 적용', key=key + '_apply_files'):
            by_source = {r['file']: str(r['label']).strip() for r in table.fillna('').to_dict('records')}
            state['labels'] = [by_source.get(name, '') for name in data.sources]
            state['revision'] += 1
            clear_results(st.session_state, tool)
            st.rerun()
        with st.expander('개별 Label · Labeling CSV'):
            pages = max(1, (len(data.X) + 499) // 500)
            page = int(st.number_input('페이지 (500개씩)', 1, pages, 1, key=key + '_page'))
            start, stop = (page-1)*500, min(page*500, len(data.X))
            edited = st.data_editor(pd.DataFrame(dict(sample_id=data.sample_ids[start:stop],
                label=state['labels'][start:stop])), hide_index=True, width='stretch', height=200,
                disabled=['sample_id'], key=f'{key}_rows_{page}_{state["revision"]}')
            if st.button('개별 Label 저장', key=key + '_apply_rows'):
                state['labels'][start:stop] = edited['label'].fillna('').astype(str).str.strip().tolist()
                state['revision'] += 1
                clear_results(st.session_state, tool)
                st.rerun()
            st.download_button('Labeling CSV 저장', csv_bytes(['sample_id', 'label'], zip(data.sample_ids, state['labels'])),
                file_name=f'{tool}_{role}_labeling.csv', mime='text/csv', on_click='ignore', key=key + '_csv')
            uploaded = st.file_uploader('Labeling CSV', type=['csv'], key=key + '_csv_upload')
            if st.button('Labeling 불러오기', disabled=uploaded is None, key=key + '_csv_apply'):
                try:
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / 'labels.csv'
                        path.write_bytes(uploaded.getvalue())
                        labels, _, count = svm_core.read_labels(path, data, state['labels'], state['groups'])
                    state.update(labels=labels, revision=state['revision']+1)
                    clear_results(st.session_state, tool)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.caption(f'Label 지정 {sum(bool(x) for x in state["labels"]):,} / {len(data.X):,}')
    return state


def classifier_config(tool):
    if tool == 'svm':
        a, b, c = st.columns(3)
        with a:
            kernel = st.selectbox('Kernel', ['rbf', 'linear', 'poly', 'sigmoid'], key='svm_kernel')
            C = st.number_input('C', min_value=.000001, value=10., format='%.6g', key='svm_C')
        with b:
            gamma = st.text_input('gamma', 'scale', key='svm_gamma', help='scale / auto / 양수')
            balanced = st.checkbox('Balanced class weights', True, key='svm_balanced')
        with c:
            norm = st.selectbox('정규화', ['L2', 'None', 'StandardScaler', 'L2 + StandardScaler'], key='svm_norm')
            cap = st.number_input('Class당 학습 수 (0=전체)', min_value=0, value=500, step=100, key='svm_cap')
        return dict(kernel=kernel, C=C, gamma=gamma.strip(), balanced=balanced,
                    normalization=norm, max_per_class=int(cap))
    a, b = st.columns(2)
    k = a.number_input('k', min_value=1, value=5, step=1, key='knn_k')
    norm = b.selectbox('정규화', ['L2 normalize', 'None', 'z-score per wavenumber',
                               'L2 normalize + z-score'], key='knn_norm')
    return dict(k=int(k), normalization=norm)


def prediction_table(result):
    data = result['data']
    frame = pd.DataFrame(dict(sample_id=data.sample_ids, true_label=result['labels'],
        predicted_label=result['predicted'], zero_signal=result['zero']))
    for k in range(min(3, len(result['classes']))):
        indices = result['ranks'][:, k]
        frame[f'top{k+1}_class'] = result['classes'][indices]
        frame[f'top{k+1}_score'] = result['probabilities'][np.arange(len(frame)), indices]
    return frame


def _classifier_zip(result):
    with tempfile.TemporaryDirectory(prefix='rslab_prediction_') as temp:
        folder = svm_core.save_results(temp, result)
        return zip_bytes({p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()})


def classifier_results(tool, result):
    scores = result['scores']
    st.markdown('**예측 결과**')
    if 'accuracy' in scores:
        st.caption(f'TEST {scores["test_count"]:,} · 정답 Label {scores["labeled_count"]:,} · '
            f'Accuracy {scores["accuracy"]:.3f} · Macro F1 {scores["macro_f1"]:.3f} · '
            f'Top-2 {scores.get("top_2_accuracy", scores["accuracy"]):.3f}')
    else:
        st.caption(f'TEST {scores["test_count"]:,} · 정답 Label 없음: 예측만 표시')
    table, matrix = st.columns([1.8, 1])
    with table:
        frame = prediction_table(result)
        st.dataframe(frame, hide_index=True, width='stretch', height=min(310, 38+35*len(frame)))
        st.download_button('예측 결과 ZIP 다운로드', data=lambda: _classifier_zip(result),
            file_name=f'RSLab_{tool}_results.zip', mime='application/zip',
            key=f'{tool}_download', on_click='ignore')
    with matrix:
        if result['matrix'] is not None:
            values, names = result['matrix'], result['matrix_labels']
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            ax.imshow(values, cmap='Blues')
            ax.set_xticks(range(len(names)), names, rotation=45, ha='right', fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.set(xlabel='Predicted', ylabel='True')
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, str(values[i, j]), ha='center', va='center', fontsize=7,
                            color='white' if values[i, j] > values.max()/2 else 'black')
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    if scores['notes']:
        with st.expander('평가 해석'):
            for note in scores['notes']:
                st.caption(note)


def _data_mode(tool, pool):
    options = ['전처리 데이터', 'CSV 직접 입력'] if pool is not None else ['CSV 직접 입력']
    mode = st.radio('데이터', options, horizontal=True, key=tool + '_input_mode')
    if st.session_state.get(tool + '_last_input_mode') != mode:
        clear_results(st.session_state, tool)
        st.session_state[tool + '_last_input_mode'] = mode
    if mode == 'CSV 직접 입력':
        st.caption('Blank-gate 검토/제외를 적용하려면 입력파일 탭에서 데이터를 불러와 검토한 뒤 ML로 보내세요. 직접 입력 CSV에는 gate를 자동 적용하지 않습니다.')
    return mode == '전처리 데이터'


def _pool_ready(pool):
    info = pool.provenance.get('blank_gate')
    if info:
        policy = '확정 blank 제외' if info['policy'] == 'exclude_confirmed' else '전체 포함 (blank 포함)'
        st.caption(f"전달 데이터: {policy} · gate {'ON' if info['enabled'] else 'OFF'} · "
                   f"확정 blank {info['confirmed_count']:,} · 제외 {info['removed_count']:,}")
        session = st.session_state.get('session')
        if session is not None and (info['state_signature'] != gate.state_signature(session) or
                info['policy'] != st.session_state.get('bg_ml_policy', info['policy']) or
                info.get('manual_exclusions', []) != [list(v) for v in sorted(session.excluded_pixels)]):
            st.warning('Blank 판정/포함 범위/수동 제외가 변경되었습니다. 입력파일 탭에서 ML로 보내기를 다시 눌러주세요.')
            return False
    return True


def _model_bytes(bundle):
    stream = io.BytesIO()
    joblib.dump(bundle, stream)
    return stream.getvalue()


def classifier(tool):
    title = 'SVM' if tool == 'svm' else 'k-NN'
    st.caption(f'{title} · 동일 전처리 조건·파수축의 TRAIN/TEST를 사용하세요.')
    pool = st.session_state.get('ml_pool')
    pooled = _data_mode(tool, pool)
    if pooled and not _pool_ready(pool):
        return
    if pooled:
        train_packet, train_labels, test_packet, test_labels = split_ui.render(tool, pool)
    else:
        a, b = st.columns(2)
        with a:
            train_packet = inputs(tool, 'train', 'TRAIN CSV', svm_core.combine)
            train_labels = label_controls(tool, 'train', train_packet) if train_packet else None
        with b:
            test_packet = inputs(tool, 'test', 'TEST CSV', svm_core.combine)
            test_labels = label_controls(tool, 'test', test_packet) if test_packet else None
    st.markdown('**3 · 모델 학습 · TEST 예측**')
    with st.expander(f'{title} 학습 설정'):
        config = classifier_config(tool)
    train_count = len(train_packet['data'].X) if train_packet else 0
    test_count = len(test_packet['data'].X) if test_packet else 0
    st.caption(f'TRAIN {train_count:,} / TEST {test_count:,}')
    a, b = st.columns(2)
    train_ready = train_packet is not None and train_labels is not None and all(train_labels['labels'])
    if a.button(f'{title} 학습', type='primary', disabled=not train_ready,
                key=tool + '_train_button', width='stretch'):
        try:
            with st.spinner(f'{title} 학습 중…'):
                trainer = svm_core.train_model if tool == 'svm' else analysis.train_knn
                bundle = trainer(train_packet['data'], train_labels['labels'], train_labels['groups'], config)
            bundle['content_hashes'] = train_packet['hashes']
            bundle['metadata'].update(input_signature=train_packet['signature'], web_app_version=__version__,
                training_labels=dict(zip(train_packet['data'].sample_ids, train_labels['labels'])),
                training_groups=dict(zip(train_packet['data'].sample_ids, train_labels['groups'])))
            if 'provenance' in train_packet:
                bundle['metadata']['spectral_preprocessing'] = train_packet['provenance']
            st.session_state[tool + '_bundle'] = bundle
            clear_results(st.session_state, tool)
            st.success(f'{title} 학습 완료')
        except Exception as exc:
            st.error(str(exc))
    bundle = st.session_state.get(tool + '_bundle')
    if b.button('TEST 예측', type='primary', disabled=bundle is None or test_packet is None,
                key=tool + '_predict_button', width='stretch'):
        try:
            with st.spinner('TEST 예측 중…'):
                result = analysis.predict(bundle, test_packet['data'], test_labels['labels'],
                                          test_labels['groups'], test_packet['hashes'])
            st.session_state[tool + '_result'] = result
        except Exception as exc:
            clear_results(st.session_state, tool)
            st.error(str(exc))
    if bundle:
        if train_packet:
            meta = bundle['metadata']
            changed = (meta.get('input_signature') != train_packet['signature'] or
                meta.get('training_labels') != dict(zip(train_packet['data'].sample_ids, train_labels['labels'])))
            if changed:
                st.warning('TRAIN 데이터 또는 Label이 변경되었습니다. 현재 모델에 반영하려면 다시 학습하세요.')
        if config != bundle['metadata'].get('config'):
            st.caption('화면 설정이 현재 모델과 다릅니다. 변경을 반영하려면 다시 학습하세요.')
        with st.expander('학습 모델 정보 · 다운로드'):
            st.json({k: v for k, v in bundle['metadata'].items()
                     if k not in ('selected_training_ids', 'training_labels', 'training_groups')})
            st.download_button('학습 모델 저장 (.joblib)', data=lambda: _model_bytes(bundle),
                file_name=f'RSLab_{tool}_model.joblib', key=tool + '_model_download', on_click='ignore')
    if st.session_state.get(tool + '_result'):
        classifier_results(tool, st.session_state[tool + '_result'])


def _labeled_refs(packet, labels):
    """Name reference templates by Label without changing the spectral arrays."""
    if packet is None:
        return None
    spectra = [replace(s, name=label, source=f'{label} | {s.source}') if label else s
               for s, label in zip(packet['data'], labels)]
    return dict(packet, data=spectra)


def _nnls_zip(result, index):
    spectrum = result['tests'][index]
    grid = result['grid']
    y = np.interp(grid, spectrum.axis, spectrum.y)
    fit = result['norms'][index] * (result['A'] @ result['coeff'][index])
    settings = dict(alpha=result['alpha'], policy=result['policy'], strict_axis=result['strict_axis'],
        references=result['names'], grid=grid, similar_pairs=result['similar_pairs'],
        interpretation='Normalized-reference fitting coefficients; not concentration or detection probability')
    with tempfile.TemporaryDirectory(prefix='rslab_nnls_') as temp:
        path = Path(temp) / 'contributions.csv'
        nnls_core.export_results(path, result)
        return zip_bytes({'contributions.csv': path.read_bytes(), 'settings.json': json_bytes(settings),
            f'fit_{index+1}.csv': csv_bytes(['Raman shift (cm^-1)', 'test', 'fit', 'residual'], zip(grid, y, fit, y-fit))})


def nnls():
    st.caption('Sparse NNLS · REFERENCE는 순물질, TEST는 분석 시료입니다. 기여율은 농도·검출 확률이 아닙니다.')
    pool = st.session_state.get('ml_pool')
    if _data_mode('nnls', pool):
        if not _pool_ready(pool):
            return
        ref_rows, ref_labels, test_rows, _ = split_ui.render('nnls', pool)
        refs = _labeled_refs(nnls_from_subset(pool, ref_rows), ref_labels['labels'])
        tests = nnls_from_subset(pool, test_rows)
    else:
        a, b = st.columns(2)
        with a:
            refs = inputs('nnls', 'refs', 'REFERENCE CSV', nnls_core.load_paths)
        with b:
            tests = inputs('nnls', 'test', 'TEST CSV', nnls_core.load_paths)
    st.markdown('**3 · Reference fitting**')
    with st.expander('NNLS 설정 · 후보 reference', expanded=True):
        a, b, c = st.columns([1, 2, 1.4])
        alpha = a.number_input('α (0=일반 NNLS)', min_value=0., value=.01, step=.005,
                               format='%.5f', key='nnls_alpha')
        policy = b.selectbox('Reference 구성', ['자동 (mapping만 파일별 평균)', '파일별 평균', '각 spectrum'], key='nnls_policy')
        strict = c.checkbox('동일 파수축만 허용', True, key='nnls_strict')
        selected = []
        if refs:
            try:
                _, names = nnls_core.reference_templates(refs['data'], policy)
                selected = st.multiselect('후보 reference', range(len(names)), default=list(range(len(names))),
                    format_func=lambda i: names[i], key='nnls_candidates_' + signature([(n, b'') for n in names])[:12])
            except Exception as exc:
                st.error(str(exc))
        if not strict:
            st.caption('연속 축만 보간합니다. modified BLaPE는 동일 파수축 모드를 사용하세요.')
    if st.button('NNLS fitting 실행', type='primary', disabled=refs is None or tests is None or not selected, key='nnls_fit'):
        try:
            with st.spinner('Reference fitting 중…'):
                result = analysis.fit_nnls(refs['data'], tests['data'], float(alpha), policy, selected, strict)
            st.session_state.nnls_result = result
        except Exception as exc:
            clear_results(st.session_state, 'nnls')
            st.error(str(exc))
    result = st.session_state.get('nnls_result')
    if result is None:
        return
    st.markdown('**Fitting 결과**')
    rows = []
    for i, spectrum in enumerate(result['tests']):
        rel, rmse, status = result['metrics'][i]
        rows.append(dict(sample=spectrum.name, source=spectrum.source, relative_residual=rel,
            RMSE=rmse, status=status,
            **{name: value for name, value in zip(result['names'], nnls_core.contributions(result['coeff'][i]))}))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=min(300, 38+35*len(rows)))
    st.caption(f'결과 설정 α={result["alpha"]:g} · {result["policy"]} · 동일 축={result["strict_axis"]}')
    with st.expander('Fitting · 잔차 확인'):
        index = int(st.number_input('시료 번호', 1, len(rows), 1, key='nnls_plot_index')) - 1
        grid = result['grid']
        spectrum = result['tests'][index]
        y = np.interp(grid, spectrum.axis, spectrum.y)
        fit = result['norms'][index] * (result['A'] @ result['coeff'][index])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3))
        axes[0].plot(*broken_line(grid, y), label='Test')
        axes[0].plot(*broken_line(grid, fit), label='Reconstruction')
        axes[0].legend(fontsize=8)
        axes[1].plot(*broken_line(grid, y-fit), color='#b44646')
        axes[1].set_ylabel('Residual')
        for ax in axes:
            ax.set_xlabel('Raman shift (cm$^{-1}$)')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    if result['similar_pairs']:
        with st.expander('유사한 reference 확인'):
            st.caption('Cosine > 0.98: 유사한 두 계수만으로 혼합물 여부를 확정할 수 없습니다.')
            st.dataframe(pd.DataFrame(result['similar_pairs'], columns=['Reference A', 'Reference B', 'Cosine']), hide_index=True)
    st.download_button('NNLS 결과 ZIP 다운로드', data=lambda: _nnls_zip(result, index),
        file_name='RSLab_NNLS_results.zip', mime='application/zip', key='nnls_download', on_click='ignore')
