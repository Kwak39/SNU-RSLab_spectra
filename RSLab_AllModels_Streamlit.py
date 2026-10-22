"""Run: python -m streamlit run RSLab_AllModels_Streamlit.py"""
import matplotlib
matplotlib.use('Agg')

import streamlit as st

from rslab_web import __version__
from rslab_web import ml_ui, preprocess_ui
from rslab_web.workflow import clear_ml


APP_ID = 'rslab-allmodels'
APP_TABS = ['입력파일 · 전처리', 'SVM', 'k-NN', 'Sparse NNLS']


def main():
    st.set_page_config(page_title='RSLab All Models v1.3', page_icon='🧪', layout='wide')
    if tuple(int(x) for x in st.__version__.split('.')[:2]) < (1, 52):
        st.error('Streamlit 1.52 이상이 필요합니다. Update_RSLab.bat을 실행한 뒤 서버를 다시 시작하세요.')
        st.stop()
    st.markdown('''<style>
        .block-container {max-width:1600px; padding-top:1.2rem; padding-bottom:1rem;}
        [data-testid="stVerticalBlock"] {gap:0.65rem;}
        h1 {font-size:1.7rem !important; padding-bottom:0.3rem !important;}
        h2,h3 {padding-top:0.4rem !important; padding-bottom:0.35rem !important;}
    </style>''', unsafe_allow_html=True)
    st.title('RSLab Spectra · All Models')
    st.caption(f'AsLS / arPLS · BLaPE · SVM / k-NN / NNLS | v{__version__}')
    with st.sidebar:
        st.header('RSLab Spectra')
        st.caption('입력파일 → 전처리 → ML로 보내기 → Label/TRAIN/TEST 배정 → 학습·예측')
        if st.button('ML 전달 데이터 · 모델 · 결과 비우기', key='clear_ml'):
            clear_ml(st.session_state)
            st.rerun()
        if st.button('현재 세션 전체 비우기', key='clear_all'):
            temp = st.session_state.get('pre_temp')
            if temp:
                temp.cleanup()
            st.session_state.clear()
            st.rerun()
        st.caption('업로드 데이터와 결과는 현재 세션에서 사용됩니다. 필요한 결과와 모델은 다운로드하세요.')
    tabs = st.tabs(APP_TABS)
    renderers = [preprocess_ui.render, lambda: ml_ui.classifier('svm'),
                 lambda: ml_ui.classifier('knn'), ml_ui.nnls]
    for tab, render in zip(tabs, renderers):
        with tab:
            try:
                render()
            except Exception as exc:
                st.error(f'이 탭을 처리하지 못했습니다: {exc}')
                with st.expander('오류 상세'):
                    st.exception(exc)


if __name__ == '__main__':
    main()
