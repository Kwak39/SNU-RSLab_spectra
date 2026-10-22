"""v6 visualization workbench connected directly to the WebSuite session."""
from __future__ import annotations

import hashlib
import io
import json

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from . import visualization as viz


def init_state():
    st.session_state.setdefault('viz_curves', {})
    st.session_state.setdefault('viz_styles', {})
    st.session_state.setdefault('viz_generation', 0)
    st.session_state.setdefault('viz_plot_cache', None)


def clear_visualization():
    for key in list(st.session_state):
        if key.startswith('viz_'):
            del st.session_state[key]


def add_curves(curves):
    init_state()
    collection = st.session_state.viz_curves
    unique = {curve.uid:curve for curve in curves}
    if len(set(collection)|set(unique))>viz.MAX_CURVES:
        raise ValueError(f'비교목록은 최대 {viz.MAX_CURVES}개 곡선입니다. 일부를 삭제한 뒤 추가하세요.')
    added=[uid for uid in unique if uid not in collection]
    collection.update(unique)
    selected=st.session_state.get('viz_selected',list(collection)[:3])
    st.session_state.viz_selected=list(dict.fromkeys(selected+added))
    st.session_state.viz_plot_cache=None
    return len(added)


def _setting(config,key,label,kind='number',**kwargs):
    statekey='viz_cfg_'+key
    widget={'number':st.number_input,'check':st.checkbox,'text':st.text_input}[kind]
    config[key]=widget(label,value=config[key],key=statekey,**kwargs)


def _sources():
    with st.expander('비교목록에 곡선 추가 · 세션 복원',expanded=not st.session_state.viz_curves):
        data_tab,file_tab,session_tab=st.tabs(['전처리 결과 선택','추가 그래프 파일','세션 JSON'])
        session=st.session_state.get('session')
        with data_tab:
            stages=viz.session_stages(session)
            if not stages:
                st.info('전처리 탭에서 파일을 목록에 추가한 뒤 ‘원본 읽기 / 형식 확인’ 또는 ‘분석 실행’을 누르세요.')
            else:
                stage=st.selectbox('추가할 처리 단계',stages,index=len(stages)-1,key='viz_source_stage')
                files=st.multiselect('추가할 파일',range(len(session.items)),default=list(range(min(3,len(session.items)))),
                                     format_func=lambda i:session.items[i]['path'].name,key='viz_source_files')
                average=st.checkbox('파일별 평균 spectrum 사용',value=False,key='viz_average',
                                    help='저장 제외되지 않은 spectrum의 평균입니다. Mapping 전체를 곡선으로 한꺼번에 추가하지 않습니다.')
                requests=[]
                for fi in files:
                    item=session.items[fi]
                    if item['kind']=='mapping' and not average:
                        choice=st.multiselect(f'{item["path"].name} · Mapping pixel',range(len(item['raw'])),default=[0],
                            format_func=lambda pi,it=item:f'#{pi+1} · x={it["coords"][pi,0]:g}, y={it["coords"][pi,1]:g}',
                            key=f'viz_pixels_{fi}',max_selections=viz.MAX_CURVES)
                        requests.extend((fi,pi,False) for pi in choice)
                    else:requests.append((fi,0,average))
                if files and session.kind=='mapping':
                    fi=files[0];item=session.items[fi]
                    with st.expander('첫 선택 파일의 Mapping 위치 확인'):
                        fig,ax=plt.subplots(figsize=(4,3.5))
                        ax.scatter(item['coords'][:,0],item['coords'][:,1],s=10,color='#197a85')
                        idx=[pi for f,pi,avg in requests if f==fi and not avg]
                        if idx:ax.scatter(item['coords'][idx,0],item['coords'][idx,1],s=55,facecolors='none',edgecolors='#e67e22')
                        ax.invert_yaxis();ax.set_aspect('equal',adjustable='box');ax.set(xlabel='x',ylabel='y')
                        st.pyplot(fig,clear_figure=True);plt.close(fig)
                if st.button('선택 spectrum을 비교목록에 추가',disabled=not requests,key='viz_add_session'):
                    try:
                        if len(requests)>viz.MAX_CURVES:raise ValueError('한 번에 추가할 곡선은 60개 이하로 선택하세요.')
                        n=add_curves([viz.session_curve(session,stage,fi,pi,average=avg) for fi,pi,avg in requests])
                        st.success(f'{n}개 곡선을 추가했습니다. 기존 곡선은 유지됩니다.')
                    except Exception as exc:st.error(str(exc))
        with file_tab:
            st.caption('Raman/UV–Vis의 2열·다중 열 CSV/TSV/TXT 및 WebSuite CSV를 읽습니다. '
                       '원본 Mapping은 파일별 평균으로 추가합니다. 개별 pixel은 전처리 결과 선택에서 고르세요.')
            layout_label=st.selectbox('그래프 파일 구조',['자동','열 단위 XY 표','원본 Mapping'],key='viz_import_layout')
            layout={'자동':'auto','열 단위 XY 표':'xy','원본 Mapping':'mapping'}[layout_label]
            uploads=st.file_uploader('그래프용 데이터 파일',type=['csv','tsv','txt','dat','asc'],accept_multiple_files=True,
                                     key=f'viz_upload_{st.session_state.viz_generation}')
            if st.button('그래프 파일 추가',disabled=not uploads,key='viz_add_files'):
                for upload in uploads or []:
                    try:
                        count=add_curves(viz.parse_plot_file(upload.name,upload.getvalue(),layout=layout))
                        st.success(f'{upload.name}: {count}개 추가')
                    except Exception as exc:st.error(f'{upload.name}: {exc}')
        with session_tab:
            st.caption('v6 세션 JSON과 이 통합 버전의 시각화 세션을 지원합니다. 복원 대상은 그래프 곡선·설정이며 ML 모델은 포함하지 않습니다.')
            upload=st.file_uploader('시각화 세션 (.json)',type=['json'],key='viz_json_upload')
            if st.button('시각화 세션 복원',disabled=upload is None,key='viz_restore'):
                try:
                    curves,styles,config,options=viz.restore_session(upload.getvalue())
                    # Restore before graph widgets are instantiated in this rerun.
                    clear_visualization();init_state();add_curves(curves)
                    st.session_state.viz_styles.update(styles)
                    st.session_state.viz_custom_color=True
                    st.session_state.viz_custom_width=True
                    for key,value in config.items():
                        if key in viz.plot_config() and key!='margins':st.session_state['viz_cfg_'+key]=value
                    for i,value in enumerate(config.get('margins',[])):st.session_state[f'viz_margin_{i}']=int(value)
                    st.session_state.viz_restored_options=options
                    st.success('시각화 세션을 복원했습니다.')
                except Exception as exc:st.error(str(exc))


def render():
    init_state()
    st.subheader('Spectrum 시각화 · 비교')
    st.caption('v6 그래프 설정과 WebSuite 전처리 결과를 연결했습니다. 그래프의 normalization·스택·오프셋은 ML 입력을 바꾸지 않습니다.')
    _sources()
    collection=st.session_state.viz_curves
    if not collection:
        st.info('비교목록에 Raw·AsLS·BLaPE·Final 또는 그래프 파일 곡선을 추가하세요. 서로 다른 단계도 함께 비교할 수 있습니다.')
        return
    ids=list(collection)
    def label(uid):return st.session_state.viz_styles.get(uid,{}).get('name',collection[uid].name)
    with st.expander(f'비교목록 관리 · {len(ids)}개 곡선'):
        st.caption('추가된 곡선은 입력/전처리를 다시 계산해도 직접 삭제할 때까지 현재 세션의 비교목록에 남습니다.')
        remove=st.multiselect('삭제할 곡선',ids,format_func=label,key='viz_remove_ids')
        c1,c2=st.columns(2)
        if c1.button('선택 곡선 삭제',disabled=not remove,key='viz_remove'):
            for uid in remove:collection.pop(uid,None);st.session_state.viz_styles.pop(uid,None)
            st.session_state.viz_selected=[uid for uid in st.session_state.get('viz_selected',[]) if uid in collection]
            st.session_state.viz_plot_cache=None
            st.rerun()
        if c2.button('시각화 목록 전체 비우기',key='viz_clear'):
            clear_visualization();st.rerun()
    valid=[uid for uid in st.session_state.get('viz_selected',ids[:3]) if uid in collection]
    st.session_state.viz_selected=valid
    selected=st.multiselect('표시할 곡선 · 선택 순서가 그림 순서',ids,format_func=label,key='viz_selected')
    if not selected:
        st.info('표시할 곡선을 선택하세요.');return
    curves=[collection[uid] for uid in selected]
    opts=dict(normalization='없음',divisor=1.,baseline='없음',degree=2,lam=1e6,p=.001,iterations=10)
    restored=st.session_state.pop('viz_restored_options',{})
    for key,value in restored.items():
        if key in opts:st.session_state['viz_opt_'+key]=value
    graph_tab,peak_tab,export_tab=st.tabs(['그래프 및 설정','피크 · 최대값 분석','그림 · 데이터 내보내기'])
    config=viz.plot_config()
    with graph_tab:
        plot_col,settings_col=st.columns([2.2,1],gap='large')
        with settings_col:
            stack_tab,visual_tab,size_tab,axis_tab=st.tabs(['수직 스택 · 처리','비주얼','크기 · 글꼴','축 설정'])
            with stack_tab:
                _setting(config,'stack','수직 스택 사용','check')
                _setting(config,'stack_step','곡선 간 추가 간격',format='%.6g')
                st.caption('수직 스택은 첫 곡선을 위쪽에 놓습니다. 각 곡선의 기준선을 내려 곡선 간 겹침을 줄입니다.')
                opts['normalization']=st.selectbox('보기용 Normalization',['없음','Min-max','Maximum','Custom divisor'],key='viz_opt_normalization')
                opts['divisor']=st.number_input('사용자 나눗수',value=1.,format='%.6g',key='viz_opt_divisor',disabled=opts['normalization']!='Custom divisor')
                with st.expander('원본/추가 파일의 보기용 baseline (선택)'):
                    st.caption('Raw/Uploaded 단계에만 적용합니다. AsLS·Range·BLaPE·Final에는 추가 baseline을 적용하지 않습니다.')
                    opts['baseline']=st.selectbox('보기용 baseline',['없음','AsLS','Polynomial'],key='viz_opt_baseline')
                    opts['degree']=st.number_input('Polynomial 차수',min_value=1,max_value=10,value=2,key='viz_opt_degree',disabled=opts['baseline']!='Polynomial')
                    opts['lam']=st.number_input('보기용 AsLS λ',min_value=1.,value=1e6,format='%.6g',key='viz_opt_lam',disabled=opts['baseline']!='AsLS')
                    opts['p']=st.number_input('보기용 AsLS p',min_value=.000001,max_value=.999999,value=.001,format='%.6f',key='viz_opt_p',disabled=opts['baseline']!='AsLS')
                    opts['iterations']=st.number_input('보기용 AsLS 반복',min_value=1,max_value=200,value=10,key='viz_opt_iterations',disabled=opts['baseline']!='AsLS')
            with visual_tab:
                palette=st.selectbox('색상 팔레트',['Origin','viridis','plasma','inferno','magma','cividis','turbo','coolwarm','rainbow'],key='viz_palette')
                custom_color=st.checkbox('곡선별 색상 지정',value=False,key='viz_custom_color')
                custom_width=st.checkbox('곡선별 선 굵기 지정',value=False,key='viz_custom_width')
                line_width=st.slider('전체 선 굵기',.5,5.,1.8,.1,key='viz_line_width')
                for i,curve in enumerate(curves):
                    previous=st.session_state.viz_styles.get(curve.uid,{})
                    with st.expander(f'{i+1}. {previous.get("name",curve.name)}'):
                        prefix='viz_style_'+curve.uid+'_'
                        name=st.text_input('곡선 이름',value=previous.get('name',curve.name),key=prefix+'name')
                        default_color=viz.palette_color(palette,i,len(curves))
                        color=st.color_picker('선 색상',value=previous.get('color',default_color),key=prefix+'color') if custom_color else default_color
                        width=st.number_input('개별 선 굵기',min_value=.5,max_value=10.,value=float(previous.get('linewidth',line_width)),step=.1,key=prefix+'linewidth') if custom_width else line_width
                        offset=st.number_input('개별 오프셋',value=float(previous.get('offset',0.)),format='%.6g',key=prefix+'offset')
                        st.session_state.viz_styles[curve.uid]=dict(name=name,color=color,linewidth=width,offset=offset)
            with size_tab:
                st.caption('전체 이미지의 픽셀 크기입니다. DPI를 바꿔도 저장 픽셀 수는 유지합니다.')
                _setting(config,'width','너비 (px)',min_value=100,max_value=10000,step=50)
                _setting(config,'height','높이 (px)',min_value=100,max_value=10000,step=50)
                margins=[]
                for i,(name,default) in enumerate(zip(['왼쪽 여백','오른쪽 여백','위쪽 여백','아래쪽 여백'],config['margins'])):
                    margins.append(st.number_input(name,min_value=0,max_value=2000,value=default,step=5,key=f'viz_margin_{i}'))
                config['margins']=margins
                _setting(config,'dpi','미리보기 / PNG 저장 DPI',min_value=100,max_value=1200,step=50)
                available={f.name for f in fm.fontManager.ttflist}
                fonts=[f for f in ['DejaVu Sans','Arial','Liberation Sans','Times New Roman','Malgun Gothic','NanumGothic','Noto Sans CJK KR','Noto Sans CJK JP'] if f in available]
                if st.session_state.get('viz_cfg_font') not in fonts:st.session_state['viz_cfg_font']=fonts[0] if fonts else 'DejaVu Sans'
                config['font']=st.selectbox('글꼴',fonts or ['DejaVu Sans'],key='viz_cfg_font')
                _setting(config,'label_size','축 제목 글자 크기',min_value=6,max_value=40)
                _setting(config,'title_size','그래프 제목 글자 크기',min_value=6,max_value=48)
                _setting(config,'tick_size','눈금 글자 크기',min_value=6,max_value=32)
                _setting(config,'legend_size','범례 글자 크기',min_value=6,max_value=32)
            with axis_tab:
                xmin=min(float(c.x[0]) for c in curves);xmax=max(float(c.x[-1]) for c in curves)
                config.update(xmin=xmin,xmax=xmax,ymin=min(float(c.y.min()) for c in curves),ymax=max(float(c.y.max()) for c in curves),
                              xlabel=curves[0].x_label,ylabel='Normalized intensity' if opts['normalization']!='없음' else curves[0].y_label)
                _setting(config,'x_auto','X축 자동 범위','check')
                _setting(config,'xmin','X 최소',format='%.6g',disabled=config['x_auto'])
                _setting(config,'xmax','X 최대',format='%.6g',disabled=config['x_auto'])
                _setting(config,'y_auto','Y축 자동 범위','check')
                _setting(config,'ymin','Y 최소',format='%.6g',disabled=config['y_auto'])
                _setting(config,'ymax','Y 최대',format='%.6g',disabled=config['y_auto'])
                for key,name in [('bottom','아래 X 눈금'),('top','위 X 눈금'),('left','왼쪽 Y 눈금'),('right','오른쪽 Y 눈금')]:
                    _setting(config,key,name,'check')
                _setting(config,'x_interval','X 눈금 간격 (0=자동)',min_value=0.,step=1.)
                _setting(config,'y_interval','Y 눈금 간격 (0=자동)',min_value=0.,step=.1)
                _setting(config,'xlabel','X축 제목','text')
                _setting(config,'ylabel','Y축 제목','text')
                _setting(config,'title','그래프 제목','text')
                _setting(config,'legend','범례 표시','check')
                _setting(config,'grid','격자 표시','check')
        styles={c.uid:st.session_state.viz_styles[c.uid] for c in curves}
        try:
            value_key=hashlib.sha256(json.dumps(dict(ids=selected,options=opts),sort_keys=True).encode()).hexdigest()
            cached_values=st.session_state.get('viz_values_cache')
            if not cached_values or cached_values['key']!=value_key:
                cached_values=dict(key=value_key,values={c.uid:viz.transform(c,**opts) for c in curves})
                st.session_state.viz_values_cache=cached_values
            values=cached_values['values']
            key=hashlib.sha256(json.dumps(dict(key=value_key,styles=styles,config=config),sort_keys=True).encode()).hexdigest()
            output=st.session_state.viz_plot_cache
            if not output or output['key']!=key:
                output=dict(key=key,**viz.export_plot(curves,values,styles,config))
                st.session_state.viz_plot_cache=output
            with plot_col:
                st.image(output['png'],width=int(config['width']))
                st.caption(f'{len(curves)}개 곡선 · {config["width"]} × {config["height"]} px · {config["dpi"]} DPI. '
                           '원본 해상도로 계산/내보내기하며 제외 구간은 선으로 연결하지 않습니다.')
        except Exception as exc:
            with plot_col:st.error(str(exc))
            return
    with peak_tab:
        peaks_tab,max_tab=st.tabs(['피크 찾기','맥스 인텐시티'])
        xlow=config['xmin'] if not config['x_auto'] else min(c.x[0] for c in curves)
        xhigh=config['xmax'] if not config['x_auto'] else max(c.x[-1] for c in curves)
        with peaks_tab:
            st.caption('보기용 변환값에서 분석하며 수직 스택/오프셋은 제외합니다. 제외 구간별로 독립적으로 피크를 찾습니다.')
            uid=st.selectbox('분석 곡선',selected,format_func=label,key='viz_peak_curve')
            prominence=st.number_input('최소 prominence',min_value=0.,value=0.,key='viz_peak_prominence')
            distance=st.number_input('최소 피크 간격 (points)',min_value=1,value=1,key='viz_peak_distance')
            curve=collection[uid];idx=viz.peaks(curve,values[uid],xlow,xhigh,prominence,distance)
            frame=pd.DataFrame(dict(index=idx,X=curve.x[idx],Y=values[uid][idx]))
            st.write(f'찾은 피크: {len(frame)}개');st.dataframe(frame,use_container_width=True,hide_index=True)
            st.download_button('피크 결과 CSV',frame.to_csv(index=False).encode('utf-8-sig'),'peaks.csv','text/csv',key='viz_download_peaks')
            if st.checkbox('피크 위치 그림 표시',value=False,key='viz_show_peak_plot'):
                fig,_=viz.draw_figure([curve],values,styles,dict(config,stack=False,stack_step=0.),peak_marks={uid:idx})
                payload=viz.render_png_fixed_size(fig,config['width'],config['height'],config['dpi']);fig.clear()
                st.image(payload,width=int(config['width']))
                st.download_button('피크 위치 PNG',payload,'spectrum_peaks.png','image/png',key='viz_download_peak_png')
        with max_tab:
            st.caption('선택한 처리 단계의 값 또는 보기용 변환값을 사용합니다. 여기서 AsLS를 자동으로 다시 실행하지 않습니다.')
            mode=st.radio('최대값 분석 기준',['선택 단계 원값','보기용 변환값'],horizontal=True,key='viz_max_mode')
            graph_range=st.checkbox('현재 그래프 X 범위 사용',value=True,key='viz_max_graph_range')
            lo,hi=xlow,xhigh
            if not graph_range:
                lo=st.number_input('최대값 X 최소',value=float(xlow),format='%.6g',key='viz_max_lower')
                hi=st.number_input('최대값 X 최대',value=float(xhigh),format='%.6g',key='viz_max_upper')
            rows=[]
            if lo>hi:st.error('최대값 X 최소 ≤ 최대여야 합니다.')
            else:
                for curve in curves:
                    result=viz.maximum(curve,curve.y if mode=='선택 단계 원값' else values[curve.uid],lo,hi)
                    if result:rows.append(dict(curve=styles[curve.uid]['name'],stage=curve.stage,basis=mode,**result))
            frame=pd.DataFrame(rows)
            st.dataframe(frame,use_container_width=True,hide_index=True)
            st.download_button('최대값 결과 CSV',frame.to_csv(index=False).encode('utf-8-sig'),'max_intensity_summary.csv','text/csv',key='viz_download_max')
    with export_tab:
        st.caption('미리보기와 같은 PNG를 저장합니다. CSV는 파일별 X/Y 쌍을 사용하여 서로 다른 원본 파수축도 보존합니다.')
        st.download_button(f'그래프 PNG ({config["width"]} × {config["height"]} px)',output['png'],'spectrum.png','image/png',key='viz_download_png')
        st.download_button('그래프 SVG (벡터)',output['svg'],'spectrum.svg','image/svg+xml',key='viz_download_svg')
        st.download_button('곡선 값 CSV · 스택/오프셋 제외',viz.curve_csv(curves,values,styles),'spectra_values.csv','text/csv',key='viz_download_csv')
        st.download_button('그림 좌표 CSV · 스택/오프셋 포함',viz.curve_csv(curves,values,styles,output['offsets']),'spectra_display.csv','text/csv',key='viz_download_display')
        st.download_button('곡선별 처리 데이터 TXT',viz.curve_txt(curves,values,styles),'spectra_processed.txt','text/plain',key='viz_download_txt')
        st.download_button('시각화 세션 JSON',viz.session_json(curves,styles,config,opts),'rslab_visualization.json','application/json',key='viz_download_json')
        st.caption('시각화 파일은 그래프용입니다. ML 입력은 전처리 탭의 Final 결과를 ML 데이터 풀로 보내세요.')
