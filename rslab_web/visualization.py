"""Spectrum selection, v6 plotting, portable sessions and gap-aware analysis.

Display transforms never modify preprocessing arrays or the ML data pool.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import is_color_like, to_hex
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, MultipleLocator
from PIL import Image
from scipy.signal import find_peaks

from . import preprocess_core as core

MAX_CURVES = 60
MAX_CANVAS_PIXELS = 16_000_000
RAMAN_LABEL = 'Raman shift (cm⁻¹)'


@dataclass
class Curve:
    uid: str
    name: str
    x: np.ndarray
    y: np.ndarray
    stage: str = 'Uploaded'
    source: str = ''
    x_label: str = RAMAN_LABEL
    y_label: str = 'Intensity'
    bands: list = field(default_factory=list)


def make_curve(name, x, y, *, stage='Uploaded', source='', x_label=RAMAN_LABEL,
               y_label='Intensity', bands=None, identity=''):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError('곡선은 길이가 같은 1차원 X/Y 배열과 최소 2 point가 필요합니다.')
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x) <= 0):
        raise ValueError('곡선 X는 중복 없는 오름차순, X/Y는 유한한 숫자여야 합니다.')
    bands = [dict(v) for v in bands] if bands else [dict(start_index=0, stop_index=len(x))]
    offset = 0
    for band in bands:
        a, b = band['start_index'], band['stop_index']
        if int(a) != a or int(b) != b or a != offset or not a < b <= len(x):
            raise ValueError('곡선의 유효 구간 인덱스를 확인하세요.')
        a,b=int(a),int(b)
        band.update(start_index=a, stop_index=b, lower=float(x[a]), upper=float(x[b-1]))
        offset = b
    if offset != len(x):
        raise ValueError('유효 구간이 전체 출력 point를 포함하지 않습니다.')
    digest = hashlib.sha256((str(identity)+'\0'+str(source)+'\0'+stage).encode())
    for array in (x, y):
        digest.update(memoryview(np.ascontiguousarray(array)).cast('B'))
    digest.update(json.dumps(bands, sort_keys=True).encode())
    # Copy only the selected 1D curve. A row view would retain an entire large
    # mapping matrix after the user clears/recalculates preprocessing results.
    xv, yv = x.copy(), y.copy()
    xv.setflags(write=False); yv.setflags(write=False)
    return Curve(digest.hexdigest()[:24], str(name), xv, yv, stage, str(source),
                 str(x_label), str(y_label), bands)


def session_stages(session):
    if session is None or not session.items:
        return []
    return ['Raw']+[name for name, result in [('AsLS', session.asls_result),
            ('Range', session.range_result), ('BLaPE', session.blape_result),
            ('Final', getattr(session, 'final_result', None))] if result is not None]


def session_curve(session, stage, fi, pi=0, *, average=False):
    if stage not in session_stages(session):
        raise ValueError(f'{stage} 결과가 없습니다.')
    item = session.items[fi]
    if stage == 'Raw':
        axis, rows, bands = item['axis'], item['raw'], None
    else:
        result = {'AsLS': session.asls_result, 'Range': session.range_result,
                  'BLaPE': session.blape_result, 'Final': session.final_result}[stage]
        axis = item['axis'] if stage == 'AsLS' else result['targets'][fi]
        rows = result['arrays'][fi]
        bands = core.bands_for(result, fi) if stage in ('BLaPE', 'Final') else None
    if average:
        kept = [j for j in range(len(rows)) if (fi, j) not in session.excluded_pixels]
        if not kept:
            raise ValueError(f'{item["path"].name}: 저장 제외되지 않은 spectrum이 없습니다.')
        row = np.mean(rows, axis=0) if len(kept) == len(rows) else np.mean(rows[kept], axis=0)
        suffix = f'mean (n={len(kept)})'
    else:
        row = rows[pi]
        suffix = f'x={item["coords"][pi,0]:g}, y={item["coords"][pi,1]:g}' if item['kind'] == 'mapping' else ''
    name = ' · '.join(v for v in (item['path'].name, suffix, stage) if v)
    return make_curve(name, axis, row, stage=stage, source=item['path'].name, bands=bands,
                      y_label='Peak feature' if stage in ('BLaPE','Final') else 'Intensity',
                      identity=f'{fi}/{pi if not average else "mean"}')


def _decoded(payload):
    for encoding in ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr', 'utf-16'):
        try:
            text = payload.decode(encoding)
            if '\x00' not in text:
                return text
        except UnicodeError:
            continue
    raise ValueError('텍스트 인코딩을 읽지 못했습니다.')


def _inferred_bands(axis):
    delta = np.diff(axis)
    cuts = np.flatnonzero(delta > 5*np.median(delta))+1 if len(delta) else np.array([], dtype=int)
    return [dict(start_index=int(a), stop_index=int(b))
            for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(axis)])]


def parse_plot_file(filename, payload, *, layout='auto'):
    """Read v6 XY/multicolumn, WebSuite exports, or native row-wise mapping.

    Native mapping is parsed by the same strict auto-detection engine. Only a
    mean is added by default; individual mapping pixels use the processing tab.
    """
    text = _decoded(payload)
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if len(lines) < 2:
        raise ValueError('두 행 이상의 데이터가 필요합니다.')
    if ',' in lines[0]: delim = ','
    elif '\t' in lines[0]: delim = '\t'
    elif ';' in lines[0]: delim = ';'
    else: delim = r'\s+'
    fields = lambda line: next(csv.reader([line], delimiter=delim)) if len(delim) == 1 else re.split(r'\s+', line.strip())
    first, second = fields(lines[0]), fields(lines[1])
    lowered = [str(v).strip().lower() for v in first]
    # Row-wise mapping header contains at least three numeric spectral values.
    def all_numbers(values):
        try: return bool(len(values)) and np.isfinite(np.asarray(values, dtype=float)).all()
        except ValueError: return False
    header_axis=first[2:] if len(first) == len(second) else first
    delta=np.diff(np.asarray(header_axis,dtype=float)) if all_numbers(header_axis) else np.array([])
    increasing_axis=bool(len(delta)) and (np.all(delta>0) or np.all(delta<0))
    native_mapping = (len(second) >= 5 and len(first) in (len(second), len(second)-2)
                      and increasing_axis
                      and (len(first) != 2)
                      and all_numbers(second))
    if layout not in ('auto','xy','mapping'):
        raise ValueError('파일 구조 선택을 확인하세요.')
    if layout=='mapping' or (native_mapping and layout=='auto'):
        import tempfile
        with tempfile.TemporaryDirectory(prefix='rslab_plot_mapping_') as temp:
            path = Path(temp)/Path(filename.replace('\\','/')).name
            path.write_text(text, encoding='utf-8')
            item = core.read_auto(path, core.InputSettings(encoding='utf-8', delimiter='auto'))
        if item['kind'] == 'mapping':
            return [make_curve(f'{filename} · mean (n={len(item["raw"])})', item['axis'],
                               item['raw'].mean(axis=0), stage='Raw', source=filename)]
    # Mapping column exports start with two coordinate header rows.
    skip = 2 if lowered and lowered[0] == 'x coordinate' else 0
    first_data = fields(lines[skip])
    header = None if all_numbers(first_data) else 0
    frame = pd.read_csv(io.StringIO('\n'.join(lines[skip:])), sep=delim, engine='python',
                        header=header, dtype=str, keep_default_na=False)
    frame = frame.loc[:, (frame != '').any(axis=0)]
    if frame.shape[1] < 2:
        raise ValueError('X/Y 숫자 열을 찾지 못했습니다.')
    columns = [str(v).strip() for v in frame.columns]
    pairs = any('| wavenumber' in c.lower() or c.lower().endswith('| x') for c in columns)
    x_candidates = [i for i,c in enumerate(columns) if any(k in c.lower() for k in ('raman','shift','wavelength','lambda','nm'))]
    xi = x_candidates[0] if x_candidates else 0
    combinations = [(j,j+1) for j in range(0,len(columns)-1,2)] if pairs else [(xi,j) for j in range(len(columns)) if j!=xi]
    curves = []
    for xj, yj in combinations:
        x = pd.to_numeric(frame.iloc[:,xj].str.replace(',', '', regex=False), errors='coerce').to_numpy(float)
        y = pd.to_numeric(frame.iloc[:,yj].str.replace(',', '', regex=False), errors='coerce').to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 2:
            continue
        x, y = x[valid], y[valid]
        order = np.argsort(x, kind='stable'); x, y = x[order], y[order]
        x_label = 'Wavelength (nm)' if any(k in columns[xj].lower() for k in ('wavelength','nm')) else RAMAN_LABEL
        label = columns[yj] if header == 0 else f'Y{yj}'
        curves.append(make_curve(f'{Path(filename).stem} · {label}', x, y, source=filename,
                                 x_label=x_label, y_label='Absorbance' if x_label.startswith('Wavelength') else 'Intensity',
                                 bands=_inferred_bands(x), identity=str(yj)))
    if not curves:
        raise ValueError('유효한 곡선을 찾지 못했습니다.')
    if len(curves)>MAX_CURVES:
        raise ValueError(f'시각화 파일 하나당 {MAX_CURVES}개 이하의 곡선을 사용하세요. Mapping은 전처리 탭에서 선택하세요.')
    return curves


def transform(curve, *, normalization='없음', divisor=1., baseline='없음',
              degree=2, lam=1e6, p=.001, iterations=10):
    """One transform per continuous band; no second baseline on processed stages."""
    y = curve.y.copy()
    raw_stage = curve.stage in ('Raw', 'Uploaded')
    if baseline != '없음' and raw_stage:
        for band in curve.bands:
            a,b = band['start_index'],band['stop_index']
            row = y[a:b]
            if baseline == 'AsLS':
                if len(row)<3: raise ValueError('AsLS에 최소 3 point가 필요합니다.')
                y[a:b] = core.asls(row, lam=lam, p=p, niter=int(iterations))[1]
            elif baseline == 'Polynomial':
                xx = np.linspace(-1,1,len(row))
                fit = np.polynomial.polynomial.polyfit(xx,row,min(int(degree),len(row)-1))
                y[a:b] -= np.polynomial.polynomial.polyval(xx,fit)
            else: raise ValueError('Baseline 방법을 확인하세요.')
    if normalization == 'Min-max':
        span = float(np.ptp(y)); y = (y-y.min())/span if span>0 else np.zeros_like(y)
    elif normalization == 'Maximum':
        maximum=float(y.max())
        if maximum!=0: y/=maximum
    elif normalization == 'Custom divisor':
        if not np.isfinite(divisor) or divisor==0: raise ValueError('사용자 나눗수는 유한한 0 이외 값이어야 합니다.')
        y/=divisor
    elif normalization != '없음':
        raise ValueError('Normalization 방법을 확인하세요.')
    if not np.isfinite(y).all(): raise ValueError('변환 결과에 유한하지 않은 값이 있습니다.')
    return y


def palette_color(name, index, total):
    cmap=mpl.colormaps['tab10' if name=='Origin' else name]
    return to_hex(cmap(index%10 if name=='Origin' else index/max(1,total-1)))


def format_axis_label(label):
    return re.sub(r'cm\s*(?:⁻¹|−1|-1|\^\s*\{?\s*-1\s*\}?)', r'cm$^{-1}$', str(label or ''), flags=re.I)


def plot_config():
    return dict(width=600,height=200,dpi=300,margins=[65,20,30,50],font='DejaVu Sans',
                label_size=12,title_size=14,tick_size=10,legend_size=9,
                stack=False,stack_step=0.,x_auto=True,y_auto=True,xmin=0.,xmax=1.,ymin=0.,ymax=1.,
                bottom=True,top=False,left=True,right=False,x_interval=0.,y_interval=0.,
                xlabel=RAMAN_LABEL,ylabel='Intensity',title='',legend=True,grid=False)


def display_offsets(curves, values, styles, config):
    offsets={}; current=0.
    if config['stack']:
        for curve in reversed(curves):
            y=values[curve.uid]
            offsets[curve.uid]=current-float(y.min())
            current+=max(float(np.ptp(y))*1.1,.01)
    else: offsets={c.uid:0. for c in curves}
    for index,curve in enumerate(curves):
        offsets[curve.uid]+=float(styles[curve.uid].get('offset',0.))+float(config['stack_step'])*index
    return offsets


def draw_figure(curves, values, styles, config, *, peak_marks=None):
    if not curves: raise ValueError('표시할 곡선을 선택하세요.')
    w,h=int(config['width']),int(config['height'])
    if min(w,h)<100 or w*h>MAX_CANVAS_PIXELS: raise ValueError('이미지는 각 변 100 px 이상, 총 1,600만 pixel 이하로 설정하세요.')
    left,right,top,bottom=map(float,config['margins'])
    if min(left,right,top,bottom)<0 or left+right>=w or top+bottom>=h:
        raise ValueError('이미지 크기보다 여백 합계가 작아야 합니다.')
    offsets=display_offsets(curves,values,styles,config)
    available={font.name for font in font_manager.fontManager.ttflist}
    fallbacks=[name for name in ['Malgun Gothic','Noto Sans CJK KR','Noto Sans CJK JP','NanumGothic','DejaVu Sans'] if name in available]
    with mpl.rc_context({'font.family':[config['font']]+fallbacks, 'svg.fonttype':'none'}):
        fig=Figure(figsize=(w/100,h/100),dpi=100);FigureCanvasAgg(fig)
        ax=fig.add_axes([left/w,bottom/h,(w-left-right)/w,(h-top-bottom)/h])
        for curve in curves:
            style=styles[curve.uid];y=values[curve.uid]+offsets[curve.uid]
            xplot,yplot=core.drawing_axis(curve.x,y,curve.bands)
            ax.plot(xplot,yplot,label=style.get('name',curve.name),color=style['color'],lw=style['linewidth'])
            if peak_marks and curve.uid in peak_marks:
                idx=peak_marks[curve.uid];ax.scatter(curve.x[idx],y[idx],color=style['color'],s=20,zorder=3)
        if not config['x_auto']:
            if not config['xmin']<config['xmax']:raise ValueError('X 최소 < X 최대여야 합니다.')
            ax.set_xlim(config['xmin'],config['xmax'])
        else: ax.set_xlim(min(c.x[0] for c in curves),max(c.x[-1] for c in curves))
        if not config['y_auto']:
            if not config['ymin']<config['ymax']:raise ValueError('Y 최소 < Y 최대여야 합니다.')
            ax.set_ylim(config['ymin'],config['ymax'])
        ax.set_xlabel(format_axis_label(config['xlabel']),fontsize=config['label_size'])
        ax.set_ylabel(format_axis_label(config['ylabel']),fontsize=config['label_size'])
        ax.set_title(config['title'],fontsize=config['title_size'])
        ax.tick_params(axis='x',bottom=config['bottom'],top=config['top'],labelbottom=config['bottom'],labeltop=config['top'],labelsize=config['tick_size'])
        ax.tick_params(axis='y',left=config['left'],right=config['right'],labelleft=config['left'],labelright=config['right'],labelsize=config['tick_size'])
        for axis,interval in [(ax.xaxis,config['x_interval']),(ax.yaxis,config['y_interval'])]:
            low,high=axis.get_view_interval()
            # Avoid millions of tick objects for an accidentally tiny interval.
            if interval>0 and abs(high-low)/interval<=500:axis.set_major_locator(MultipleLocator(interval))
            elif interval>0:raise ValueError('눈금이 500개를 넘습니다. 눈금 간격을 늘리세요.')
            else:axis.set_major_locator(MaxNLocator(nbins='auto'))
        if config['grid']:ax.grid(True,alpha=.3)
        if config['legend']:ax.legend(fontsize=config['legend_size'])
        return fig,offsets


def render_png_fixed_size(fig,width,height,dpi):
    pixels=int(width)*int(height)
    if min(width,height)<=0 or pixels>MAX_CANVAS_PIXELS or not 72<=dpi<=1200:
        raise ValueError('PNG 크기/DPI 설정을 확인하세요.')
    scale=max(1.,min(float(dpi)/100,max(1.,(8_000_000/pixels)**.5)))
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=100*scale)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image=image.convert('RGBA')
        if image.size!=(int(width),int(height)):
            image=image.resize((int(width),int(height)),Image.Resampling.LANCZOS)
        output=io.BytesIO();image.save(output,format='PNG',dpi=(dpi,dpi))
    return output.getvalue()


def export_plot(curves,values,styles,config):
    fig,offsets=draw_figure(curves,values,styles,config)
    png=render_png_fixed_size(fig,config['width'],config['height'],config['dpi'])
    svg=io.BytesIO();fig.savefig(svg,format='svg');fig.clear()
    return dict(png=png,svg=svg.getvalue(),offsets=offsets)


def curve_csv(curves,values,styles,offsets=None):
    """Per-curve XY pairs support different axes/counts without interpolation."""
    stream=io.StringIO(newline='');writer=csv.writer(stream)
    writer.writerow([label for curve in curves for label in
                    (f'{styles[curve.uid].get("name",curve.name)} | X',f'{styles[curve.uid].get("name",curve.name)} | Y')])
    for i in range(max((len(c.x) for c in curves),default=0)):
        row=[]
        for curve in curves:
            row.extend([curve.x[i],values[curve.uid][i]+(offsets or {}).get(curve.uid,0.)] if i<len(curve.x) else ['',''])
        writer.writerow(row)
    return stream.getvalue().encode('utf-8-sig')


def curve_txt(curves,values,styles):
    stream=io.StringIO()
    for curve in curves:
        stream.write(f'# {styles[curve.uid].get("name",curve.name)} | {curve.stage}\nX\tY\n')
        np.savetxt(stream,np.column_stack([curve.x,values[curve.uid]]),delimiter='\t',fmt='%.12g')
        stream.write('\n')
    return stream.getvalue().encode('utf-8-sig')


def peaks(curve,y,xmin,xmax,prominence=0.,distance=1):
    found=[]
    for band in curve.bands:
        a,b=band['start_index'],band['stop_index']
        idx,_=find_peaks(y[a:b],prominence=prominence,distance=int(distance))
        found.extend((idx+a).tolist())
    found=np.asarray(found,dtype=int)
    return found[(curve.x[found]>=xmin)&(curve.x[found]<=xmax)]


def maximum(curve,y,xmin,xmax):
    idx=np.flatnonzero((curve.x>=xmin)&(curve.x<=xmax))
    if not len(idx):return None
    i=int(idx[np.argmax(y[idx])])
    return dict(index=i,X=float(curve.x[i]),Y=float(y[i]))


def session_json(curves,styles,config,options):
    return json.dumps(dict(format='RSLabVisualization1',
        curves=[dict(name=styles.get(c.uid,{}).get('name',c.name),x=c.x.tolist(),raw=c.y.tolist(),
                     stage=c.stage,source=c.source,x_label=c.x_label,y_label=c.y_label,bands=c.bands,
                     style=styles.get(c.uid,{})) for c in curves],graph=config,options=options),
        ensure_ascii=False,allow_nan=False).encode('utf-8')


def restore_session(payload):
    raw=json.loads(payload.decode('utf-8-sig'))
    legacy=isinstance(raw,list)
    if not legacy and (not isinstance(raw,dict) or raw.get('format')!='RSLabVisualization1'):
        raise ValueError('지원하는 시각화 세션 JSON이 아닙니다.')
    records=raw if legacy else raw['curves']
    if not isinstance(records,list) or len(records)>MAX_CURVES:
        raise ValueError(f'시각화 세션은 최대 {MAX_CURVES}개 곡선을 지원합니다.')
    curves=[];styles={}
    for i,record in enumerate(records):
        x,y=np.asarray(record['x'],dtype=float),np.asarray(record['raw'],dtype=float)
        if legacy:
            order=np.argsort(x,kind='stable');x,y=x[order],y[order]
        curve=make_curve(record['name'],x,y,stage=record.get('stage','Uploaded'),
                         source=record.get('source','JSON session'),x_label=record.get('x_label',RAMAN_LABEL),
                         y_label=record.get('y_label','Intensity'),bands=record.get('bands') or _inferred_bands(x),identity=f'json/{i}')
        curves.append(curve)
        style=record.get('style',dict(offset=record.get('offset',0.)))
        if not isinstance(style,dict):raise ValueError('세션의 곡선 스타일을 확인하세요.')
        style=dict(style)
        for key in ('offset','linewidth'):
            if key in style:
                style[key]=float(style[key])
                if not np.isfinite(style[key]):raise ValueError('오프셋/선 굵기는 유한한 숫자여야 합니다.')
        if 'linewidth' in style and not .5<=style['linewidth']<=10:raise ValueError('선 굵기는 0.5~10이어야 합니다.')
        if 'color' in style and not is_color_like(style['color']):raise ValueError('세션 색상 형식이 잘못되었습니다.')
        styles[curve.uid]=style
    config={} if legacy else raw.get('graph',{})
    if not isinstance(config,dict):raise ValueError('세션의 그래프 설정을 확인하세요.')
    config={k:value for k,value in config.items() if k in plot_config()}
    complete=dict(plot_config(),**config)
    for key,low,high in [('width',100,10000),('height',100,10000),('dpi',100,1200),
                         ('label_size',6,40),('title_size',6,48),('tick_size',6,32),('legend_size',6,32)]:
        value=complete[key]
        if not isinstance(value,(int,float)) or not np.isfinite(value) or value!=int(value) or not low<=value<=high:
            raise ValueError(f'세션의 {key} 설정을 확인하세요.')
        if key in config:config[key]=int(value)
    if complete['width']*complete['height']>MAX_CANVAS_PIXELS:raise ValueError('세션 이미지가 너무 큽니다.')
    margins=complete['margins']
    if not isinstance(margins,list) or len(margins)!=4 or not all(isinstance(v,(int,float)) and np.isfinite(v) and 0<=v<=2000 for v in margins):
        raise ValueError('세션 여백 설정을 확인하세요.')
    if margins[0]+margins[1]>=complete['width'] or margins[2]+margins[3]>=complete['height']:
        raise ValueError('세션 이미지보다 여백 합계가 작아야 합니다.')
    options={} if legacy else raw.get('options',{})
    if not isinstance(options,dict):raise ValueError('세션의 보기용 처리 설정을 확인하세요.')
    return curves,styles,config,options
