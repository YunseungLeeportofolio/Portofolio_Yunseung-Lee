# 부산 청년 통계 결합 분석 (2026 부산 청년 실태보고서 · ANCHOR 연구 프로젝트 빅데이터팀)
#
# 무엇을 하는 코드인가
#   부산 16개 구군의 청년(만 18–39세) 고용 · 건강 · 부채 · 창업 · 전출 지표를 서로 맞대어,
#   함께 움직이는 지표가 있는지 상관분석으로 확인합니다.
#
# 데이터
#   ../data/anchor/district_2023.csv  구군별 지표 (2023년, 신규 사업체만 2022년)
#   ../data/anchor/yearly.csv         부산 전체 연도별 지표 (2014–2023년)
#   출처: 부산광역시 빅데이터 플랫폼 Big-데이터웨이브 (https://data.busan.go.kr/bdip/)
#
# 분석 원칙
#   1. 인원수는 비율로 바꿉니다. 전출인구, 채무보유자, 신규 사업체 수는 구군 인구가 많을수록 커지므로
#      청년 인구로 나눈 값을 씁니다. 인원수끼리 비교하면 구군의 크기만 재게 됩니다.
#   2. 상관계수마다 n, 95% 신뢰구간, p값을 함께 보고하고, 여러 쌍을 동시에 검정하므로
#      Benjamini-Hochberg 방식으로 보정한 p값도 함께 봅니다.
#   3. 연도별 자료는 대부분 해마다 한 방향으로 움직이므로(추세), 원래 값끼리 상관을 내면
#      관계가 없어도 상관이 크게 나옵니다. 그래서 전년 대비 변화량끼리 비교합니다.
#   4. 구군 단위 결과는 개인 단위로 해석하지 않습니다 (생태학적 오류).
#
# 실행
#   pip install pandas numpy scipy plotly kaleido
#   python anchor_statistics.py
#   결과는 이 파일 옆 outputs/ 폴더에 저장됩니다 (표: CSV, 그림: HTML · PNG).

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

# ------------------------------------------------------------------ 0. 설정
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data' / 'anchor'
OUT = HERE / 'outputs'
OUT.mkdir(exist_ok=True)

FONT = 'Pretendard, Apple SD Gothic Neo, Malgun Gothic, Noto Sans CJK KR, sans-serif'
TEAL, CORAL, GREY, INK, MUTED = '#1C7C74', '#C9473C', '#AEB9C7', '#1F2328', '#6E7781'


# ------------------------------------------------------------------ 1. 데이터 불러오기
def load_district():
    """구군별 자료를 읽고, 인원수 지표를 청년 인구 대비 비율로 바꿉니다."""
    d = pd.read_csv(DATA / 'district_2023.csv')
    # 구군별 청년 인구는 원자료에 따로 없어서, 취업자 수와 고용률로 역산합니다.
    # 청년 인구 = 취업자 / 고용률 (예: 부산진구 70.1천 명 / 67.1% ≈ 10.4만 명)
    d['청년인구_추정'] = d['청년취업자_천명'] * 1000 / (d['고용률'] / 100)
    d['시외전출률'] = d['시외전출인구'] / d['청년인구_추정'] * 100        # %
    d['채무보유율'] = d['채무보유자'] / d['청년인구_추정'] * 100          # %
    d['신규사업체율'] = d['신규사업체_2022'] / d['청년인구_추정'] * 1000   # 청년 1,000명당
    return d


def load_yearly():
    """연도별 자료를 읽고, 구직자의 희망임금을 구간 중간값으로 가중평균합니다."""
    y = pd.read_csv(DATA / 'yearly.csv')
    # 희망임금 구간의 중간값(만 원). '250만 원 이상'은 상한이 없어 275만 원으로 가정합니다.
    bins = {'희망임금_100만원미만': 75, '희망임금_100_150': 125, '희망임금_150_200': 175,
            '희망임금_200_250': 225, '희망임금_250이상': 275}
    share = y[list(bins)]
    share = share.div(share.sum(axis=1), axis=0)              # 기타 · 무응답을 빼고 다시 100%로 맞춤
    y['희망임금_가중평균'] = (share * pd.Series(bins)).sum(axis=1)
    return y


# ------------------------------------------------------------------ 2. 상관분석 도구
def correlate(x, y):
    """Pearson r과 95% 신뢰구간, p값, Spearman rho를 함께 계산합니다."""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    n = len(x)
    r, p = stats.pearsonr(x, y)
    rho, p_rho = stats.spearmanr(x, y)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)                 # Fisher z 변환으로 신뢰구간 계산
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return dict(n=n, r=r, ci_low=lo, ci_high=hi, p=p, rho=rho, p_rho=p_rho)


def bh_adjust(p_values):
    """Benjamini-Hochberg 방식으로 여러 검정의 p값을 보정합니다."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def run_pairs(df, pairs):
    rows = [dict(x=x, y=y, **correlate(df[x], df[y])) for x, y in pairs]
    res = pd.DataFrame(rows)
    res['p_보정'] = bh_adjust(res['p'])
    return res


# ------------------------------------------------------------------ 3. 구군 단위 분석 (n = 16)
DISTRICT_PAIRS = [                     # 분석 전에 정한 비교 쌍
    ('고용률', '우울감경험률'),
    ('실업률', '우울감경험률'),
    ('고용률', '스트레스인지율'),
    ('고용률', '주관적건강인지율'),
    ('실업률', '시외전출률'),
    ('채무보유율', '우울감경험률'),
    ('신규사업체율', '실업률'),
]


# ------------------------------------------------------------------ 4. 연도 단위 분석 (변화량 기준)
YEARLY_PAIRS = [
    ('실업률', '우울감경험률'),
    ('희망임금_가중평균', '총전출자수'),
    ('고용률', '순이동률'),
]
MIN_N = 6                              # 변화량이 이보다 적으면 계산하지 않음 (예: 채무보유자는 2019년부터라 4개)


def run_yearly(y):
    diff = y.drop(columns='연도').diff()                    # 전년 대비 변화량
    rows = []
    for a, b in YEARLY_PAIRS:
        level = correlate(y[a], y[b])
        change = correlate(diff[a], diff[b]) if (diff[a].notna() & diff[b].notna()).sum() >= MIN_N else None
        rows.append(dict(x=a, y=b, 원값_r=level['r'], 원값_p=level['p'],
                         변화량_n=change['n'] if change else None,
                         변화량_r=change['r'] if change else None,
                         변화량_p=change['p'] if change else None))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. 그림
def p_label(p):
    return 'p < .001' if p < .001 else f'p = {p:.3f}'.replace('0.', '.')


def fig_scatter(d, x, y, row):
    """한 쌍의 산점도: 구군 이름, 회귀선, r · 신뢰구간 · p값 표시"""
    slope, intercept = np.polyfit(d[x], d[y], 1)
    xs = np.linspace(d[x].min() - 1, d[x].max() + 1, 50)
    color = TEAL if row['p_보정'] < .05 else CORAL if row['p'] < .05 else GREY
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode='lines',
                             line=dict(color=GREY, dash='dash', width=1.5), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=d[x], y=d[y], mode='markers+text', text=d['구군'], textposition='top center',
                             textfont=dict(size=11, color=INK), showlegend=False,
                             marker=dict(size=11, color=color, line=dict(width=1, color='white')),
                             hovertemplate='<b>%{text}</b><br>' + x + ': %{x:.1f}<br>' + y + ': %{y:.1f}<extra></extra>'))
    stat = (f"r = {row['r']:.2f}  (95% CI {row['ci_low']:.2f} ~ {row['ci_high']:.2f}),  {p_label(row['p'])},  "
            f"보정 {p_label(row['p_보정'])},  n = {row['n']}")
    fig.update_layout(
        title=dict(text=f'구군별 청년 {x} × {y} (2023)<br><sup>{stat}</sup>', x=0.02, font=dict(size=17, color=INK)),
        font=dict(family=FONT, size=12, color=INK), width=820, height=560,
        paper_bgcolor='white', plot_bgcolor='white', margin=dict(t=90, l=70, r=30, b=70),
        xaxis=dict(title=f'{x} (%)', showgrid=True, gridcolor='#EAEEF2', zeroline=False),
        yaxis=dict(title=f'{y} (%)', showgrid=True, gridcolor='#EAEEF2', zeroline=False),
    )
    return fig


def fig_summary(res):
    """모든 비교 쌍의 상관계수와 95% 신뢰구간을 한눈에 보는 그림"""
    res = res.sort_values('r')
    labels = res['x'] + ' × ' + res['y']
    colors = [TEAL if q < .05 else CORAL if p < .05 else GREY for p, q in zip(res['p'], res['p_보정'])]
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color='#D8DEE4', width=1))
    fig.add_trace(go.Scatter(
        x=res['r'], y=labels, mode='markers', showlegend=False,
        marker=dict(size=12, color=colors),
        error_x=dict(type='data', symmetric=False, array=res['ci_high'] - res['r'],
                     arrayminus=res['r'] - res['ci_low'], color='#AEB9C7', thickness=2),
        customdata=np.stack([res['p'], res['p_보정'], res['n']], axis=1),
        hovertemplate='%{y}<br>r = %{x:.2f}<br>p = %{customdata[0]:.3f} (보정 %{customdata[1]:.3f})<br>n = %{customdata[2]}<extra></extra>'))
    fig.update_layout(
        title=dict(text='구군 단위 상관계수와 95% 신뢰구간 (n = 16)<br>'
                        '<sup>빨강: p < .05이지만 다중 비교 보정 후에는 유의하지 않음 · 회색: 유의하지 않음</sup>',
                   x=0.02, font=dict(size=17, color=INK)),
        font=dict(family=FONT, size=12, color=INK), width=820, height=460,
        paper_bgcolor='white', plot_bgcolor='white', margin=dict(t=90, l=200, r=30, b=60),
        xaxis=dict(title='Pearson r', range=[-1, 1], showgrid=True, gridcolor='#EAEEF2', zeroline=False),
    )
    return fig


def save(fig, name):
    fig.write_html(OUT / f'{name}.html', include_plotlyjs='cdn')
    try:
        fig.write_image(OUT / f'{name}.png', scale=2)       # kaleido가 설치된 경우에만 PNG 저장
    except Exception as e:
        print(f'  PNG 저장 생략 ({name}): {e.__class__.__name__}')


# ------------------------------------------------------------------ 실행
def main():
    pd.set_option('display.width', 200)
    d, y = load_district(), load_yearly()

    district = run_pairs(d, DISTRICT_PAIRS)
    district.to_csv(OUT / 'district_correlations.csv', index=False, encoding='utf-8-sig')
    print('\n[구군 단위 상관분석, n = 16]')
    print(district.round(3).to_string(index=False))

    yearly = run_yearly(y)
    yearly.to_csv(OUT / 'yearly_correlations.csv', index=False, encoding='utf-8-sig')
    print('\n[연도 단위: 원래 값 vs 전년 대비 변화량]')
    print(yearly.round(3).to_string(index=False))

    trend = correlate(y['연도'], y['희망임금_가중평균'])
    print(f"\n참고: 희망임금 가중평균은 연도와의 상관이 r = {trend['r']:.2f}로 거의 해마다 오르기만 했습니다. "
          '원래 값끼리의 상관은 이 추세 때문에 생긴 것입니다.')

    save(fig_summary(district), 'district_correlations')
    top = district.loc[district['p'].idxmin()]
    save(fig_scatter(d, top['x'], top['y'], top), 'district_top_pair')
    print(f'\n결과 저장: {OUT}')


if __name__ == '__main__':
    main()
