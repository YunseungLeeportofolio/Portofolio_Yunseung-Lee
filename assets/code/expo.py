import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import pingouin as pg

df = pd.read_excel('expo_data.xlsx',sheet_name='Raw_data')
df.info()

############################인구 통계 ########################
df.rename(columns = {'q1':'성별','q2_1':'나이','aq4':'지역'},inplace= True)
df['성별'].replace({1:0, 2:1},inplace = True) # 0->male,1->female

region_dict = {201: '강서구', 202: '금정구', 203: '기장군',
               204: '남구', 205: '동구', 206: '동래',
               207: '부산진구', 208: '북구',
               209: '사상구', 210: '사하구', 211: '서구',
               212: '수영구', 213: '연제구', 214: '영도구',
               215: '중구', 216: '해운대구'}
df['지역'] = df['지역'].replace(region_dict)
df['지역'].value_counts()
df['나이'].mean()
##################양가 감정###############
df.rename(columns = {'q6':'확신','q7':'불안', 'q8':'비관', 'q9':'만족', 'q10':'기대', 'q11':'흥미', 'q12':'지루', 'q13':'피로', 'q14':'관심', 'q15':'찬성', 'q16':'긍정', 'q17':'화'},inplace=True)

#1. 확신 정규 분포 =1
#2. 불안 정규 분포
df['불안'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['불안'], kde=True)
plt.title('불안의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#3. 비관 정교 분포
df['비관'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['비관'], kde=True)
plt.title('비관의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#4. 만족 정규
df['만족'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['만족'], kde=True)
plt.title('만족의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#5 기대
df['기대'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['기대'], kde=True)
plt.title('기대의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#6. 흥미
df['흥미'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['흥미'], kde=True)
plt.title('훙미의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#7. 지루
df['지루'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['지루'], kde=True)
plt.title('지루의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()
#8. 피로
df['피로'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['피로'], kde=True)
plt.title('피로의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()
#9. 관심
df['관심'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['관심'], kde=True)
plt.title('관심의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()
#10. 찬성
df['찬성'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['찬성'], kde=True)
plt.title('찬성의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()
#11. 긍정
df['긍정'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['긍정'], kde=True)
plt.title('긍정의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()
#12, 화
df['화'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['화'], kde=True)
plt.title('화의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

# 양가감정 변수 생성
df['양가감정'] = df[['확신','불안', '비관', '만족', '기대','흥미','지루','피로','관심', '찬성', '긍정', '화']].mean(axis=1)
df['양가감정']
pg.cronbach_alpha(df[['확신','불안', '비관', '만족', '기대','흥미','지루','피로','관심', '찬성', '긍정', '화']]) #Cronbach's alpha = .94


##################### 소셜미디어 콘텐츠 노출 ###########
df.rename(columns = {'q18':'연예인_TV','q19':'연예인_노래', 'q20':'예능', 'q21':'웹툰', 'q22':'공식영상', 'q23':'총회', 'q24':'행사영상', 'q25':'챌린지', 'q26':'응원송', 'q27':'위촉식', 'q28':'뉴스_다큐', 'q29':'연예인강연', 'q30':'정치인강연', 'q31':'릴레이', 'q32':'일반인브이', 'q33':'뉴스_특정인물'},inplace=True)
df['미디어이용'] = df[['연예인_TV','연예인_노래', '예능', '웹툰', '공식영상', '총회', '행사영상', '챌린지', '응원송', '위촉식', '뉴스_다큐', '연예인강연', '정치인강연', '릴레이', '일반인브이', '뉴스_특정인물']].mean(axis=1)
pg.cronbach_alpha(df[['연예인_TV','연예인_노래', '예능', '웹툰', '공식영상', '총회', '행사영상', '챌린지', '응원송', '위촉식', '뉴스_다큐', '연예인강연', '정치인강연', '릴레이', '일반인브이', '뉴스_특정인물']]) #Cronbach's alpha = .96
# 정규분포
df['미디어이용'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['미디어이용'], kde=True)
plt.title('미디어이용의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

##################### 시민 대화####################
df.rename(columns = {'q34':'연인대화','q35':'배우자대화', 'q36':'가족혹친척대화', 'q37':'친구대화', 'q38':'지인대화'},inplace=True)
df['시민대화'] = df[['연인대화','배우자대화', '가족혹친척대화', '친구대화', '지인대화']].mean(axis=1)
pg.cronbach_alpha(df[['연인대화','배우자대화', '가족혹친척대화', '친구대화', '지인대화']]) #Cronbach's alpha = .90

# 정규분포
df['시민대화'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['시민대화'], kde=True)
plt.title('시민대화의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#####################부산 만족도 #####################
df['부산만족도'] = df[['q39','q40']].mean(axis=1)
pg.cronbach_alpha(df[['q39','q40']]) #Cronbach's alpha = .86

# 정규분포
df['부산만족도'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['부산만족도'], kde=True)
plt.title('부산만족도의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

##################### 사회 신뢰 ###################
df['사회신뢰'] = df[['q41','q42']].mean(axis=1)
pg.cronbach_alpha(df[['q41','q42']]) #Cronbach's alpha = .85

# 정규분포
df['사회신뢰'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['사회신뢰'], kde=True)
plt.title('사회신뢰의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

##################### 엑스포 관심 ###################
df.rename(columns = {'q43':'엑스포관심'},inplace= True)

# 정규분포
df['엑스포관심'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['엑스포관심'], kde=True)
plt.title('엑스포관심의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

################### 내적 효능감 #################
df['내적효능감'] = df[['q44','q45']].mean(axis=1)
pg.cronbach_alpha(df[['q44','q45']]) #Cronbach's alpha = .65

# 정규분포
df['내적효능감'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['내적효능감'], kde=True)
plt.title('내적효능감의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#################### 연결적 사회자본 ##############
df['연결사회자본'] = df[['q46','q47','q48']].mean(axis=1)
pg.cronbach_alpha(df[['q46','q47','q48']]) #Cronbach's alpha = .87

# 정규분포
df['연결사회자본'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['연결사회자본'], kde=True)
plt.title('연결사회자본의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#################### 결속적 사회자본 ##############
df['결속사회자본'] = df[['q49','q50','q51']].mean(axis=1)
pg.cronbach_alpha(df[['q49','q50','q51']]) #Cronbach's alpha = .89

# 정규분포
df['결속사회자본'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['결속사회자본'], kde=True)
plt.title('결속사회자본의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#################### 시민참여 ##############
df['시민참여'] = df[['q52','q53','q54','q55','q56','q57','q58','q59','q60','q61','q62']].mean(axis=1)
pg.cronbach_alpha(df[['q52','q53','q54','q55','q56','q57','q58','q59','q60','q61','q62']]) #Cronbach's alpha = .95
df['시민참여'].mean()
# 정규분포
df['시민참여'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['시민참여'], kde=True)
plt.title('시민참여의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

################### 오프라인 시민참여 ###########
df['오프라인시민참여'] = df[['q52','q53','q54','q55','q56','q57']].mean(axis=1)
pg.cronbach_alpha(df[['q52','q53','q54','q55','q56','q57']]) #Cronbach's alpha = .90

# 정규분포
df['오프라인시민참여'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['오프라인시민참여'], kde=True)
plt.title('오프라인시민참여의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

#################### 온라인 시민참여 ##############
df['온라인시민참여'] = df[['q58','q59','q60','q61','q62']].mean(axis=1)
pg.cronbach_alpha(df[['q58','q59','q60','q61','q62']]) #Cronbach's alpha = .95

# 정규분포
df['온라인시민참여'].value_counts()
plt.rcParams['font.family'] = 'AppleGothic'
sns.histplot(df['온라인시민참여'], kde=True)
plt.title('온라인시민참여의 정규분포')
plt.xlabel('확신 값')
plt.ylabel('빈도')
plt.show()

df['한경'] = df[['q68','q69']].mean(axis=1)
pg.cronbach_alpha(df[['q74','q75','q76','q77','q78']]) #Cronbach's alpha = .95
df['q80_1'].mean()

# 동영상 유형
Celebrityfeatured_TV_commercials = df['q18'].mean()

Musicvideos_by_celebrities = df['q19'].mean()

Talkshows_comedy_shows = df['q20'].mean()

Animationscartoons = df['q21'].mean()

Officialcommittee_videos = df['q22'].mean()

Expoconference_videos = df['q23'].mean()

Exporelated_event_promotions = df['q24'].mean()

Citizenschallenge_videos = df['q25'].mean()

Citizenscheering_songs = df['q26'].mean()

Inaugurationceremony_videos = df['q27'].mean()

Expo_relatednews_documents = df['q28'].mean()

Celebritypublic_speeches = df['q29'].mean()

Speechesby_politicians_businessman = df['q30'].mean()

Citizensrelay_cheering_videos = df['q31'].mean()

CitizensVlog_videos = df['q32'].mean()

Video_clipsfocusing_on_specific_individuals = df['q33'].mean()


# 상호작용 그래프
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.factorplots import interaction_plot

emotion = df['양가감정']
emotion.value_counts()
talk = df['시민대화']
talk.value_counts()

df['q18'].value_counts()
conditions = [df['q18'].isin([1, 2]), df['q18'].isin([3, 4, 5])]

# 범주화 과정
categories = ['low', 'high']
Celebrity_featured_TV_commercials_about_the_Expo_bid = pd.cut(df['q18'], bins=[1, 2, 5], labels=categories)
Celebrity_featured_TV_commercials_about_the_Expo_bid.value_counts()
df['Celebrity_featured_TV_commercials_about_the_Expo_bid'] = Celebrity_featured_TV_commercials_about_the_Expo_bid

categories = ['low', 'high']
animations_and_cartoons_about_the_Expo_bid= pd.cut(df['q21'], bins=[1, 2, 5], labels=categories)
animations_and_cartoons_about_the_Expo_bid.value_counts()
df['animations_and_cartoons_about_the_Expo_bid'] = animations_and_cartoons_about_the_Expo_bid


categories = ['low', 'high']
videos_for_the_Expo_bid_created_by_the_official_committee = pd.cut(df['q22'], bins=[1, 2, 5], labels=categories)
videos_for_the_Expo_bid_created_by_the_official_committee.value_counts()
df['videos_for_the_Expo_bid_created_by_the_official_committee'] = videos_for_the_Expo_bid_created_by_the_official_committee

categories = ['low', 'high']
Expo_related_conference_videos = pd.cut(df['q23'], bins=[1, 2, 5], labels=categories)
Expo_related_conference_videos.value_counts()
df['Expo_related_conference_videos'] = Expo_related_conference_videos

categories = ['low', 'high']
citizens_s_challenge_videos = pd.cut(df['q25'], bins=[1, 2, 5], labels=categories)
citizens_s_challenge_videos.value_counts()
df['citizens_s_challenge_videos'] = citizens_s_challenge_videos

categories = ['low', 'high']
inauguration_ceremony_videos = pd.cut(df['q27'], bins=[1, 2, 5], labels=categories)
inauguration_ceremony_videos.value_counts()
df['inauguration_ceremony_videos'] = inauguration_ceremony_videos

categories = ['low', 'high']
celebrity_public_speeches = pd.cut(df['q29'], bins=[1, 2, 5], labels=categories)
celebrity_public_speeches.value_counts()
df['celebrity_public_speeches'] = celebrity_public_speeches

categories = ['low', 'high']
speeches_by_politicians_businessmen = pd.cut(df['q30'], bins=[1, 2, 5], labels=categories)
speeches_by_politicians_businessmen.value_counts()
df['speeches_by_politicians_businessmen'] = speeches_by_politicians_businessmen

categories = ['low', 'high']
citizens_s_v_log_videos = pd.cut(df['q32'], bins=[1, 2, 5], labels=categories)
citizens_s_v_log_videos.value_counts()
df['citizens_s_v_log_videos'] = citizens_s_v_log_videos

categories = ['low', 'high']
edited_video_clips_focusing_on_specific_individuals = pd.cut(df['q33'], bins=[1, 2, 5], labels=categories)
edited_video_clips_focusing_on_specific_individuals.value_counts()
df['edited_video_clips_focusing_on_specific_individuals'] = edited_video_clips_focusing_on_specific_individuals

categories = ['low', 'high']
Exporelated_event_promotions = pd.cut(df['q24'], bins=[1, 2, 5], labels=categories)
Exporelated_event_promotions.value_counts()
df['Exporelated_event_promotions'] = edited_video_clips_focusing_on_specific_individuals


df = df[['Exporelated_event_promotions','양가감정','videos_for_the_Expo_bid_created_by_the_official_committee','Expo_related_conference_videos','animations_and_cartoons_about_the_Expo_bid','citizens_s_challenge_videos','inauguration_ceremony_videos','celebrity_public_speeches','speeches_by_politicians_businessmen', '시민대화','citizens_s_v_log_videos', 'Celebrity_featured_TV_commercials_about_the_Expo_bid','edited_video_clips_focusing_on_specific_individuals']]


# 'videos_for_the_Expo_bid_created_by_the_official_committee'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='videos_for_the_Expo_bid_created_by_the_official_committee', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Videos for the Expo bid created by the official committee")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

# 'Expo_related_conference_videos'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='Expo_related_conference_videos', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Expo-related conference videos")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

#'animations_and_cartoons_about_the_Expo_bid'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='animations_and_cartoons_about_the_Expo_bid', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Animations and cartoons about the Expo bid")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

#'citizens_s_challenge_videos'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='citizens_s_challenge_videos', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Citizens\'s challenge videos")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

#'inauguration_ceremony_videos'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='inauguration_ceremony_videos', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Inauguration/ceremony videos")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

#'celebrity_public_speeches'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='celebrity_public_speeches', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Celebrity public speeches")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

# 'speeches_by_politicians_businessmen'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='speeches_by_politicians_businessmen', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Speeches by politicians/businessmen")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

# 'citizens_s_v_log_videos'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='citizens_s_v_log_videos', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Citizens\'s vlog videos")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

# 'Celebrity_featured_TV_commercials_about_the_Expo_bid'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='Celebrity_featured_TV_commercials_about_the_Expo_bid', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Celebrity-featured TV commercials about the Expo bid")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

# 'edited_video_clips_focusing_on_specific_individuals'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='edited_video_clips_focusing_on_specific_individuals', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Edited video clips focusing on specific individuals")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()

# 'Expo-related event promotions'
plt.figure(figsize=(8, 6))
sns.set(style="white")
g = sns.lmplot(x='양가감정', y='시민대화', hue='Celebrity_featured_TV_commercials_about_the_Expo_bid', data=df, ci=None, scatter=False, legend=False)  # legend 매개변수를 False로 설정하여 범례 제거
plt.grid(False)  # 그리드 제거
plt.xlabel("Emotion")
plt.ylabel("Talk")
plt.title("Expo-related event promotions")
handles = g._legend_data.values()
labels = ['low', 'high']  # 원하는 범례 레이블 설정
plt.legend(handles=handles, labels=labels, loc='upper right')
plt.show()
