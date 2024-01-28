from sklearn.feature_extraction import DictVectorizer   # 데이터 실수화

v = DictVectorizer(sparse=False)
D = [{"feature1" : 1, "feature" : 3, "feature3" : 5},
     {"feature1" : 6, "feature2" : 0, "feature3": 8},
     {"feature1" : 2, "feature2" : 4, "feature3" : 9}]

print(v.fit_transform(D))    #범주형 자료에서 수량형 자료로 변경

#범주형 자료
x=[{'city':'Paris', 'temp':1.0}, 
   {'city':'Seoul', 'temp':10.0}, 
   {'city':'Sydney', 'temp':20.0}]
x
print(x)


vec = DictVectorizer(sparse=False)    
vec.fit_transform(x) #X를 범주형 수량화 자료로 변환
print(vec.fit_transform(x))


#텍스트 자료
text = ["한 꼬마 두 꼬마 세 꼬마 인디언", 
        "네 꼬마 다섯 꼬마 여섯 꼬마 인디언",
        "일곱 꼬마 여덟 꼬마 아홉 꼬마 인디언",
        "열 꼬마 인디언 보이 열 꼬마 아홉 꼬마 여덟 꼬마 인디언",
        "일곱 꼬마 여섯 꼬마 다섯 꼬마 인디언", 
        "네 꼬마 세 꼬마 두 꼬마 인디언 한 꼬마 인디언 보이"]
from sklearn.feature_extraction.text import CountVectorizer # 단어의 출현 횟수 

vec2 = CountVectorizer() #sparse=True(디폴트옵션)     # 단어의 출현 횟수 
t = vec2.fit_transform(text).toarray() #sparse=True를 풀고 text를 수량화 배열 자료로 변환     #toarray() 배열로 보기 # fit_transform() 범주형 자료에서 수량형 자료로 변경   즉, 텍스트 자료에서 수량형 자료로 변경함.
print(t)        # 각 행마다 단어 반복이 얼마만큼 되는지 카운트한 값 출력 
import pandas as pd
t1 = pd.DataFrame(t, columns=vec2.get_feature_names_out())        # 각각의 데이터프레임 획득 가능 # columns 열은 vec2의 featue 이름을 꺼내옴 
print(t1)

from sklearn.feature_extraction.text import TfidfVectorizer     # TF-IDF 자주 등장하여 분석에 의미를 갖지 못하는 단어의 중요도를 낮추는 기법(ex 관사 the, a 등 을, 를 이 가) 

tfid = TfidfVectorizer() #sparse=True(디폴트옵션)
x2 = tfid.fit_transform(text).toarray() #sparse=True를 풀고 text를 수량화 배열 자료로 변환
x3 = pd.DataFrame(x2, columns=tfid.get_feature_names_out())
print(x3)