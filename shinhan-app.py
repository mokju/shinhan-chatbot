# 신한 AI 공모전

import openai
import streamlit as st
import yfinance
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet
from yahooquery import Ticker

st.header("신한AI, 해외주식 알림이 챗봇")
st.subheader("made by TopGun🛩️")
st.text('반드시 api 키를 입력하고 엔터를 먼저 눌러주세요.')


# api 입력하는 창 만들기
API_KEY = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]", 
                placeholder="본인의 api 키를 입력해 주세요! (sk-...)",
                type="password")

# API_KEY 불러오기
openai.api_key = API_KEY

# Messages Session State 초기화
if "messages1" not in st.session_state:
    st.session_state["messages1"] = ""

if "messages2" not in st.session_state:
    st.session_state["messages2"] = ""

if "messages3" not in st.session_state:
    st.session_state["messages3"] = ""

if "messages4" not in st.session_state:
    st.session_state["messages4"] = ""

if "output" not in st.session_state:
    st.session_state["output"] = ""



# 기업의 회사 정보 추출해주는 것.
BASE_PROMPT_CP = [
        {"role": "system", "content": "The system outputs the company's name and yfinance symbol in the form of {company name: symbol} in a news article."},
        {"role": "user", "content": """
Apple and Nike is developing an AI-powered health coaching service code named Quartz, according to a new report from Bloomberg’s Mark Gurman. The tech giant is reportedly also working on technology for tracking emotions and plans to roll out an iPad version of the iPhone Health app this year.
The AI-powered health coaching service is designed to help users stay motivated to exercise, improve their eating habits and sleep better. The idea behind the service is to use AI and information from a user’s Apple Watch to develop coaching programs specially tailored for them. As with Apple’s other services, the health coaching service is expected to have a monthly fee.
Several teams at Apple are reportedly working on the project, including the company’s health, Siri and AI teams. Gurman writes that the service is planned for next year but notes that it could be postponed or shelved altogether.
In addition, the report says Apple’s Health app will be getting tools for tracking emotions and managing vision conditions, such as nearsightedness. The launch version of the emotion tracker will allow users to log their mood, answer questions about their day and compare their results over time. In the future, Apple reportedly hopes the mood tracker will be able to use algorithms to understand a user’s mood based on their speech, text and other data.
        """},
        {"role": "assistant", "content": """
Apple:AAPL,Nike:NKE
        """}
]

# 뉴스 기사에서 전문 용어를 설명해주는 것.
BASE_PROMPT_TC = [
        {"role": "system", "content": "이 시스템은 뉴스기사에서 전문 기술 용어를 기술1: 설명 형태로 보여주는 시스템이다."},
        {"role": "user", "content": """
Apple and Nike is developing an AI-powered health coaching service code named Quartz, according to a new report from Bloomberg’s Mark Gurman. The tech giant is reportedly also working on technology for tracking emotions and plans to roll out an iPad version of the iPhone Health app this year.
The AI-powered health coaching service is designed to help users stay motivated to exercise, improve their eating habits and sleep better. The idea behind the service is to use AI and information from a user’s Apple Watch to develop coaching programs specially tailored for them. As with Apple’s other services, the health coaching service is expected to have a monthly fee.
Several teams at Apple are reportedly working on the project, including the company’s health, Siri and AI teams. Gurman writes that the service is planned for next year but notes that it could be postponed or shelved altogether.
In addition, the report says Apple’s Health app will be getting tools for tracking emotions and managing vision conditions, such as nearsightedness. The launch version of the emotion tracker will allow users to log their mood, answer questions about their day and compare their results over time. In the future, Apple reportedly hopes the mood tracker will be able to use algorithms to understand a user’s mood based on their speech, text and other data.
        """},
        {"role": "assistant", "content": """
1. AI-powered health coaching service: 인공지능을 활용하여 사용자의 운동 동기부여, 식습관 개선, 수면 향상 등을 도와주는 건강 코칭 서비스입니다.
2. Quartz: 애플과 나이키가 협력하여 개발 중인 AI 기반 건강 코칭 서비스의 코드명입니다.
3. Apple Watch: 애플이 제조하는 스마트워치로, 사용자의 신체 활동, 심박수 등 다양한 건강 데이터를 추적하고 기록할 수 있습니다.
4. iPad version of the iPhone Health app: 아이폰용 건강 앱의 아이패드 버전으로, 애플이 올해 출시할 예정입니다.
5. Emotion tracker: 사용자의 기분을 추적하고 관리하는 도구로, 애플의 건강 앱에 추가될 예정입니다.
6. Vision conditions: 시력 문제를 포함한 다양한 시각 관련 질환을 의미합니다. 예를 들어, 근시(nearsightedness)가 이에 해당합니다.
7. Algorithm: 문제를 해결하기 위해 사용되는 일련의 계산 및 처리 절차로, 여기서는 사용자의 기분을 파악하기 위해 말, 텍스트 및 기타장치 데이터를 분석하는데 사용됩니다.
        """}
]



BASE_PROMPT_SM = [
        {"role": "system", "content": "This system summarizes news articles and shows them in Korean."},
        {"role": "user", "content": """
Apple and Nike is developing an AI-powered health coaching service code named Quartz, according to a new report from Bloomberg’s Mark Gurman. The tech giant is reportedly also working on technology for tracking emotions and plans to roll out an iPad version of the iPhone Health app this year.
The AI-powered health coaching service is designed to help users stay motivated to exercise, improve their eating habits and sleep better. The idea behind the service is to use AI and information from a user’s Apple Watch to develop coaching programs specially tailored for them. As with Apple’s other services, the health coaching service is expected to have a monthly fee.
Several teams at Apple are reportedly working on the project, including the company’s health, Siri and AI teams. Gurman writes that the service is planned for next year but notes that it could be postponed or shelved altogether.
In addition, the report says Apple’s Health app will be getting tools for tracking emotions and managing vision conditions, such as nearsightedness. The launch version of the emotion tracker will allow users to log their mood, answer questions about their day and compare their results over time. In the future, Apple reportedly hopes the mood tracker will be able to use algorithms to understand a user’s mood based on their speech, text and other data.
        """},
        {"role": "assistant", "content": """
애플과 나이키가 AI 기반 건강 코칭 서비스 '쿼츠(Quartz)'를 개발 중이라고 블룸버그의 마크 구르만이 보도했습니다. 애플은 감정 추적 기술을 개발하고 있으며 올해 아이폰 헬스 앱의 아이패드 버전을 출시할 계획이라고 합니다. AI 건강 코칭 서비스는 사용자가 운동을 꾸준히 할 수 있도록 도와주고, 식습관을 개선하고, 수면 상태를 개선하도록 설계되었습니다. 애플의 건강, 시리, AI 팀 등이 프로젝트에 참여하고 있으며 내년 출시를 목표로 하지만 연기되거나 전혀 출시되지 않을 수도 있다고 합니다. 애플 헬스 앱은 근시와 같은 시력 문제를 관리하고 감정을 추적하는 도구를 제공할 예정이며, 사용자의 음성, 텍스트 및 기타 데이터를 기반으로 사용자의 기분을 이해하는 알고리즘을 사용할 수 있도록 발전시키려고 합니다.
        """}
]



BASE_PROMPT_ST = [
        {"role": "system", "content": "주식 투자에 도움이 될 수 있는 정보만 추출한 다음, 요약해서 쉽게 해석을 진행해주는 시스템."},
        {"role": "user", "content": """
{'AAPL': {'maxAge': 86400, 'currentPrice': 172.57, 'targetHighPrice': 190.28, 'targetLowPrice': 107.43, 'targetMeanPrice': 162.85, 'targetMedianPrice': 163.87, 'recommendationMean': 2.0, 'recommendationKey': 'buy', 'numberOfAnalystOpinions': 38, 'totalCash': 55872000000, 'totalCashPerShare': 3.552, 'ebitda': 123788001280, 'totalDebt': 109614997504, 'quickRatio': 0.764, 'currentRatio': 0.94, 'totalRevenue': 385095008256, 'debtToEquity': 176.349, 'revenuePerShare': 24.116, 'returnOnAssets': 0.20559, 'returnOnEquity': 1.4560499, 'grossProfits': 170782000000, 'freeCashflow': 83796623360, 'operatingCashflow': 109583998976, 'earningsGrowth': 0.0, 'revenueGrowth': -0.025, 'grossMargins': 0.43181, 'ebitdaMargins': 0.32145, 'operatingMargins': 0.29163, 'profitMargins': 0.24493, 'financialCurrency': 'USD'}}
        """},
        {"role": "assistant", "content": """
이 정보를 기반으로 한 주식 투자 결정은 다음과 같을 수 있습니다:

Apple의 현재 주가는 $172.57이며, 분석가들은 주식에 긍정적인 전망을 가지고 있습니다. 목표 평균가는 $162.85이며, 이는 현재 주가보다 높습니다.
Apple는 현금 보유액이 크며, 안정적인 재무 상태를 유지하고 있습니다. 또한, 자산 대비 수익성 지표인 자산 수익률과 자기 자본 수익률이 양호합니다.
현금 흐름과 총이익은 크고, 영업 및 순이익 마진도 꽤 높습니다.
매출은 최근 감소했지만, 이익 성장률은 0.0으로 아주 안정적이라고 단정하기는 어렵습니다.
주식 분석가들은 Apple 주식에 대해 매수를 권장하고 있습니다.
Apple의 총 부채는 $109,614,997,504로 상당히 큽니다. 이는 자기 자본 대비 높은 부채를 나타내는데, 이는 재무 위험 요소일 수 있습니다.
총부채가 많아서 이자비용이라든지, 상환할 금액들이 앞으로 추가 발생할 가능성이 커서 재무활동현금흐름은 영업활동현금흐름에 비해 엄청 낮을 수 있습니다.
Apple의 총 매출은 $385,095,008,256으로 매우 높습니다. 하지만 최근 매출 성장률은 -0.025로 감소했습니다.
Apple의 주당 매출액은 $24.116이며, 이는 기업의 각 주식으로 얻을 수 있는 매출을 의미합니다.
Apple의 자유 현금 흐름은 $83,796,623,360로 큰 금액입니다. 이는 기업이 영업 활동과 관련하여 생성하는 현금 흐름을 나타냅니다.
이 정보를 종합해 볼 때, Apple은 안정적인 재무 상태를 유지하고 있으며, 매출은 감소하였지만, 이익은 동일합니다. 현재 주가는 분석가들이 매수를 권장하고 있습니다. 그러나 부채가 높고 최근 매출 성장률이 감소한 점을 고려하여 투자 결정을 내리는 것이 중요합니다. 추가적인 조사와 기업 분석을 통해 개인의 투자 목표와 위험 성향을 고려하여 투자 결정을 내리는 것이 좋습니다. 주식 투자는 위험 요소를 포함하므로 신중한 판단이 필요합니다.
        """}
]



prompt = st.text_input("해외 뉴스 기사 입력", placeholder="해외 뉴스 기사를 입력해 주세요.")



text1 = st.empty()
text1.text_area("기술용어 설명", height=400)

text3 = st.empty()
text3.text_area("뉴스 기사 요약")

text2 = st.empty()
text2.text_area("관련 기업 추출")

text4 = st.empty()
text4.text_area("재무제표 해석")

#text2 = st.text_area('예시2', value=st.session_state['output'])

if st.button("Send"):

    st.session_state["messages1"] = BASE_PROMPT_CP
    st.session_state["messages2"] = BASE_PROMPT_TC
    st.session_state["messages3"] = BASE_PROMPT_SM
    st.session_state["messages4"] = BASE_PROMPT_ST

    with st.spinner("Generating response..."):
        st.session_state["messages1"] += [{"role": "user", "content": prompt}]

        make_code = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=st.session_state["messages1"]
        )

        cp_result = make_code["choices"][0]["message"]["content"]
        
        try:
            data_company = [cp_result.split(',')[i] for i in range(len(cp_result.split(',')))]

            company_code = []
            company_name = []

            for i in range(len(data_company)):
                company_code.append(data_company[i].split(':')[1])
                company_name.append(data_company[i].split(':')[0])
            
            text2.text_area("관련 기업 추출", value=[company_code, company_name])

            today = datetime.today()
            year_ago = today - timedelta(365)

            t = Ticker(company_code[0], asynchronous=True)
            financial_info = t.financial_data

            yf_data = yfinance.download (tickers = company_code[0].strip(), start = year_ago.strftime('%Y-%m-%d'), end = today.strftime('%Y-%m-%d'), interval = "1d")
            yf_df = pd.DataFrame()      # emptry df and assign with column name 
            yf_df['Close'] = yf_data.Close
            yf_df['ret'] = yf_data.Close.pct_change().dropna()
            yf_df.dropna(inplace=True)

            fig, ax = plt.subplots()
            ax = yf_df.Close.plot()

            yf_df.ret.plot(secondary_y=True, ax=ax)
            ax.legend()

            st.text('최근 1년 추이')
            st.pyplot(fig)

            test_df = yf_df.reset_index()[['Date', 'Close']]
            test_df.columns = ['ds', 'y']
            prophet = Prophet(seasonality_mode = 'multiplicative',
                    yearly_seasonality=True, 
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    changepoint_prior_scale=0.5)

            prophet.fit(test_df)

            future_data = prophet.make_future_dataframe(periods = 5, freq = 'd')
            forecast_data = prophet.predict(future_data)

            st.text('5일 이후 예측')
            st.dataframe(forecast_data[['ds','yhat', 'yhat_lower', 'yhat_upper']].tail(5))
            fig1 = prophet.plot(forecast_data)

            st.text('예측 보조 그래프')
            st.pyplot(fig1)



        except:
            text2.text_area("관련 기업 추출", value="기업 정보 없음.")
        





        st.session_state["messages2"] += [{"role": "user", "content": prompt}]

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=st.session_state["messages2"]
        )

        message_response = response["choices"][0]["message"]["content"]
        st.session_state["messages2"] += [
            {"role": "system", "content": message_response}
        ]
        st.session_state['output'] += message_response
        text1.text_area("기술용어 설명", value=st.session_state['output'], height=400)




        st.session_state["messages3"] += [{"role": "user", "content": prompt}]

        response_SM = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=st.session_state["messages3"]
        )

        message_response_SM = response_SM["choices"][0]["message"]["content"]
        text3.text_area("뉴스기사 요약", value=message_response_SM)



        try:
            st.session_state["messages4"] += [{"role": "user", "content": str(financial_info)}]

            response_ST = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=st.session_state["messages4"]
            )

            message_response_ST = response_ST["choices"][0]["message"]["content"]
            text4.text_area("재무제표 해석", value=message_response_ST, height=400)
        except:
            text4.text_area("재무제표 해석", value='정보 없음')


if st.button("Clear"):
    st.session_state["messages1"] = ""
    st.session_state["messages2"] = ""
    st.session_state["messages3"] = ""
    st.session_state["output"] = ""