import openai
import streamlit as st
import yfinance
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from prophet import Prophet


st.header("신한은행, 해외주식 알림이 챗봇")



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


prompt = st.text_input("해외 뉴스 기사 입력", placeholder="해외 뉴스 기사를 입력해 주세요.")



text1 = st.empty()
text1.text_area("기술용어 설명")

text2 = st.empty()
text2.text_area("관련 기업 추출")


#text2 = st.text_area('예시2', value=st.session_state['output'])

if st.button("Send"):

    st.session_state["messages1"] = BASE_PROMPT_CP
    st.session_state["messages2"] = BASE_PROMPT_TC

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
        text1.text_area("기술용어 설명", value=st.session_state['output'])



if st.button("Clear"):
    st.session_state["messages1"] = ""
    st.session_state["messages2"] = ""
    st.session_state["output"] = ""