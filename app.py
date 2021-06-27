import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
# embed streamlit docs in a streamlit app
#components.iframe("https://docs.streamlit.io/en/latest")
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
figure(figsize=(15, 6), dpi=80)




#code_jlkb22SU0Jku6ik7dMLO


html_temp = """
<div style = "background.color:teal; padding:10px">
<h2 style = "color:white; text_align:center;"> Mono ML Transaction Classifer</h2>
<p style = "color:white; text_align:center;"> </p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


#st.cache()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.write("Connect to your financial account or use mono test account with GTbank")
# bootstrap 4 collapse example
components.html(
    """
    <!DOCTYPE html>
<html lang="en">
<head>
    <title>Click here to Connect Mono</title>
    <style>
        .p-5 {
            padding: 5em;

        }
    </style>
    <script type="application/javascript" src="https://connect.withmono.com/connect.js"></script>
</head>
<body>
<div className="p-5">
    <p>Welcome to Mono Connect.</p>
    <button id="launch-btn">Link a financial account</button>
</div>
<script type="application/javascript">
  const copyToClipboard = text => {
    const elm = document.createElement('textarea');
    elm.value = text;
    document.body.appendChild(elm);
    elm.select();
    document.execCommand('copy');
    document.body.removeChild(elm);
  };
  
  var connect;
  var config = {
    key: "test_pk_n5y3gX6wbeYvyCqdSaJR",
    onSuccess: function (response) {
      copyToClipboard(response.code);
      console.log(JSON.stringify(response));
      alert("The code has been copied to your clipboard kindly paste on the text field below to continue");
      console.log(JSON.stringify(response));
      
      /**
       response : { "code": "code_xyz" }
       you can send this code back to your server to get this
       authenticated account and start making requests.
       */
    },
    onClose: function () {
      console.log('user closed the widget.')
    }
  };
  connect = new Connect(config);
  connect.setup();
  var launch = document.getElementById('launch-btn');
  launch.onclick = function (e) {
    connect.open();
  };
</script>
</body>
</html>
    """,
    height=400,scrolling=True
)

import requests
import json

def mono_auth(codenum):
  url = "https://api.withmono.com/account/auth"
  payload = json.dumps({
  "code": str(codenum)
  })
  headers = {
      'Content-Type': 'application/json',
      'mono-sec-key': 'test_sk_GG76pmCI8SQtDNSPl03S'
  }
  response = requests.request("POST", url, headers=headers, data=payload)
  id = response.json()
  return id

def get_trans(id):
  url = "https://api.withmono.com/accounts/"+str(id)+"/transactions"
  querystring = {"paginate":"true"}
  payload={}
  headers = {
    'mono-sec-key': 'test_sk_GG76pmCI8SQtDNSPl03S'
  }
  response = requests.request("GET", url, headers=headers, data=payload, params=querystring)
  result = response.json()
  return result


def prediction(narration):
    url = "https://monotransapi-emekaborisama.cloud.okteto.net/mono_transapi"
    payload={'text': narration}
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)
    return(response.text)

date = []
type_ = []
narration =[]
category = []


def convert_dataframe(result):
    for i in result['data']:
        narration.append(i["narration"])
        type_.append(i['type'])
        date.append(i['date'])
        category.append(prediction(i['narration']))
    date_ = pd.DataFrame(date,columns=['date'])
    type_s = pd.DataFrame(type_,columns=['type_'])
    narrations = pd.DataFrame(narration,columns=['narration'])
    category_ = pd.DataFrame(category,columns=['labels'])
    frame = [date_, type_s, narrations, category_]
    df = pd.concat(frame, axis = 1)
    #cat = pd.DataFrame(l,columns=[['date', 'type_', 'narration']])
    return df

st.write('------------------------------------------------------------------------')
st.write("Dataset used for this model")
st.code("https://github.com/Emekaborisama/mono_transaction_classifierapp/tree/master/app/data")

st.markdown("Notebook and API code can be found in the github link below")

st.code("https://github.com/Emekaborisama/mono_transaction_classifierapp/")

st.subheader("Test Transaction classifier model")
coden = st.text_input("paste the code here")
col1, col2 = st.beta_columns(2)

if col1.button("Test with Mono data"):
    
    results = mono_auth(coden)
    esults = get_trans(results['id'])
    with st.spinner('Wait for it...'):
        df = convert_dataframe(result = esults)
        st.balloons()
        st.success('Done!')
    st.subheader("chart")
    catcount = df.groupby('labels').count()
    plt.bar(catcount.index.values, catcount['narration'])
    plt.xlabel("Categories")
    plt.ylabel('Number of transactions')
    st.pyplot()
    st.write(df)
    df.to_csv("data.csv")


if col2.button("Test with with your input data"):
    inputdata = st.text_input("Transaction description")
    if st.button("submit"):
        st.markdown(prediction(inputdata))





st.write('------------------------------------------------------------------------')
st.subheader("Conclusion and further work")

st.markdown("So far we have been able to build a transaction classifer with xgboost and we achieved 92% " "accuracy on the test data with a balanced trade off between variance and bias")



st.markdown("Further work will be data annotation, to improve the model performance, Build another model using Transformer technique, and A/B testing")



st.write('--------------------------------------------------------------------------')
primaryColor = st.get_option("theme.primaryColor")
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; background-color:#000; color:#fff;}}
<style>
"""
st.markdown(s, unsafe_allow_html=True)