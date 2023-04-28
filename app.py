import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from streamlit_option_menu import option_menu
from PIL import Image

diabetes=joblib.load('D:\PycharmProjects\Sem-6 Health AI\pickle\diabetes_model.pkl')
kidney=joblib.load('D:\PycharmProjects\Sem-6 Health AI\pickle\kidney_model.pkl')
liver=joblib.load('D:\PycharmProjects\Sem-6 Health AI\pickle\liver_model.pkl')
cancer=joblib.load('D:\PycharmProjects\Sem-6 Health AI\pickle\cancer_model.pkl')
heart=joblib.load('D:\PycharmProjects\Sem-6 Health AI\pickle\heart_model.pkl')

with st.sidebar:
   add_selectbox=option_menu(
      menu_title='Select The Diseas',options=["Home","Diabetes", "Kidney", "Liver","Breast Cancer","Heart"],default_index=0,
                             orientation='vertical')



# add_selectbox=option_menu(
#       menu_title=None,options=["Home","Diabtes", "Kidney", "Liver","Cancer","Heart"],default_index=0,menu_icon='cast',
#                              orientation='horizontal')

# add_selectbox = st.sidebar.selectbox(
#     "How would you like to be contacted?",
#     ("Diabetes", "Kidney", "Liver","Breast Cancer","Heart")
#     ,orientation="horizontal"
# )




#Pregnancies	Glucose	  BloodPressure	   SkinThickness	Insulin	
# BMI	DiabetesPedigreeFunction	Age	
if add_selectbox=='Home':


# Set page title
   # st.set_page_config(page_title="My Home Page", page_icon=":house_with_garden:")

   # Define background color and font style
   st.markdown("""
   <style>
   body {
      background-color: #f0f2f6;
      font-family: 'Helvetica', sans-serif;
   }
   </style>
   """, unsafe_allow_html=True)

   # Add a banner image
   image = Image.open("images.png")
   st.image(image, caption="", use_column_width=True)

   # Create a title for the page
   st.title("Welcome AI-Hospital")

   # Add some text to introduce the page
   st.write("AI in hospitals can not only ease hospital patient flow, but it can also help develop pharmaceutical drugs, keep and analyze data and patient records, and even help diagnose illnesses like cancer. When people spend less time in the ER waiting room, more people can be treated in a timely manner.")
   col1,col2,col3=st.columns(3)

   with col1:
      img_1=Image.open('diabetes.jfif')
      st.image(img_1, caption="", use_column_width=True)
      st.write("AI in hospitals can not only ease hospital patient flow, but it can also help develop pharmaceutical drugs, keep and analyze data and patient records, and even help diagnose illnesses like cancer.")
   with col2:
      img_2=Image.open('download.jfif')
      st.image(img_2, caption="", use_column_width=True)
      st.write("AI in hospitals can not only ease hospital patient flow, but it can also help develop pharmaceutical drugs, keep and analyze data and patient records, and even help diagnose illnesses like cancer.")
   with col3:
      img_3=Image.open('Home.jfif')
      st.image(img_3, caption="", use_column_width=True)
      st.write("AI in hospitals can not only ease hospital patient flow, but it can also help develop pharmaceutical drugs, keep and analyze data and patient records, and even help diagnose illnesses like cancer.")
   
   # Create a sidebar menu with icons
   menu = ["About :information_source:", "Contact :email:"]
   choice = st.sidebar.selectbox("Select an option", menu)

   # Show content based on user's choice
   # if choice == "About :information_source:":
   st.subheader("About This Site")
   st.write("AI in hospitals can not only ease hospital patient flow, but it can also help develop pharmaceutical drugs, keep and analyze data and patient records, and even help diagnose illnesses like cancer. When people spend less time in the ER waiting room, more people can be treated in a timely manner.")
   # elif choice == "Contact :email:":
   st.subheader("Contact Us")
   st.write("You can contact us at contact@example.com.")

#Diabetessssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss

if add_selectbox=="Diabetes":   #1
   st.title(f'{add_selectbox} Diseas Detection')
   st.markdown("""
   <style>
   body {
      background-color: #f0f2f6;
      font-family: 'Helvetica', sans-serif;
      
   }
   </style>
   """, unsafe_allow_html=True)

   # Add a banner image
   image = Image.open("diabetes.jfif")
   st.image(image, caption="", use_column_width=True)
   col1,col2=st.columns(2)
   with col1:
      
      p=st.number_input('Enter Pregnancies')
      g=st.number_input('Enter Glucose')
      b=st.number_input('Enter BloodPressure')
   # s=st.number_input('Enter d')
   # i=st.number_input('Enter s')
   with col2:
      bi=st.number_input('Enter BMI')
      dp=st.number_input('Enter DiabetesPedigreeFunction')
      a=st.number_input('Enter Age')

   btn=st.button('Submit')

   if btn:
      

      pred_list=np.array([p,g,b,bi,dp,a]).reshape(1,6)
      pred_list1=np.array([6,146,75,33,0.64,50]).reshape(1,6)
      your=[10,20,30]
      requir=[120,80,90]
      name=['p','g','b']
      df=pd.DataFrame(list(zip(name,your,requir)),columns=['Nutritian','Yours','Required'])
      c1,c2,c3=st.columns(3)
      with c2:

         st.write(df)
      bar_plot=px.bar(df,x='Nutritian',y=['Required','Yours'],title='Health Graph',barmode='group')
      st.plotly_chart(bar_plot)

      result=diabetes.predict(pred_list1)
      prob=diabetes.predict_proba(pred_list1)

      if result[0]==0:
         st.header(f'You are health with chance of{prob[0][0]}')
      else:
         st.header(f'You Have aa diseas with probability {prob[0][1]}')
      lst=['Probbility of Patient do not have Diseas','Probability of Patient have Diseas']

      # st.header(result[0])
      st.header(prob[0])
      df1=pd.DataFrame(list(zip(lst,prob[0])),columns=['Region','Prob'])
      st.write(df1)
      bar_plot1=px.pie(df1,values='Prob',names='Region',title='Health Graph')
      st.plotly_chart(bar_plot1)


if add_selectbox=="Kidney":  #2
   st.title(f'{add_selectbox} Diseas Detection')

   p=st.number_input('Enter Pregnancies')
   g=st.number_input('Enter Glucose')
   b=st.number_input('Enter BloodPressure')
   # s=st.number_input('Enter d')
   # i=st.number_input('Enter s')
   bi=st.number_input('Enter BMI')
   dp=st.number_input('Enter DiabetesPedigreeFunction')
   a=st.number_input('Enter Age')
   temp=st.number_input('Tenter temp')

   btn=st.button('Submit')

   if btn:
      

      pred_list=np.array([p,g,b,bi,dp,a,temp]).reshape(1,7)
      pred_list1=np.array([6,146,75,33,0.64,50,70]).reshape(1,7)
      your=[10,20,30]
      requir=[120,80,90]
      name=['p','g','b']
      df=pd.DataFrame(list(zip(name,your,requir)),columns=['Nutritian','Yours','Required'])


      st.text(df)
      bar_plot=px.bar(df,x='Nutritian',y=['Required','Yours'],title='Health Graph',barmode='group')
      st.plotly_chart(bar_plot)


      result=kidney.predict(pred_list1)
      prob=kidney.predict_proba(pred_list1)
      if result[0]==0:
         st.header(f'You are health with chance of Kidney{prob[0][0]}')
      else:
         st.header(f'You Have aa diseas with probability Kidney{prob[0][1]}')

      st.header(result[0])
      st.header(prob)


if add_selectbox=="Liver":  #3
   st.title(f'{add_selectbox} Diseas Detection')

   p=st.number_input('Enter Pregnancies')
   g=st.number_input('Enter Glucose')
   b=st.number_input('Enter BloodPressure')
   # s=st.number_input('Enter d')
   # i=st.number_input('Enter s')
   bi=st.number_input('Enter BMI')
   dp=st.number_input('Enter DiabetesPedigreeFunction')
   a=st.number_input('Enter Age')
   temp=st.number_input('Tenter temp')

   btn=st.button('Submit')

   if btn:
      

      pred_list=np.array([p,g,b,bi,dp,a,temp]).reshape(1,7)
      pred_list1=np.array([6,146,75,33,0.64,50,70]).reshape(1,7)
      your=[10,20,30]
      requir=[120,80,90]
      name=['p','g','b']
      df=pd.DataFrame(list(zip(name,your,requir)),columns=['Nutritian','Yours','Required'])


      st.text(df)
      bar_plot=px.bar(df,x='Nutritian',y=['Required','Yours'],title='Health Graph',barmode='group')
      st.plotly_chart(bar_plot)

      result=liver.predict(pred_list1)
      prob=liver.predict_proba(pred_list1)
      if result[0]==0:
         st.header(f'You are health with chance of Liver{prob[0][0]}')
      else:
         st.header(f'You Have aa diseas with probability Liver{prob[0][1]}')

      st.header(result[0])
      st.header(prob)

if add_selectbox=="Breast Cancer":  #4
   st.title(f'{add_selectbox} Diseas Detection')

   p=st.number_input('Enter Pregnancies')
   g=st.number_input('Enter Glucose')
   b=st.number_input('Enter BloodPressure')
   # s=st.number_input('Enter d')
   # i=st.number_input('Enter s')
   bi=st.number_input('Enter BMI')
   dp=st.number_input('Enter DiabetesPedigreeFunction')
   

   btn=st.button('Submit')

   if btn:
      

      pred_list=np.array([p,g,b,bi,dp]).reshape(1,5)
      pred_list1=np.array([6,146,75,33,0.64]).reshape(1,5)
      your=[10,20,30]
      requir=[120,80,90]
      name=['p','g','b']
      df=pd.DataFrame(list(zip(name,your,requir)),columns=['Nutritian','Yours','Required'])


      st.text(df)
      bar_plot=px.bar(df,x='Nutritian',y=['Required','Yours'],title='Health Graph',barmode='group')
      st.plotly_chart(bar_plot)


      result=cancer.predict(pred_list1)
      prob=cancer.predict_proba(pred_list1)
      if result[0]==0:
         st.header(f'You are health with chance of cancer{prob[0][0]}')
      else:
         st.header(f'You Have aa diseas with probability cancer{prob[0][1]}')

      st.header(result[0])
      st.header(prob)


if add_selectbox=="Heart":  #5
   st.title(f'{add_selectbox} Diseas Detection')

   p=st.number_input('Enter Pregnancies')
   g=st.number_input('Enter Glucose')
   b=st.number_input('Enter BloodPressure')
   # s=st.number_input('Enter d')
   # i=st.number_input('Enter s')
   bi=st.number_input('Enter BMI')
   dp=st.number_input('Enter DiabetesPedigreeFunction')
   a=st.number_input('Enter Age')
   temp=st.number_input('Tenter temp')

   btn=st.button('Submit')

   if btn:
      

      pred_list=np.array([p,g,b,bi,dp,a,temp]).reshape(1,7)
      pred_list1=np.array([6,146,75,33,0.64,50,70]).reshape(1,7)
      your=[10,20,30]
      requir=[120,80,90]
      name=['p','g','b']
      df=pd.DataFrame(list(zip(name,your,requir)),columns=['Nutritian','Yours','Required'])


      st.text(df)

      result=liver.predict(pred_list1)
      prob=liver.predict_proba(pred_list1)
      if result[0]==0:
         st.header(f'You are health with chance of Liver{prob[0][0]}')
      else:
         st.header(f'You Have aa diseas with probability Liver{prob[0][1]}')

      st.header(result[0])
      st.header(prob)
