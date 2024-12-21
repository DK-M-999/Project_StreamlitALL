import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns 
import altair as alt
from UI import *
from matplotlib import pyplot as plt
from datetime import date, timedelta
#pip install streamlit-extras
#https://pypi.org/project/streamlit-extras/
from streamlit_extras.dataframe_explorer import dataframe_explorer


#page layout
st.set_page_config(page_title="Sales", page_icon="🟡", layout="wide")

st.header("Sales Analytics KPI  &  Trends  | Descriptive Analytics")
st.write("Pick a date range from sidebar to view sales trends | the default date is today")

#streamlit theme=none
theme_plotly = None 

# load CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
    
#load user interface file
UI()

#load dataset
df = pd.read_csv('sales.csv')


#filter date to view sales
with st.sidebar:
 st.title("Select Date Range")
 start_date=st.sidebar.date_input("Start Date",date.today()-timedelta(days=365*4))

with st.sidebar:
 end_date=st.date_input(label="End Date")
st.error("Business Metrics between[ "+str(start_date)+"] and ["+str(end_date)+"]")

#compare date
df2 = df[(df['OrderDate'] >= str(start_date)) & (df['OrderDate'] <= str(end_date))]

#dataframe
with st.expander("Filter Excel Dataset"):
 filtered_df = dataframe_explorer(df2, case=False)
 st.dataframe(filtered_df, use_container_width=True)

b1, b2=st.columns(2)
with b1:  
 from add_data import *
 st.subheader('Add New Record to Excel File',)
 #this function is called from another file named 'add_data.py' file  
 add_data()

 #metric cards
 with b2:
    st.subheader('Dataset Metrics',)
    from streamlit_extras.metric_cards import style_metric_cards
    col1, col2, = st.columns(2)
    col1.metric(label="All Inventory Products ", value=df2.Product.count(), delta="Number of Items in stock")
    col2.metric(label="Sum of Product Price USD:", value= f"{df2.TotalPrice.sum():,.0f}",delta=df2.TotalPrice.median())
    
    col11, col22,col33, = st.columns(3)
    col11.metric(label="Maximum Price  USD:", value= f"{ df2.TotalPrice.max():,.0f}",delta="High Price")
    col22.metric(label="Minimum Price  USD:", value= f"{ df2.TotalPrice.min():,.0f}",delta="Low Price")
    col33.metric(label="Total Price Range  USD:", value= f"{ df2.TotalPrice.max()-df2.TotalPrice.min():,.0f}",delta="Annual Salary Range")
    #style the metric
    style_metric_cards(background_color="#1f252b", border_left_color="#d16f0d", border_color="#d16f0d", box_shadow="#F71938")

#dot Plot
a1,a2=st.columns(2)
with a1:
 st.subheader('Products & Total Price',)
 source = df2
 chart = alt.Chart(source).mark_circle().encode(
    x='Product',
    y='TotalPrice',
    color='Category',
 ).interactive()

# 自定義圖表的字型與顏色
 chart = chart.configure_title(
        fontSize=16,
        font='Times New Roman',
   
    ).configure_axis(
        labelFont='Times New Roman',
        labelFontSize=12,
        labelColor='#5564c9',
        titleFont='Times New Roman',
        titleFontSize=14,
        titleColor='#5564c9'
    ).configure_legend(
        labelFont='Times New Roman',
        labelFontSize=12,
        labelColor='#5564c9',
        titleFont='Times New Roman',
        titleFontSize=14,
        titleColor='#5564c9'
    )
 st.altair_chart(chart, theme="streamlit", use_container_width=True)


with a2:
 st.subheader('Products & Unit Price',)
 energy_source = pd.DataFrame({
    "Product": df2["Product"],
    "UnitPrice ($)":  df2["UnitPrice"],
    "Date": df2["OrderDate"]
    })
 
#bar Graph
 bar_chart = alt.Chart(energy_source).mark_bar().encode(
        x="month(Date):O",
        y="sum(UnitPrice ($)):Q",
        color="Product:N"
    )
 
# 自定義圖表的字型與顏色
 chart = chart.configure_title(
        fontSize=16,
        font='Times New Roman',
   
    ).configure_axis(
        labelFont='Times New Roman',
        labelFontSize=12,
        labelColor='#5564c9',
        titleFont='Times New Roman',
        titleFontSize=14,
        titleColor='#5564c9'
    ).configure_legend(
        labelFont='Times New Roman',
        labelFontSize=12,
        labelColor='#5564c9',
        titleFont='Times New Roman',
        titleFontSize=14,
        titleColor='#5564c9'
    )
 st.altair_chart(bar_chart, use_container_width=True,theme=theme_plotly)
 
 
 #select only numeric or number data
 #pip install pandas-select
 #https://pypi.org/project/pandas-select/
p1,p2=st.columns(2) 
with p1:
# Select features to display scatter plot
 st.subheader('Features by Frequency',)
 feature_x = st.selectbox('Select feature for x Qualitative data', df2.select_dtypes("object").columns)
 feature_y = st.selectbox('Select feature for y Quantitative Data', df2.select_dtypes("number").columns)

# Display scatter plot
 fig, ax = plt.subplots()
 sns.scatterplot(data=df2, x=feature_x, y=feature_y, hue=df2.Product, ax=ax)
 fig.patch.set_edgecolor('#5564c9')
 fig.patch.set_facecolor('#1f252b')
 ax.set_facecolor('#1f252b')
 ax.set_xlabel(feature_x, color='#5564c9')
 ax.set_ylabel(feature_y, color='#5564c9')

# Customize legend
 legend = ax.legend()
 legend.get_frame().set_facecolor('#1f252b')
 legend.get_frame().set_edgecolor('#1f252b')
 for text in legend.get_texts():
     text.set_color('#5564c9')

# 設定圖表其他文字樣式 (假設您已經有 fig 物件)
 for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
     item.set_fontfamily('Times New Roman')
     item.set_fontsize(12)
     item.set_color('#5564c9')

 st.pyplot(fig)

with p2:
 st.subheader('Products & Qantities',)
 source = pd.DataFrame({
        "Quantity ($)": df2["Quantity"],
        "Product": df2["Product"]
      })
 
 bar_chart = alt.Chart(source).mark_bar().encode(
        x="sum(Quantity ($)):Q",
        y=alt.Y("Product:N", sort="-x")
    )
# 設定圖表主題和樣式
 chart = bar_chart.properties(
        title=alt.Title('Products & Qantities',
                       anchor='middle',  # 標題置中
                       fontWeight='bold',  # 字體粗體
                       fontSize=16,
                       color='#5564c9'),
        width=600,  # 調整圖表寬度
        height=400,  # 調整圖表高度
    )

# 設定圖表背景和網格線
 chart = chart.configure(
        background='#1f252b',
        axisX=alt.Axis(grid=True, gridColor='#5564c9', gridOpacity=0.3),
        axisY=alt.Axis(grid=True, gridColor='#5564c9', gridOpacity=0.3)
    )

# 設定座標軸標題和文字
 chart = chart.configure_axis(
        labelColor='#5564c9',
        titleColor='#5564c9',
        titleFontSize=14,
        labelFontSize=12,
        labelFont='Times New Roman',
        titleFont='Times New Roman',
        titleFontWeight='bold'
    )

# 設定圖例標題和文字
 chart = chart.configure_legend(
        labelColor='#5564c9',
        titleColor='#5564c9',
        titleFontSize=14,
        labelFontSize=12,
        labelFont='Times New Roman',
        titleFontWeight='bold',
        orient='bottom',  # 圖例置於下方
    )

# 設定條形圖填充顏色
 chart = chart.mark_bar(color='#36366b')
 
 st.altair_chart(bar_chart, use_container_width=True,theme=theme_plotly)
st.sidebar.image("data/logo3.png")

 















 


