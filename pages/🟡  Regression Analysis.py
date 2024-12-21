import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from streamlit_extras.metric_cards import style_metric_cards


#navicon and header
st.set_page_config(page_title="Dashboard", page_icon="🟡", layout="wide")  

st.header("Predictive Analytics Dashboard")
st.image("data/logo2.webp",caption="")
st.write("Multiple Regression With  SSE, SE, SSR, SST, R2, ADJ[R2], Residual")
st.success("The main objective is to measure if Number of family dependents and Wives may influence a person to supervise many projects")
 
# load CSS Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#logo

st.sidebar.title("Predict New Values")

df = pd.read_excel('regression.xlsx')
X = df[['Dependant', 'Wives']]
Y = df['Projects']


# Fit a linear regression model
model = LinearRegression()
model.fit(X, Y)

# Make predictions
predictions = model.predict(X)

# Predictions on the same data
y_pred = model.predict(X)

#Regression coefficients (Bo, B1, B2)
intercept = model.intercept_ #Bo
coefficients = model.coef_ #B1, B2

# Calculate R-squared Coefficient of determination
r2 = r2_score(Y, predictions)

# Calculate Adjusted R-squared
n = len(Y)
p = X.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Calculate Sum Squared Error (SSE) and SSR
sse = np.sum((Y - predictions)**2)
ssr = np.sum((y_pred - np.mean(Y)) ** 2)


#regression line
with st.expander("Regression Coefficient Equestion Output"):
 col1,col2,col3=st.columns(3)
 col1.metric('INTERCEPT:',value= f'{intercept:.4f}',delta="(Bo)")
 col2.metric('B1 COEFFICIENT:',value= f'{coefficients[0]:.4f}',delta=" for X1 number of Dependant (B1)")
 col3.metric('B2 COEFFICIENT',value= f'{coefficients[1]:.4f}',delta=" for X2 number of Wives (B2):")
 style_metric_cards(background_color="#1f252b", border_left_color="#d16f0d", border_color="#d16f0d", box_shadow="#F71938")


# Print R-squared, Adjusted R-squared, and SSE
with st.expander("Measure Of Variations"):
 col1,col2,col3=st.columns(3)
 
 col1.metric('R-SQUARED:',value= f'{r2:.4f}',delta="Coefficient of Determination")
 col2.metric('ADJUSTED R-SQUARED:',value= f'{adjusted_r2:.4f}',delta="Adj[R2]")
 col3.metric('SUM SQUARED ERROR (SSE):',value= f'{sse:.4f}',delta="Squared(Y-Y_pred)")
 style_metric_cards(background_color="#1f252b", border_left_color="#d16f0d", border_color="#d16f0d", box_shadow="#F71938")

 # Print a table with predicted Y
with st.expander("Prediction Table"):
 result_df = pd.DataFrame({'Name':df['Name'],'No of Dependant':df['Dependant'], 'No of Wives': df['Wives'], 'Done Projects | Actual Y': Y, 'Y_predicted': predictions})
 # Add SSE and SSR to the DataFrame
 result_df['SSE'] = sse
 result_df['SSR'] = ssr
 st.dataframe(result_df,use_container_width=True)

 #download predicted csv
 df_download = result_df.to_csv(index=False).encode('utf-8')
 st.download_button(
    label="Download Predicted Dataset",
    data=df_download,
    key="download_dataframe.csv",
    file_name="my_dataframe.csv"
 )

with st.expander("Residual & Line Of Best Fit"):
 # Calculate residuals
 residuals = Y - predictions
 # Create a new DataFrame to store residuals
 residuals_df = pd.DataFrame({'Actual': Y, 'Predicted': predictions, 'Residuals': residuals})
 # Print the residuals DataFrame
 st.dataframe(residuals_df,use_container_width=True)

 col1, col2=st.columns(2)
 # Left plot: Residual vs Predicted
 with col1:
    fig, ax = plt.subplots(figsize=(8, 6))  
    fig.patch.set_facecolor('#1f252b')
    font = font_manager.FontProperties(family='Times New Roman', weight='bold', size='12')
    ax.scatter(Y, predictions, color='yellow')
    ax.plot([min(Y), max(Y)], [min(Y), max(Y)], '--k', color='red', label='Best Fit Line')
    ax.set_xlabel('Actual Y | number of Projects', fontproperties=font, color='#66d19b')
    ax.set_ylabel('Predicted Y', fontproperties=font, color='#66d19b')
    ax.set_title('Residual & Line Of Best Fit', fontproperties=font, color='#f5dd5d', pad=20)
    ax.set_facecolor('#1f252b')
    ax.spines['bottom'].set_color('#a3a828')
    ax.spines['top'].set_color('#a3a828')
    ax.spines['left'].set_color('#a3a828')
    ax.spines['right'].set_color('#a3a828')
    ax.grid(True, color='#3a3d4f', linestyle='-', linewidth=0.5)
    ax.legend(prop=font, facecolor='#f5dd5d', edgecolor='none', loc='lower right')
    
    # Set tick labels font and color
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)
        tick.set_color('#eef26f')
        
    st.pyplot(fig)

 # Right plot: KDE plot of residuals
 with col2:
    # Create the seaborn displot
    fig, ax = plt.subplots(figsize=(8, 6)) 
    sns.kdeplot(residuals, ax=ax, color='blue', fill=True)
    sns.set_style("whitegrid")  # Set the style to whitegrid 
    fig.patch.set_facecolor('#1f252b')
    font = font_manager.FontProperties(family='Times New Roman', weight='bold', size='12')
    ax.set_xlabel('Projects', fontproperties=font, color='#66d19b')
    ax.set_ylabel('Density', fontproperties=font, color='#66d19b')
    ax.set_title('KDE', fontproperties=font, color='#f5dd5d', pad=20)
    ax.set_facecolor('#1f252b')
    ax.spines['bottom'].set_color('#a3a828')
    ax.spines['top'].set_color('#a3a828')
    ax.spines['left'].set_color('#a3a828')
    ax.spines['right'].set_color('#a3a828')
    ax.grid(True, color='#3a3d4f', linestyle='-', linewidth=0.5)

    # Set tick labels font and color
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(font)
        tick.set_color('#eef26f')

       
    # Display the plot in Streamlit
    st.pyplot(fig)

 

# User input for X1 and X2
with st.sidebar:
 with st.form("input_form",clear_on_submit=True):
  x1 = st.number_input("Enter Dependant",)
  x2 = st.number_input("Number of Wives",)
  submit_button = st.form_submit_button(label="Predict")
  
  
if submit_button:
 # Make predictions
 new_data = np.array([[x1, x2]])
 new_prediction = model.predict(new_data)
 # Display prediction
 with st.expander("New Incomming Data Prediction"):
  st.write(f"<span style='font-size: 34px;color:black;'>Predicted Output: </span> <span style='font-size: 34px;'> {new_prediction}</span>", unsafe_allow_html=True)
 
st.sidebar.image("data/logo3.png",caption="Online Analytics")
















 
