# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle

st.set_page_config(
    page_title="Car Decision Helper",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_car_data2.csv')
    data['Is_Diesel']        = (data['Fuel_Type'] == 'Diesel').astype(int)
    data['Is_Automatic']     = (data['Transmission'] == 'Automatic').astype(int)
    data['Is_Dealer']        = (data['Selling_type'] == 'Dealer').astype(int)
    data['Is_First_Owner']   = (data['Owner'] == 0).astype(int)
    data['Car_Age']          = 2025 - data['Year']
    data['km_Per_year']      = data['Driven_kms'] / data['Car_Age']
    data['Transmission_Numeric'] = (data['Transmission'] == 'Manual').astype(int)

    return data

@st.cache_resource
def load_model():
    with open('xgb_car_price_model2.pkl', 'rb') as f:
        return pickle.load(f)

data  = load_data()
model = load_model()

# Sidebar Navigation
with st.sidebar:
    st.title("🚗 Car Decision Helper")
    st.markdown("---")
    page = st.radio("", [
        "🔍 Factor Analysis", 
        "🚗 Car Explore",
        "🎯 Car Decision Helper"
    ])

if page == "🔍 Factor Analysis":
    st.title("📊 Market Overview")
    
    tab1, tab2, tab3, tab4 , tab5 , tab6 , tab7= st.tabs([
        "📅 Age Group",
        "📈 Price Distribution", 
        "Transimission" , 
        "⛽ Fuel Type",
        "🏪 Seller Type" , 
        " 👤 Owner " , 
        " 🚗 Drive KM "
    ])
    
    # ── TAB 1: Age Group ──
    with tab1:
        st.subheader("Age Group Analysis")
        
        age_group_sales = (data.groupby('Age_Group')['Selling_Price']
                        .mean().reset_index())
        age_group_sales.columns = ['Age Group', 'Average Selling Price']
        
        fig = px.bar(age_group_sales,
                    x='Age Group', y='Average Selling Price',
                    title='Average Selling Price by Age Group',
                    color='Average Selling Price',
                    color_continuous_scale='Reds',
                    text='Average Selling Price',
                    template='plotly_dark', height=500)
        fig.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.box(data, x='Age_Group', y='Selling_Price',
                        color='Age_Group',
                        title='Price Distribution by Age Group',
                        template='plotly_dark', height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:  
            fig3 = px.box(data, x='Age_Group', y='Depreciation_Rate',
                        color='Age_Group',
                        title='Depreciation Rate by Age Group',
                        template='plotly_dark', height=400)
            st.plotly_chart(fig3, use_container_width=True)  

    # ── TAB 2: Price Distribution ──
    with tab2:
        st.subheader("Price Distribution")
        
        car_counts = data['Car_Name'].value_counts().reset_index()
        car_counts.columns = ['Car_Name', 'Count']
        
        st.markdown("#### Most Sold Cars")
        fig = px.bar(car_counts.head(10),        
                    x='Car_Name', y='Count',
                    title='Top 10 Most Sold Cars',
                    color='Count',
                    color_continuous_scale='Blues',
                    text='Count',
                    template='plotly_dark', height=450)
        fig.update_traces(textposition='outside', texttemplate='%{y:.0f}')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(car_counts.head(10)) 

        st.markdown("#### Least Sold Cars")
        fig2 = px.bar(car_counts.tail(10),      
                    x='Car_Name', y='Count',
                    title='Bottom 10 Least Sold Cars',
                    color='Count',
                    color_continuous_scale='Reds',
                    text='Count',
                    template='plotly_dark', height=450)
        fig2.update_traces(textposition='outside', texttemplate='%{y:.0f}')
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(car_counts.tail(10)) 

        st.markdown("#### Avg Selling Price per Car")
        car_avg = (data.groupby('Car_Name')['Selling_Price']
                .mean().reset_index()
                .sort_values('Selling_Price', ascending=False)
                .head(10))
        car_avg.columns = ['Car_Name', 'Avg_Price']
        fig3 = px.bar(car_avg,                   
                    x='Car_Name', y='Avg_Price',
                    title='Top 10 Cars by Avg Selling Price',
                    color='Avg_Price',
                    color_continuous_scale='Reds',
                    text='Avg_Price',
                    template='plotly_dark', height=450)
        fig3.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
        fig3.update_xaxes(tickangle=45)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(car_avg.head(10)) 

    with tab3:
        st.subheader("Transmission Analysis")

        st.markdown("#### Automatic vs Manual — Avg Selling Price")
        avg_trans = data.groupby('Transmission')['Selling_Price'].mean().reset_index()
        avg_trans.columns = ['Transmission', 'Avg_Selling_Price']
        fig = px.bar(avg_trans,
                    x='Transmission', y='Avg_Selling_Price',
                    color='Transmission',
                    title='Avg Selling Price by Transmission',
                    template='plotly_dark', height=450)   
        fig.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(avg_trans) 

        st.markdown("#### Price Distribution by Transmission")
        fig2 = px.box(data,                    
                    x='Transmission', y='Selling_Price',
                    color='Transmission',
                    title='Price Distribution by Transmission Type',
                    template='plotly_dark', height=450)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(data[['Transmission', 'Selling_Price']].groupby('Transmission').describe().reset_index()) 

        st.markdown("#### Transmission Market Share")
        trans_counts = data['Transmission'].value_counts().reset_index()
        trans_counts.columns = ['Transmission', 'Count']
        fig3 = px.pie(trans_counts,
                    names='Transmission',      
                    values='Count',            
                    title='Transmission Type Distribution',
                    template='plotly_dark', height=450)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(trans_counts) 
    
    with tab4: 
        st.subheader("Fuel Type Analysis") 

        st.markdown("#### Avg Selling Price by Fuel Type") 
        avg_fuel = data.groupby('Fuel_Type')['Selling_Price'].mean().reset_index()
        avg_fuel.columns = ['Fuel_Type', 'Avg_Selling_Price']
        fig = px.bar(avg_fuel,
                    x='Fuel_Type', y='Avg_Selling_Price',
                    color='Fuel_Type',
                    title='Avg Selling Price by Fuel Type',
                    template='plotly_dark', height=450)
        fig.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(avg_fuel)

        st.markdown("#### Price Distribution by Fuel Type")
        fig2 = px.box(data,
                    x='Fuel_Type', y='Selling_Price',
                    color='Fuel_Type',
                    title='Price Distribution by Fuel Type',
                    template='plotly_dark', height=450)
        st.plotly_chart(fig2, use_container_width=True) 
        st.dataframe(data[['Fuel_Type', 'Selling_Price']].groupby('Fuel_Type').describe().reset_index()) 
        
        st.markdown("#### Fuel Type Market Share")
        fuel_counts = data['Fuel_Type'].value_counts().reset_index() 
        fig = px.pie(fuel_counts, 
                    names='Fuel_Type', 
                    values='count', 
                    title='Fuel Type Distribution',
                    template='plotly_dark', height=450) 
        st.plotly_chart(fig, use_container_width=True) 
        st.dataframe(fuel_counts) 

        st.markdown("#### Avg Fuel Type Depreciation Rate") 
        avg_depr = data.groupby('Fuel_Type')['Depreciation_Rate'].mean().reset_index() 
        avg_depr.columns = ['Fuel_Type', 'Avg_Depreciation_Rate'] 
        fig2 = px.bar(avg_depr, 
                    x='Fuel_Type', y='Avg_Depreciation_Rate', 
                    color='Fuel_Type', 
                    title='Avg Depreciation Rate by Fuel Type',
                    template='plotly_dark', height=450) 
        fig2.update_traces(textposition='outside', texttemplate='%{y:.2f}%')
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(avg_depr)
        
    with tab5:
        st.subheader("Seller Type Analysis")

        # 1. Car Count per Seller
        st.markdown("#### How Many Cars per Seller Type?")
        sell_counts = data['Selling_type'].value_counts().reset_index()
        sell_counts.columns = ['Selling_type', 'Count']
        fig = px.pie(sell_counts,
                    names='Selling_type', values='Count',
                    title='Market Share by Seller Type',
                    template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)

        # 2. Avg Price
        with col1:
            avg_sell_price = data.groupby('Selling_type')['Selling_Price'].mean().reset_index()
            avg_sell_price.columns = ['Selling_type', 'Avg_Selling_Price']
            fig2 = px.bar(avg_sell_price,           
                        x='Selling_type', y='Avg_Selling_Price',
                        color='Selling_type',
                        title='Avg Selling Price by Seller Type',
                        template='plotly_dark', height=400)
            fig2.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
            st.plotly_chart(fig2, use_container_width=True)

        # 3. Avg Car Age
        with col2:
            avg_age_sell = data.groupby('Selling_type')['Car_Age'].mean().reset_index()
            avg_age_sell.columns = ['Selling_type', 'Avg_Car_Age']
            fig3 = px.bar(avg_age_sell,             
                        x='Selling_type', y='Avg_Car_Age',
                        color='Selling_type',
                        title='Avg Car Age by Seller Type',
                        template='plotly_dark', height=400)
            fig3.update_traces(textposition='outside', texttemplate='%{y:.1f} yrs')
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Price Distribution by Seller Type")
        fig4 = px.box(data,                         
                    x='Selling_type', y='Selling_Price',
                    color='Selling_type',
                    title='Price Distribution — Dealer vs Individual',
                    template='plotly_dark', height=450)
        st.plotly_chart(fig4, use_container_width=True)

        # 5. Depreciation
        st.markdown("#### Avg Depreciation Rate by Seller Type")
        avg_dep_sell = data.groupby('Selling_type')['Depreciation_Rate'].mean().reset_index()
        avg_dep_sell.columns = ['Selling_type', 'Avg_Depreciation_Rate']
        fig5 = px.bar(avg_dep_sell,
                    x='Selling_type', y='Avg_Depreciation_Rate',
                    color='Selling_type',
                    title='Avg Depreciation Rate by Seller Type',
                    template='plotly_dark', height=400)
        fig5.update_traces(textposition='outside', texttemplate='%{y:.2f}%')
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(avg_dep_sell) 

    with tab6:
        st.subheader("Owner Type Analysis")

        owner_map = {0: 'First Owner', 1: 'Second Owner', 3: 'Third or More'}
        data_owner = data.copy()
        data_owner['Owner_Label'] = data_owner['Owner'].map(owner_map).fillna('Other')

        st.markdown("#### Market Share by Owner Type")
        owner_counts = data_owner['Owner_Label'].value_counts().reset_index()
        owner_counts.columns = ['Owner', 'Count']
        fig = px.pie(owner_counts,
                    names='Owner', values='Count',
                    title='Market Share by Owner Type',
                    template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            avg_owner_price = data_owner.groupby('Owner_Label')['Selling_Price'].mean().reset_index()
            avg_owner_price.columns = ['Owner', 'Avg_Selling_Price']
            fig2 = px.bar(avg_owner_price,
                        x='Owner', y='Avg_Selling_Price',
                        color='Owner',
                        title='Avg Selling Price by Owner Type',
                        template='plotly_dark', height=400)
            fig2.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            avg_dep_owner = data_owner.groupby('Owner_Label')['Depreciation_Rate'].mean().reset_index()
            avg_dep_owner.columns = ['Owner', 'Avg_Depreciation_Rate']
            fig3 = px.bar(avg_dep_owner,
                        x='Owner', y='Avg_Depreciation_Rate',
                        color='Owner',
                        title='Avg Depreciation Rate by Owner Type',
                        template='plotly_dark', height=400)
            fig3.update_traces(textposition='outside', texttemplate='%{y:.1f}%')
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Price Distribution by Owner Type")
        fig4 = px.box(data_owner,
                    x='Owner_Label', y='Selling_Price',
                    color='Owner_Label',
                    title='Price Distribution by Owner Type',
                    template='plotly_dark', height=450)
        st.plotly_chart(fig4, use_container_width=True)

    with tab7:
        st.subheader("Kilometers Driven Analysis")

        def km_range(k):
            if k <= 50000:    return 'Low (0-50k)'
            elif k <= 100000: return 'Medium (50k-100k)'
            else:             return 'High (100k+)'

        data_km = data.copy()
        data_km['KM_Range'] = data_km['Driven_kms'].apply(km_range)

        st.markdown("#### Market Share by KM Range")
        km_counts = data_km['KM_Range'].value_counts().reset_index()
        km_counts.columns = ['KM_Range', 'Count']
        fig = px.pie(km_counts,
                    names='KM_Range', values='Count',
                    title='Market Share by KM Range',
                    template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            avg_km_price = data_km.groupby('KM_Range')['Selling_Price'].mean().reset_index()
            avg_km_price.columns = ['KM_Range', 'Avg_Selling_Price']
            fig2 = px.bar(avg_km_price,
                        x='KM_Range', y='Avg_Selling_Price',
                        color='KM_Range',
                        title='Avg Selling Price by KM Range',
                        template='plotly_dark', height=400)
            fig2.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            avg_km_dep = data_km.groupby('KM_Range')['Depreciation_Rate'].mean().reset_index()
            avg_km_dep.columns = ['KM_Range', 'Avg_Depreciation_Rate']
            fig3 = px.bar(avg_km_dep,
                        x='KM_Range', y='Avg_Depreciation_Rate',
                        color='KM_Range',
                        title='Avg Depreciation Rate by KM Range',
                        template='plotly_dark', height=400)
            fig3.update_traces(textposition='outside', texttemplate='%{y:.1f}%')
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("#### Price Distribution by KM Range")
        fig4 = px.box(data_km,
                    x='KM_Range', y='Selling_Price',
                    color='KM_Range',
                    title='Price Distribution by KM Range',
                    template='plotly_dark', height=450)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("#### Scatter: KMs vs Selling Price")
        fig5 = px.scatter(data_km,
                        x='Driven_kms', y='Selling_Price',
                        color='KM_Range',
                        title='KMs Driven vs Selling Price',
                        template='plotly_dark', height=450,
                        trendline='ols')
        st.plotly_chart(fig5, use_container_width=True)

elif page == "🚗 Car Explore":
    st.title("🚗 Car Explorer")
    st.markdown("Filter Cars by your preferences and explore the market")
    st.markdown("---")

    car_name = st.selectbox("Choose car", sorted(data['Car_Name'].unique()))
    car_data = data[data['Car_Name'] == car_name]
    st.markdown("---")

    # ── Summary Cards ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Avg Price",        f"{car_data['Selling_Price'].mean():.2f}L")
    col2.metric("🛣️ Avg KMs",          f"{car_data['Driven_kms'].mean():,.0f}")
    col3.metric("📅 Avg Age",          f"{car_data['Car_Age'].mean():.1f} yrs")
    col4.metric("📉 Avg Depreciation", f"{car_data['Depreciation_Rate'].mean():.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(data, x='Selling_Price',
                            title=f'{car_name} Price vs Market',
                            template='plotly_dark', height=400,
                            color_discrete_sequence=['#3498db'])
        fig1.add_vline(x=car_data['Selling_Price'].mean(),
                    line_dash='dash', line_color='red',
                    annotation_text=f'{car_name}: {car_data["Selling_Price"].mean():.2f}L',
                    annotation_font_color='red')
        fig1.add_vline(x=data['Selling_Price'].mean(),
                    line_dash='dash', line_color='green',
                    annotation_text=f'Market Avg: {data["Selling_Price"].mean():.2f}L',
                    annotation_font_color='green')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(data, x='Depreciation_Rate',
                            title=f'{car_name} Depreciation vs Market',
                            template='plotly_dark', height=400,
                            color_discrete_sequence=['#e74c3c'])
        fig2.add_vline(x=car_data['Depreciation_Rate'].mean(),
                    line_dash='dash', line_color='red',
                    annotation_text=f'{car_name}: {car_data["Depreciation_Rate"].mean():.1f}%',
                    annotation_font_color='red')
        fig2.add_vline(x=data['Depreciation_Rate'].mean(),
                    line_dash='dash', line_color='green',
                    annotation_text=f'Market Avg: {data["Depreciation_Rate"].mean():.1f}%',
                    annotation_font_color='green')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fuel_counts = car_data['Fuel_Type'].value_counts().reset_index()
        fuel_counts.columns = ['Fuel_Type', 'Count']
        fig3 = px.pie(fuel_counts,
                    names='Fuel_Type', values='Count',
                    title=f'Fuel Type Distribution - {car_name}',
                    template='plotly_dark', height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        trans_counts = car_data['Transmission'].value_counts().reset_index()  
        trans_counts.columns = ['Transmission', 'Count']
        fig4 = px.pie(trans_counts,
                    names='Transmission',   
                    values='Count',
                    title=f'Transmission Distribution - {car_name}',
                    template='plotly_dark', height=350)
        st.plotly_chart(fig4, use_container_width=True)

    if len(car_data) > 1:
        fig5 = px.scatter(car_data,          
                        x='Driven_kms', y='Selling_Price',
                        color='Fuel_Type',  
                        title=f'KMs vs Price - {car_name}',
                        template='plotly_dark', height=400,
                        trendline='ols' if len(car_data) > 2 else None)  
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        owner_price = car_data.groupby('Owner_Group')['Selling_Price'].mean().reset_index()
        fig6 = px.bar(owner_price,
                    x='Owner_Group', y='Selling_Price',
                    color='Owner_Group',
                    title=f'Owner vs Selling Price - {car_name}',
                    template='plotly_dark', height=400,
                    text='Selling_Price')
        fig6.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
        st.plotly_chart(fig6, use_container_width=True)

    with col2:
        selling_price = car_data.groupby('Selling_type')['Selling_Price'].mean().reset_index()
        fig7 = px.bar(selling_price,
                    x='Selling_type', y='Selling_Price',
                    color='Selling_type',   
                    title=f'Avg Price by Seller Type - {car_name}',
                    template='plotly_dark', height=400,
                    text='Selling_Price')
        fig7.update_traces(textposition='outside', texttemplate='%{y:.2f}L')
        st.plotly_chart(fig7, use_container_width=True)

    # ── Insight ──
    st.markdown("---")
    avg_price  = car_data['Selling_Price'].mean()
    market_avg = data['Selling_Price'].mean()
    avg_dep    = car_data['Depreciation_Rate'].mean()
    market_dep = data['Depreciation_Rate'].mean()
    price_diff = ((avg_price / market_avg) - 1) * 100
    dep_diff   = avg_dep - market_dep

    price_msg = (f"⬆️ Higher than market avg by {price_diff:.1f}%" if price_diff > 10
                else f"⬇️ Lower than market avg by {abs(price_diff):.1f}% — Good Deal!" if price_diff < -10
                else "➡️ Close to market average")

    dep_msg = (f"✅ Better depreciation than market by {abs(dep_diff):.1f}%" if dep_diff < 0
            else f"⚠️ Worse depreciation than market by {dep_diff:.1f}%")

    st.info(f"""
    💡 **{car_name} Analysis:**
    - {price_msg}
    - {dep_msg}
    - Listings in data: {len(car_data)} cars
    """)

    # ── Full Table ──
    st.markdown("---")
    st.subheader(f"📋 All {car_name} Listings ({len(car_data)})")
    st.dataframe(
        car_data[['Year', 'Selling_Price', 'Present_Price',
                'Driven_kms', 'Depreciation_Rate', 'Fuel_Type',
                'Selling_type', 'Transmission', 'Owner_Group', 'KM_Category']]
        .sort_values('Selling_Price')
        .reset_index(drop=True),
        use_container_width=True)


elif page == "🎯 Car Decision Helper":
    st.title("🎯 Car Decision Helper")
    st.markdown("Enter your car details to get a price prediction")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        car_age    = st.slider("Car Age (years)", 1, 25, 5)
        driven_kms = st.number_input("Driven KMs", 500, 500000, 30000, step=1000)
        owner      = st.selectbox("Owner Type", [0, 1, 3],
                                format_func=lambda x: {
                                    0: 'First Owner',
                                    1: 'Second Owner',
                                    3: 'Third or More'
                                }[x])

    with col2:
        transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
        fuel         = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
        seller       = st.selectbox("Seller Type", ['Dealer', 'Individual'])

    with col3:
        st.markdown("#### Summary")
        st.write(f"🚗 Car Age: **{car_age} years**")
        st.write(f"🛣️ KMs: **{driven_kms:,}**")
        st.write(f"⛽ Fuel: **{fuel}**")
        st.write(f"⚙️ Transmission: **{transmission}**")
        st.write(f"🏪 Seller: **{seller}**")

    st.markdown("---")

    if st.button("🔍 Predict Price", type='primary'):
        km_per_year          = driven_kms / car_age
        is_first_owner       = 1 if owner == 0 else 0
        is_automatic         = 1 if transmission == 'Automatic' else 0
        is_dealer            = 1 if seller == 'Dealer' else 0
        transmission_numeric = 0 if transmission == 'Automatic' else 1
        is_diesel            = 1 if fuel == 'Diesel' else 0

        input_data = pd.DataFrame([{
            'Driven_kms':           driven_kms,
            'Car_Age':              car_age,
            'Owner':                owner,
            'km_Per_year':          km_per_year,
            'Is_First_Owner':       is_first_owner,
            'Is_Automatic':         is_automatic,
            'Is_Dealer':            is_dealer,
            'Transmission_Numeric': transmission_numeric,
            'Is_Diesel':            is_diesel
        }])

        predicted_price = model.predict(input_data)[0]

        predicted_price = max(predicted_price, 0.1)

        st.success(f"💰 Estimated Selling Price: **{predicted_price:.2f}L**")