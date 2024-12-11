import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# App Title
st.title("Data Visualization Dashboard")

# Database Connection
@st.cache_resource
def connect_to_database():
    db_url = "postgresql+psycopg2://postgres:admin@localhost/retail_db"
    engine = create_engine(db_url)
    return engine

engine = connect_to_database()

# Load Data from Database
@st.cache_data
def load_data(query):
    return pd.read_sql(query, engine)

# Query to Load Data
query = """
SELECT 
    sf.sales_id,
    sf.quantity,
    sf.total_price,
    t.invoice_date,
    c.customer_id,
    co.country_name,
    p.description
FROM sales_fact sf
JOIN time_dim t ON sf.time_key = t.time_key
JOIN customer_dim c ON sf.customer_id = c.customer_id
JOIN country_dim co ON c.country_id = co.country_id
JOIN product_dim p ON sf.product_id = p.product_id;
"""
data = load_data(query)

# Sidebar Filters
st.sidebar.header("Filter Options")
data["invoice_date"] = pd.to_datetime(data["invoice_date"])

# Date Range Filter
start_date, end_date = st.sidebar.date_input(
    "Select Date Range",
    [data["invoice_date"].min().date(), data["invoice_date"].max().date()]
)
if start_date and end_date:
    data = data[(data["invoice_date"] >= pd.Timestamp(start_date)) & (data["invoice_date"] <= pd.Timestamp(end_date))]

# Country Filter
country = st.sidebar.selectbox("Select Country", ["All"] + list(data["country_name"].unique()))
if country != "All":
    data = data[data["country_name"] == country]

# Product Filter
product = st.sidebar.selectbox("Select Product", ["All"] + list(data["description"].unique()))
if product != "All":
    data = data[data["description"] == product]

# Display Filter Summary
st.markdown(f"""
**Filter Summary**
- Date Range: {start_date} to {end_date}
- Country: {country}
- Product: {product}
""")

# Key Metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Sales ($)", f"${data['total_price'].sum():,.2f}")
with col2:
    st.metric("Total Customers", data["customer_id"].nunique())
with col3:
    st.metric("Unique Products Sold", data["description"].nunique())

# Visualizations
st.subheader("Visualizations")

# 1. Top-Selling Products
st.markdown("### Top-Selling Products")
top_products = data.groupby("description")["quantity"].sum().nlargest(10).reset_index()
fig1 = px.bar(
    top_products, 
    x="description", 
    y="quantity", 
    title="Top-Selling Products",
    labels={"description": "Product", "quantity": "Units Sold"},
    color="quantity",
    color_continuous_scale="Viridis"
)
st.plotly_chart(fig1, use_container_width=True)

# 2. Sales by Region
st.markdown("### Sales by Region")
sales_by_country = data.groupby("country_name")["total_price"].sum().reset_index()
fig2 = px.pie(
    sales_by_country, 
    values="total_price", 
    names="country_name", 
    title="Sales Distribution by Region"
)
st.plotly_chart(fig2, use_container_width=True)

# 3. Monthly Sales Trends
st.markdown("### Monthly Sales Trends")
data["month"] = data["invoice_date"].dt.to_period("M").astype(str)
monthly_sales = data.groupby("month")["total_price"].sum().reset_index()
fig3 = px.line(
    monthly_sales, 
    x="month", 
    y="total_price", 
    title="Monthly Sales Trends",
    labels={"month": "Month", "total_price": "Sales ($)"}
)
st.plotly_chart(fig3, use_container_width=True)

# Download Filtered Data
st.markdown("### Download Filtered Data")
csv_data = data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Data as CSV",
    data=csv_data,
    file_name="filtered_sales_data.csv",
    mime="text/csv",
)
