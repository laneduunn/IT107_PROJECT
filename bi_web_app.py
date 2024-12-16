import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# App Title
st.title("Plotlydict Business Intelligence Web Application")

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

# Navigation Sidebar
st.sidebar.title("Navigation")
pages = ["Data Visualization", "Data Mining"]
selected_page = st.sidebar.radio("Choose a Module", pages)

# Page 1: Data Visualization
if selected_page == "Data Visualization":
    st.header("Interactive Sales Analytics Dashboard")

    query = """
    SELECT 
        sf.sales_id,
        sf.quantity,
        sf.total_price,
        t.invoice_date,
        t.year,
        t.month,
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

    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales ($)", f"${data['total_price'].sum():,.2f}")
    with col2:
        st.metric("Total Customers", data["customer_id"].nunique())
    with col3:
        st.metric("Unique Products Sold", data["description"].nunique())

    # Modern Visualizations
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

# Page 2: Data Mining
elif selected_page == "Data Mining":
    st.header("Data Mining: Customer Segmentation & Sales Forecasting")

    tabs = st.tabs(["Customer Segmentation", "Sales Forecasting"])

    # Tab 1: Customer Segmentation
    with tabs[0]:
        st.subheader("Customer Segmentation")

        query = """
        SELECT 
            sf.customer_id AS CustomerID,
            SUM(sf.total_price) AS TotalSpend,
            COUNT(sf.sales_id) AS PurchaseFrequency,
            MAX(t.invoice_date) AS LastPurchaseDate
        FROM sales_fact sf
        JOIN time_dim t ON sf.time_key = t.time_key
        GROUP BY sf.customer_id;
        """
        customer_data = load_data(query)

        customer_data.columns = customer_data.columns.str.lower()
        customer_data["lastpurchasedate"] = pd.to_datetime(customer_data["lastpurchasedate"])
        current_date = customer_data["lastpurchasedate"].max()
        customer_data["recency"] = (current_date - customer_data["lastpurchasedate"]).dt.days

        features = customer_data[["totalspend", "purchasefrequency", "recency"]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=4, random_state=42)
        customer_data["cluster"] = kmeans.fit_predict(scaled_features)

        st.subheader("Cluster Summary")
        cluster_summary = customer_data.groupby("cluster")[["totalspend", "purchasefrequency", "recency"]].mean()
        st.dataframe(cluster_summary)

        # Cluster Visualization (Simplified)
        st.subheader("Cluster Visualization")
        st.write("This chart shows customer clusters based on two key features.")
        x_axis = st.selectbox("Select X-axis Feature", ["totalspend", "purchasefrequency", "recency"])
        y_axis = st.selectbox("Select Y-axis Feature", ["recency", "totalspend", "purchasefrequency"], index=1)

        fig_cluster = px.scatter(
            customer_data,
            x=x_axis,
            y=y_axis,
            color="cluster",
            title="Customer Clusters",
            labels={x_axis: x_axis.capitalize(), y_axis: y_axis.capitalize()},
            hover_data=["totalspend", "purchasefrequency", "recency"]
        )
        st.plotly_chart(fig_cluster)

    # Tab 2: Sales Forecasting
    with tabs[1]:
        st.subheader("Sales Forecasting")

        query = """
        SELECT t.invoice_date AS InvoiceDate, SUM(sf.total_price) AS TotalPrice
        FROM sales_fact sf
        JOIN time_dim t ON sf.time_key = t.time_key
        GROUP BY t.invoice_date
        ORDER BY t.invoice_date;
        """
        sales_data = load_data(query)

        sales_data.columns = sales_data.columns.str.lower()
        sales_data["invoicedate"] = pd.to_datetime(sales_data["invoicedate"])
        monthly_sales = sales_data.groupby(sales_data["invoicedate"].dt.to_period("M"))["totalprice"].sum().reset_index()
        monthly_sales.columns = ["Month", "Sales"]
        monthly_sales["MonthIndex"] = range(1, len(monthly_sales) + 1)

        X = monthly_sales[["MonthIndex"]]
        y = monthly_sales["Sales"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse:.2f}")

        # Visualization: Actual vs Predicted
        fig_actual_pred = go.Figure()
        fig_actual_pred.add_trace(go.Scatter(x=X_test["MonthIndex"], y=y_test, mode="markers", name="Actual"))
        fig_actual_pred.add_trace(go.Scatter(x=X_test["MonthIndex"], y=y_pred, mode="lines", name="Predicted"))
        fig_actual_pred.update_layout(title="Actual vs Predicted Sales", xaxis_title="Month Index", yaxis_title="Sales")
        st.plotly_chart(fig_actual_pred)

        # Visualization: Future Sales Forecast
        future_months = pd.DataFrame({"MonthIndex": range(len(monthly_sales) + 1, len(monthly_sales) + 13)})
        future_sales = model.predict(future_months)

        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=monthly_sales["MonthIndex"], y=monthly_sales["Sales"], mode="lines", name="Historical Sales"))
        fig_forecast.add_trace(go.Scatter(x=future_months["MonthIndex"], y=future_sales, mode="lines", name="Forecasted Sales"))
        fig_forecast.update_layout(title="Future Sales Forecast", xaxis_title="Month Index", yaxis_title="Sales")
        st.plotly_chart(fig_forecast)
