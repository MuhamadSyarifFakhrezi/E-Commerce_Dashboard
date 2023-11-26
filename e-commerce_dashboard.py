#!/usr/bin/env python
# coding: utf-8

# In[38]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency
from PIL import Image

sns.set(style='dark')


# # Menyiapkan DataFrame

# ### Helper Function

# #### create monthly_orders data

# In[2]:


def create_monthly_orders(df):
    monthly_orders = df.resample(rule='M', on='order_date').order_id.nunique()
    monthly_orders.index = monthly_orders.index.strftime('%B %Y') # mengubah format tanggal menjadi nama bulan
    monthly_orders = monthly_orders.reset_index()
    monthly_orders.rename(columns={'order_id': 'order_count'}, inplace=True)
    
    return monthly_orders


# #### Create weekly_orders data

# In[3]:


def create_weekly_orders(df):
    weekly_orders = df.resample(rule='W', on='order_date').order_id.nunique()
    weekly_orders.index = weekly_orders.index.strftime('%Y-%m-%d')
    weekly_orders = weekly_orders.reset_index()
    weekly_orders.rename(columns={'order_id': 'order_count'}, inplace=True)
    
    return weekly_orders    


# #### Create daily_orders data

# In[4]:


def create_daily_orders(df):
    daily_orders = df.resample(rule='D', on='order_date').order_id.nunique()
    daily_orders.index = daily_orders.index.strftime('%Y-%m-%d') # mengubah format tanggal menjadi nama bulan
    daily_orders = daily_orders.reset_index()
    daily_orders.rename(columns={'order_id': 'order_count'}, inplace=True)
    
    return daily_orders


# #### Create number_of_product_sold data

# In[5]:


def create_number_of_product_sold(df):
    number_of_product_sold = df.groupby(by='product_category_name').order_id.nunique().reset_index()
    number_of_product_sold.rename(columns={
        'order_id':'number_of_order'
    }, inplace=True)
    
    return number_of_product_sold


# #### Create byseller data

# In[6]:


def create_byseller(df):
    byseller = df.groupby('seller_id').order_id.nunique().reset_index()
    byseller.rename(columns={'order_id':'number_of_order'}, inplace=True)
    
    return byseller


# #### Create top_seller data

# In[7]:


def create_top_seller(df):
    top_seller = df.groupby('seller_id').agg({
        'review_score':'mean',
        'order_id':'nunique'
    }).reset_index()
    top_seller.rename(columns={
        'order_id':'number_of_order'
    }, inplace=True)

    # SCORING #
    # perangkingan
    top_seller['review_score_rank'] = top_seller['review_score'].rank(ascending=True)
    top_seller['number_of_order_rank'] = top_seller['number_of_order'].rank(ascending=True)
    
    # normalizing
    top_seller['review_score_rank_normalized'] = (top_seller['review_score_rank']/top_seller['review_score_rank'].max())*100
    top_seller['number_of_order_rank_normalized'] = (top_seller['number_of_order_rank']/top_seller['number_of_order_rank'].max())*100
    
    top_seller.drop(columns=['review_score_rank', 'number_of_order_rank'], inplace=True)
    
    # scoring
    top_seller['seller_score'] = (0.60*top_seller['review_score_rank_normalized']) + (0.40*top_seller['number_of_order_rank_normalized'])
    # komposisi yg digunakan 60:40
    
    top_seller = top_seller.round(2)

    # seller rank
    top_seller['rank'] = top_seller['seller_score'].rank(ascending=False)
    
    return top_seller


# #### Create seller_rank data

# In[8]:


def create_seller_rank(df): 
    seller_rank = df[['seller_id', 'review_score', 'number_of_order', 'rank']]
    seller_rank.rename(columns={'review_score': 'average_review_score', 'number_of_order':'number_of_sales'}, inplace=True)
    
    return seller_rank


# #### Create bycity data

# In[9]:


def create_bycity(df):
    bycity = df.groupby('customer_city').customer_unique_id.nunique().reset_index()
    bycity.rename(columns={
        'customer_unique_id':'customer_count'}, inplace=True)
    
    return bycity


# #### Create bystate data

# In[10]:


def create_bystate(df):
    bystate = df.groupby('customer_state').customer_unique_id.nunique().reset_index()
    bystate.rename(columns={
        'customer_unique_id':'customer_count'}, inplace=True)

    return bystate


# #### Create rfm data

# In[11]:


def create_rfm(df):
    rfm = df.groupby(by='customer_unique_id', as_index=False). agg({
        'order_date': 'max',
        'order_id': 'nunique',
        'total_payment': 'sum'
    })
    rfm.columns = ['customer_id', 'last_order', 'frequency', 'monetary']
    
    # recency diambil dari tanggal order terakhir masuk
    recent_date = df['order_date'].max() 
    rfm['recency'] = rfm['last_order'].apply(lambda x: (recent_date - x).days)
    
    rfm.drop('last_order', axis=1, inplace=True)

    # RFM SCORING #
    # perangkingan berdasarkan nilai RFM
    rfm['r_rank'] = rfm['recency'].rank(ascending=False)
    rfm['f_rank'] = rfm['frequency'].rank(ascending=True)
    rfm['m_rank'] = rfm['monetary'].rank(ascending=True)

    # normalizing nilai rank
    rfm['r_rank_normalized'] = (rfm['r_rank']/rfm['r_rank'].max())*100
    rfm['f_rank_normalized'] = (rfm['f_rank']/rfm['f_rank'].max())*100
    rfm['m_rank_normalized'] = (rfm['m_rank']/rfm['m_rank'].max())*100
    
    rfm.drop(columns=['r_rank', 'f_rank', 'm_rank'], inplace=True)

    # scoring
    rfm['rfm_score'] = (0.20*rfm['r_rank_normalized'])+(0.40 * rfm['f_rank_normalized'])+(0.40 * rfm['m_rank_normalized']) # komposisi 20:40:40
    rfm['rfm_score'] *= 0.05
    rfm = rfm.round(2)

    # CUSTOMER SEGMENTATION #
    rfm['customer_segment'] = np.where(
        rfm['rfm_score'] > 4.5, 'Top Customers', np.where(
        rfm['rfm_score'] > 4, 'High Value Customers', np.where(
        rfm['rfm_score'] > 3, 'Medium Value Customers', np.where(
        rfm['rfm_score'] > 1.6, 'Low Value Customers',
                                'Lost Customers'))))
    
    return rfm   


# #### Create customer_segmentation data

# In[12]:


def create_customer_segmentation(df):
    customer_segmentation = df.groupby('customer_segment').customer_id.nunique().sort_values(ascending=False).reset_index()
    customer_segmentation.rename(columns={'customer_id':'customer_count'}, inplace=True)
    
    return customer_segmentation


# ### Load Data

# In[13]:


df = pd.read_csv('C:/Users/LENOVO/anaconda3/envs/main-ds/e-commerce_data.csv')


# In[14]:


# pastikan order_date bertipe data datetime
df['order_date'] = pd.to_datetime(df['order_date'])

# urutkan berdasarkan order_date untuk filtering pada dashboard
df.sort_values(by='order_date', inplace=True)
df.reset_index(inplace=True)


# In[15]:


map_df = pd.read_csv('C:/Users/LENOVO/anaconda3/envs/main-ds/E-Commerce Public Dataset/map_df.csv')
map_df.rename(columns={
    'geolocation_lat':'latitude',
    'geolocation_lng':'longitude',
}, inplace=True)


# # Membuat Komponen

# ## Membuat Side Bar

# In[39]:


with st.sidebar:
    # menambahkan logo perusahaan
    img = Image.open('C:/Users/LENOVO/anaconda3/envs/main-ds/E-Commerce Public Dataset/logo.png')
    st.image(img)
    
    # menentukan start date dan end date pada date input
    start_date, end_date = st.date_input(
        label = 'Rentang Waktu',
        min_value = df['order_date'].min(),
        max_value = df['order_date'].max(),
        value = [df['order_date'].min(), df['order_date'].max()])


# ### Membuat main_df/dataframe yang telah difilter

# In[17]:


main_df = df[(df['order_date'] >= str(start_date)) &
            (df['order_date'] <= str(end_date))]


# ### Membuat data yang diperlukan untuk visualisasi dengan helper function

# In[18]:


monthly_orders = create_monthly_orders(main_df)
weekly_orders = create_weekly_orders(main_df)
daily_orders = create_daily_orders(main_df)
number_of_product_sold = create_number_of_product_sold(main_df)
byseller = create_byseller(main_df)
top_seller = create_top_seller(main_df)
seller_rank = create_seller_rank(top_seller)
bycity = create_bycity(main_df)
bystate = create_bystate(main_df)
rfm = create_rfm(main_df)
customer_segmentation = create_customer_segmentation(rfm)


# ## Membuat Dashboard

# ### Melengkapi Dashboard dengan Berbagai Visualisasi Data

# #### Header

# In[19]:


st.header('E-Commerce Dashboard :sparkles:')


# ##### Menentukan warna yang akan digunakan

# In[20]:


main_color = '#BE2773'
second_color = '#3B6BFF'
background_color = '#0E1117'
rank_colors = ["#BE2773", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730"]
rank_colors2 = ["#BE2773", "#BE2773", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730", "#262730"]
segment_colors = ["#BE2773", '#46244C', '#712B75','#C74B50', '#D49B54']


# ##### Menentukan warna background visualisasi dan mengatur jarak antar visualisasi

# In[ ]:


plt.rcParams['figure.facecolor'] = background_color
plt.tight_layout()


# #### Total Orders

# In[47]:


st.subheader('Total Orders :heavy_dollar_sign:')

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 30))

# monthly
sns.lineplot(x='order_date', y='order_count', data=monthly_orders, color=main_color, ax=ax[0])
ax[0].set_xticks(monthly_orders['order_date'])
ax[0].set_xticklabels(monthly_orders['order_date'], rotation=45, color='white', size=20)
ax[0].tick_params(axis='y', labelcolor='white', labelsize=12)
ax[0].tick_params(axis='x', labelcolor='white', labelsize=15)
ax[0].set_xlabel(None)
ax[0].set_ylabel(None)
ax[0].set_title('Total Sales Per Month', fontsize=30, color='white')
ax[0].legend()

# weekly
sns.lineplot(x='order_date', y='order_count', data=weekly_orders, color=main_color, ax=ax[1])
ax[1].set_xticks(range(0, len(weekly_orders['order_date']), 15))
ax[1].set_xticklabels(weekly_orders['order_date'][::15], color='white')
ax[1].tick_params(axis='y', labelcolor='white', labelsize=12)
ax[1].tick_params(axis='x', labelcolor='white', labelsize=17)
ax[1].set_xlabel(None)
ax[1].set_ylabel(None)
ax[1].set_title('Total Sales Per Week', fontsize=30, color='white')
ax[1].legend()

# daily
max_index = daily_orders['order_count'].idxmax()
max_date = daily_orders['order_date'].iloc[max_index]

sns.lineplot(x='order_date', y='order_count', data=daily_orders, color=main_color, ax=ax[2])
ax[2].set_xticks(range(0, len(daily_orders['order_date']), 90))
ax[2].set_xticklabels(daily_orders['order_date'][::90], color='white')
ax[2].text(max_date, daily_orders['order_count'].iloc[max_index],
         f'Date: {daily_orders["order_date"].iloc[max_index]}',
         fontsize=17, color=second_color, va='center')
ax[2].tick_params(axis='y', labelcolor='white', labelsize=12)
ax[2].tick_params(axis='x', labelcolor='white', labelsize=15)
ax[2].set_xlabel(None)
ax[2].set_ylabel(None)
ax[2].set_title('Total Sales Per Day', fontsize=30, color='white')
ax[2].legend()


st.pyplot(fig)


# #### Best and Worst Performing Product by Number of Sales

# In[ ]:


st.subheader('Best and Worst Performing Product by Number of Orders')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

sns.barplot(x="number_of_order", y="product_category_name", data=number_of_product_sold.sort_values(by="number_of_order", ascending=False).head(10), palette=rank_colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel('Number of Orders', fontsize=35, color='white')
ax[0].set_title("Best Performing Product", loc="center", fontsize=40, color='white')
ax[0].tick_params(axis ='y', labelsize=30, labelcolor='white')
ax[0].tick_params(axis ='x', labelsize=23, labelcolor='white')
 
sns.barplot(x="number_of_order", y="product_category_name", data=number_of_product_sold.sort_values(by="number_of_order", ascending=True).head(10), palette=rank_colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel('Number of Orders', fontsize=35, color='white')
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=40, color='white')
ax[1].tick_params(axis='y', labelsize=30, labelcolor='white')
ax[1].tick_params(axis='x', labelsize=23, labelcolor='white') 

st.pyplot(fig)


# #### Number of Sales by Seller

# In[25]:


st.subheader('Number of Sales by Seller')

fig, ax = plt.subplots(figsize=(20, 10))

sns.barplot(
    x="number_of_order", 
    y="seller_id",
    data=byseller.sort_values(by="number_of_order", ascending=False).head(10),
    palette=rank_colors
)
ax.set_ylabel('Seller Id', fontsize=35, color='white')
ax.set_xlabel(None)
ax.tick_params(axis='y', labelsize=30, labelcolor='white')
ax.tick_params(axis='x', labelsize=20, labelcolor='white')

st.pyplot(fig)


# #### Top Seller

# In[45]:


st.subheader('Top Seller :+1:')
seller_rank = seller_rank.sort_values(by='rank', ascending=True).reset_index()
seller_rank.drop(columns=['index'], inplace=True)

st.dataframe(seller_rank)


# #### Customer Demographics and Distribution of Customer Location

# In[46]:


st.subheader('Customer Demographics and Distribution of Customer Location :earth_americas:')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

sns.barplot(x='customer_count', y='customer_city', data=bycity.sort_values(by='customer_count', ascending=False).head(10), palette=rank_colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title('Number of Customer by City', loc='center', fontsize=40, color='white')
ax[0].tick_params(axis = 'x', labelsize=23, labelcolor='white')
ax[0].tick_params(axis = 'y', labelsize=30, labelcolor='white')
 
sns.barplot(x='customer_count', y='customer_state', data=bystate.sort_values(by='customer_count', ascending=False).head(10), palette=rank_colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title('Number of Customer by State', loc='center', fontsize=40, color='white')
ax[1].tick_params(axis = 'x', labelsize=23, labelcolor='white')
ax[1].tick_params(axis = 'y', labelsize=30, labelcolor='white')

st.pyplot(fig)


# In[33]:


st.map(map_df, use_container_width=False)


# #### Best Customer Based on RFM Parameters

# In[ ]:


st.subheader('Best Customer Based on RFM Parameters')

col1, col2, col3 = st.columns(3)

# Average Recency
with col1:
    avg_recency = round(rfm.recency.mean(), 1)
    st.metric('Average Recency (days)', value=avg_recency)

# average Frequency
with col2:
    avg_frequency = round(rfm.frequency.mean(), 2)
    st.metric('Average Frequency', value=avg_frequency)

# average Monetary
with col3:
    avg_monetary = format_currency(rfm.monetary.mean(), 'EUR', locale='es_CO')
    st.metric('Average Monetary', value=avg_monetary)


# In[ ]:


top_recency = rfm.sort_values(by='recency', ascending=True).reset_index().head(5)
top_recency = top_recency['customer_id']
top_frequency = rfm.sort_values(by='frequency', ascending=False).reset_index().head(5)
top_frequency = top_frequency['customer_id']
top_monetary = rfm.sort_values(by='monetary', ascending=False).reset_index().head(5)
top_monetary = top_monetary['customer_id']

col1, col2, col3 = st.columns(3)
# recency
with col1:
    fig, ax = plt.subplots(figsize=(30,30))
    
    sns.barplot(x='customer_id', y='recency', data=rfm.sort_values(by='recency', ascending=True).head(5), color=main_color, ax=ax)
    ax.set_ylabel(None)
    ax.set_xlabel('Customer Id', fontsize=120, color='white')
    ax.set_title('By Recency (days)', loc='center', fontsize=150, color='white')
    ax.tick_params(axis = 'x', labelsize=100, labelcolor='white')
    ax.tick_params(axis = 'y', labelsize=100, labelcolor='white')
    ax.set_xticklabels(top_recency, rotation=90, color='white')
    st.pyplot(fig)

# frequancy
with col2:
    fig, ax = plt.subplots(figsize=(30,30))
    
    sns.barplot(x='customer_id', y='frequency', data=rfm.sort_values(by='frequency', ascending=False).head(5), color=main_color, ax=ax)
    ax.set_ylabel(None)
    ax.set_xlabel('Customer Id', fontsize=120, color='white')
    ax.set_title('By Frequency', loc='center', fontsize=150, color='white')
    ax.tick_params(axis = 'x', labelsize=100, labelcolor='white')
    ax.tick_params(axis = 'y', labelsize=100, labelcolor='white')
    ax.set_xticklabels(top_frequency, rotation=90, color='white')
    st.pyplot(fig)

# monetary
with col3:
    fig, ax = plt.subplots(figsize=(30,30))
    
    sns.barplot(x='customer_id', y='monetary', data=rfm.sort_values(by='monetary', ascending=False).head(5), color=main_color, ax=ax)
    ax.set_ylabel(None)
    ax.set_xlabel('Customer Id', fontsize=120, color='white')
    ax.set_title('By Monetary', loc='center', fontsize=150, color='white')
    ax.tick_params(axis = 'x', labelsize=100, labelcolor='white')
    ax.tick_params(axis = 'y', labelsize=100, labelcolor='white')
    ax.set_xticklabels(top_monetary, rotation=90, color='white')
    st.pyplot(fig)


# #### Distribution of Customer Segment

# In[26]:


st.subheader('Distribution of Customer Segment')

segments = customer_segmentation['customer_segment']
customer_counts = customer_segmentation['customer_count']
percentages = [count / sum(customer_counts) * 100 for count in customer_counts]

fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(
    customer_counts,
    labels=[None]*len(segments), 
    autopct='', 
    startangle=90, 
#     wedgeprops=dict(width=0.6), # donut
    pctdistance=0.5,
    shadow=True,
    explode=(0.05, 0.05, 0, 0, 0),
    colors=segment_colors
)
ax.legend(labels=segments, loc='center left', bbox_to_anchor=(1, 0.5))

# Menambahkan lingkaran di tengah untuk membuat donut chart
# centre_circle = plt.Circle((0,0),0.1,fc=background_color)
# ax.add_patch(centre_circle)

ax.text(0, -1.2, 'Total 100%', ha='center', va='center', fontsize=12, color='white')

for i, (segment, percentage) in enumerate(zip(segments, percentages)):
    if percentage > 10:
        autotexts[i].set_text(f'{percentage:.1f}%')
        autotexts[i].set_size(10)
        autotexts[i].set_color('white')

st.pyplot(fig)

st.caption('Muhamad Syarif Fakhrezi:copyright:')

