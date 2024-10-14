import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./BTC-Daily.csv')
df = df.drop_duplicates()

# Range of dates covered
df['date'] = pd.to_datetime(df['date'])
date_range = str(df['date'].dt.date.min()) + ' to ' + \
    str(df['date'].dt.date.max())
print(date_range)

# Tạo cột năm, tháng, ngày từ cột 'date'
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Giả sử unique_years chứa danh sách các năm có trong dữ liệu
unique_years = df['year'].unique()

for year in unique_years:
    # Lọc dữ liệu theo từng năm
    year_month_day = df[df['year'] == year][[
        'year', 'month', 'day']].drop_duplicates()

    # Kết hợp dữ liệu theo năm, tháng, ngày
    merged_data = pd.merge(year_month_day, df, on=[
                           'year', 'month', 'day'], how='left')

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(merged_data['date'], merged_data['close'])
    plt.title(f'Bitcoin Closing Prices - {year}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price (USD)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
