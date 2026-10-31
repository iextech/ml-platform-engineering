
from io import BytesIO

### 
def save_df_to_s3(client, df, bucket_name, path):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)
    client.put_object(bucket_name, path, data=csv_buffer, length=len(csv_bytes), content_type='application/csv')
