
def read_csv_from_minio(
    minio_endpoint: str,
    minio_secret: str,
    minio_key: str,
    bucket_name: str,
    object_name: str
):

    import minio
    import pandas as pd

    from io import StringIO
    
    minio_client = minio.Minio(
        minio_endpoint, 
        minio_secret,
        minio_key,
        secure=False
    )

    # This returns binary encoded data
    response = minio_client.get_object(bucket_name, object_name)
    decoded_csv_string = response.data.decode('utf-8')
    
    return pd.read_csv(StringIO(decoded_csv_string))

