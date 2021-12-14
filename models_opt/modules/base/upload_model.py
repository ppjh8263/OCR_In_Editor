import os
from google.cloud import storage 
import psycopg2
import json

def upload(filename, model_name, epoch):
    # Upload model .pth file to google cloud storage
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/opt/ml/key.json"
    storage_client = storage.Client() 
    bucket = storage_client.bucket("oe_ocr_models")
    model_pth = f"{model_name}/{filename.split('/')[1]}_{epoch}.pth.tar"
    blob = bucket.blob(model_pth)
    blob.upload_from_filename(filename)

    # Add model url to db
    db = psycopg2.connect(host=os.environ["sql_host"], dbname='ocr',user='postgres',password=os.environ["sql_passwd"],port=5432)
    cursor = db.cursor()

    schema = 'public'
    table = 'models'
    data = ','.join([f"'{model_name}'", f"'{filename.split('/')[1]}'", f"'{model_pth}'"])
    query = f"INSERT INTO {schema}.{table} VALUES ({data})"
    cursor.execute(query)
    db.commit()
    
    db.close()