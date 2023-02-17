import string
import pyspark
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
from pyspark.sql import functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col

def generate_unique_body(subject: str, has_attachments: str, importance: str) -> str:
    prompt_text = 'Write UniqueBody with Subject ' + subject + ', HasAttachements ' + has_attachments + ' and Importance ' + importance + '.'
    generated_text = prompt_text # TODO - Add call to GTP method
    return generated_text

def generate_body_preview(unique_body:string) -> string:
    if(len(unique_body) <= 255):
        return unique_body 
    else:
        return unique_body[0:255]

udf_gpt_unique_body = F.udf(generate_unique_body, StringType())
udf_gpt_body_preview = F.udf(generate_body_preview, StringType())

def get_dataframe_gpt_unique_body(df: DataFrame) -> DataFrame:
    df_decorated = df.withColumn("UniqueBody", udf_gpt_unique_body('Subject','HasAttachments','Importance','InferenceClassification','UnsubscribeEnabled')).withColumn("BodyPreview",udf_gpt_body_preview('UniqueBody'))
    return df_decorated