from shrike.compliant_logging import DataCategory

def read_dataframe(spark, path, format, logger):
    logger.info(f"Reading {format} dataset from path {path}",category=DataCategory.PUBLIC)
    if format.lower() == 'parquet':
        df = spark.read.option("recursiveFileLookup", "true").parquet(path)
    else:
        df = spark.read.option("recursiveFileLookup", "true").json(path)
    return df

def write_dataframe(df, path, format, partitions, logger):
    logger.info(f"Writting {format} dataset to path {path}",category=DataCategory.PUBLIC)
    df_repartitioned = df if partitions == 0 else df.repartition(int(partitions))
    if format.lower() == 'parquet':
        df_repartitioned.write.parquet(path)
    else:
        df_repartitioned.write.json(path)