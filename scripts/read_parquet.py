import pyarrow.parquet as pq

# Path to your Parquet file
parquet_file = 'parquet_data/sample.parquet'

# Read the Parquet file
table = pq.read_table(parquet_file)

# Print the contents of the Parquet file
print(table)