# import boto3
# import os
# import logging
# from pathlib import Path
# from botocore.exceptions import ClientError

# logger = logging.getLogger("tiki_crawler.upload_s3")

# def upload_to_s3(file_path, bucket_name, s3_file_name=None):
#     """
#     Kiểm tra file đã tồn tại trên S3 chưa, nếu chưa thì upload.
#     """
#     s3_client = boto3.client('s3')
    
#     if s3_file_name is None:
#         s3_file_name = os.path.basename(file_path)

#     # 1. Kiểm tra tồn tại
#     try:
#         s3_client.head_object(Bucket=bucket_name, Key=s3_file_name)
#         logger.info(f"File {s3_file_name} đã tồn tại trên S3. Bỏ qua.")
#         return True
#     except ClientError as e:
#         # Nếu lỗi 404 nghĩa là file chưa có -> Tiến hành upload
#         if e.response['Error']['Code'] == "404":
#             try:
#                 logger.info(f"Đang upload {file_path} lên s3://{bucket_name}/{s3_file_name}...")
#                 s3_client.upload_file(file_path, bucket_name, s3_file_name)
#                 logger.info("Upload thành công!")
#                 return True
#             except Exception as upload_error:
#                 logger.error(f"Lỗi khi upload: {upload_error}")
#                 return False
#         else:
#             logger.error(f"Lỗi kiểm tra S3: {e}")
#             return False

# if __name__ == "__main__":
#     # Test nhanh
#     BUCKET = "tiki-data-phi-quyen" # Thay bằng tên bucket của bà
#     DATA_FILE = "data/raw/products_reviews.csv" 
#     if os.path.exists(DATA_FILE):
#         upload_to_s3(DATA_FILE, BUCKET)