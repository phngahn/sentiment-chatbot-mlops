#  Terraform – S3 -> SQS -> Lambda pipeline (chunk-aware)

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables

variable "aws_region"      { default = "ap-southeast-1" }
variable "input_bucket"    { default = "my-reviews-data" }
variable "output_bucket"   { default = "my-reviews-data" }
variable "trigger_prefix"  { default = "raw/" }
variable "chunk_prefix"    { default = "chunks/" }
variable "output_prefix"   { default = "processed/" }
variable "final_prefix"    { default = "final/" }
variable "chunk_size"      { default = "5000" }
variable "gemini_api_key"  { sensitive = true }
variable "wandb_api_key" {
  sensitive = true
  default   = ""
}
variable "wandb_project"   { default = "reviews-pipeline" }
variable "lambda_timeout"  { default = 900 }
variable "lambda_memory"   { default = 3008 }
variable "image_uri"       { description = "Full ECR image URI" }

locals {
  function_name = "reviews-pipeline"
  ecr_repo_name = "reviews-pipeline"
}

# ECR

resource "aws_ecr_repository" "pipeline" {
  name                 = local.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

# SQS Dead Letter Queue
resource "aws_sqs_queue" "pipeline_dlq" {
  name                        = "reviews-pipeline-dlq"
  message_retention_seconds = 1209600 
}

# SQS Main Queue
resource "aws_sqs_queue" "pipeline_queue" {
  name                        = "reviews-pipeline-queue"
  visibility_timeout_seconds = 960
  message_retention_seconds  = 604800

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.pipeline_dlq.arn
    maxReceiveCount     = 10
  })
}

# SQS Policy: cho phép S3 gửi message

resource "aws_sqs_queue_policy" "pipeline_queue_policy" {
  queue_url = aws_sqs_queue.pipeline_queue.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "s3.amazonaws.com" }
      Action    = "sqs:SendMessage"
      Resource  = aws_sqs_queue.pipeline_queue.arn
      Condition = {
        ArnLike = {
          "aws:SourceArn" = "arn:aws:s3:::${var.input_bucket}"
        }
      }
    }]
  })
}

# S3 Notification -> SQS

resource "aws_s3_bucket_notification" "pipeline_trigger" {
  bucket     = var.input_bucket
  depends_on = [aws_sqs_queue_policy.pipeline_queue_policy]

  queue {
    id            = "raw-csv-trigger"
    queue_arn     = aws_sqs_queue.pipeline_queue.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "raw/"
    filter_suffix = ".csv"
  }

  queue {
    id            = "chunk-csv-trigger"
    queue_arn     = aws_sqs_queue.pipeline_queue.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "chunks/"
    filter_suffix = ".csv"
  }
}

# IAM Role cho Lambda

resource "aws_iam_role" "lambda_exec" {
  name = "${local.function_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_s3" {
  name = "s3-access"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",  # xóa chunkmeta sau khi concat xong
        "s3:ListBucket",
        "s3:HeadObject",    # check chunkdone flags
      ]
      Resource = [
        "arn:aws:s3:::${var.input_bucket}",
        "arn:aws:s3:::${var.input_bucket}/*",
        "arn:aws:s3:::${var.output_bucket}",
        "arn:aws:s3:::${var.output_bucket}/*",
      ]
    }]
  })
}

resource "aws_iam_role_policy" "lambda_sqs" {
  name = "sqs-access"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:GetQueueAttributes",
        "sqs:ChangeMessageVisibility",
      ]
      Resource = aws_sqs_queue.pipeline_queue.arn
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Lambda Function

resource "aws_lambda_function" "pipeline" {
  function_name = local.function_name
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  timeout       = var.lambda_timeout
  memory_size   = var.lambda_memory

  ephemeral_storage { size = 5120 }

  environment {
    variables = {
      # AWS
      INPUT_BUCKET   = var.input_bucket
      OUTPUT_BUCKET  = var.output_bucket
      OUTPUT_PREFIX  = var.output_prefix
      TRIGGER_PREFIX = var.trigger_prefix

      # Chunk config
      CHUNK_PREFIX   = var.chunk_prefix
      CHUNK_SIZE     = var.chunk_size
      FINAL_PREFIX   = var.final_prefix

      # Gemini
      GEMINI_API_KEY = var.gemini_api_key
      GEMINI_MODEL   = "gemini-3.1-flash-lite"
      GEMINI_SLEEP   = "4"

      # W&B
      WANDB_API_KEY  = var.wandb_api_key
      WANDB_PROJECT  = var.wandb_project

      # Lambda
      CHECKPOINT_DIR = "/tmp/checkpoints"
      SQS_QUEUE_URL  = aws_sqs_queue.pipeline_queue.url
    }
  }
}

# SQS -> Lambda trigger
resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = aws_sqs_queue.pipeline_queue.arn
  function_name    = aws_lambda_function.pipeline.arn
  batch_size       = 1 

  scaling_config {
    maximum_concurrency = 2
  }
}

# Chunk Scheduler Lambda (cron 14:05 VN = 07:05 UTC)

resource "aws_lambda_function" "chunk_scheduler" {
  function_name = "reviews-chunk-scheduler"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  timeout       = 120
  memory_size   = 512

  image_config {
    command = ["lambda_scheduler.handler"]
  }

  environment {
    variables = {
      INPUT_BUCKET  = var.input_bucket
      OUTPUT_BUCKET = var.output_bucket
      CHUNK_PREFIX  = var.chunk_prefix
      CHUNK_SIZE    = var.chunk_size
      GEMINI_API_KEY = var.gemini_api_key
    }
  }
}

resource "aws_cloudwatch_event_rule" "daily_chunk_scheduler" {
  name                = "reviews-daily-chunk-scheduler"
  description         = "Chạy mỗi ngày 07:05 UTC (14:05 VN) để upload chunk tiếp theo"
  schedule_expression = "cron(5 7 * * ? *)"
}

resource "aws_cloudwatch_event_target" "chunk_scheduler_target" {
  rule      = aws_cloudwatch_event_rule.daily_chunk_scheduler.name
  target_id = "chunk-scheduler"
  arn       = aws_lambda_function.chunk_scheduler.arn
}

resource "aws_lambda_permission" "allow_eventbridge_scheduler" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chunk_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_chunk_scheduler.arn
}

# Outputs

output "lambda_arn" { value = aws_lambda_function.pipeline.arn }
output "ecr_repo"   { value = aws_ecr_repository.pipeline.repository_url }
output "sqs_url"    { value = aws_sqs_queue.pipeline_queue.url }
output "dlq_url"    { value = aws_sqs_queue.pipeline_dlq.url }
