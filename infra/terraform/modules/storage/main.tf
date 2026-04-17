variable "name_prefix" {
  type        = string
  description = "Prefix for resource names, e.g. auto-research-<team>-<env>."
}

variable "tags" {
  type    = map(string)
  default = {}
}

resource "aws_s3_bucket" "artifacts" {
  bucket = var.name_prefix
  tags   = var.tags
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "ledger" {
  name         = "${var.name_prefix}-ledger"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  range_key    = "sk"

  attribute {
    name = "run_id"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }

  point_in_time_recovery {
    enabled = true
  }

  tags = var.tags
}

output "bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "bucket_arn" {
  value = aws_s3_bucket.artifacts.arn
}

output "table_name" {
  value = aws_dynamodb_table.ledger.name
}

output "table_arn" {
  value = aws_dynamodb_table.ledger.arn
}
