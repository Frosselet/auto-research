variable "name_prefix" {
  type = string
}

variable "bucket_arn" {
  type = string
}

variable "table_arn" {
  type = string
}

variable "secret_arn" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}

# ── Lambda execution role ──────────────────────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${var.name_prefix}-lambda"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = var.tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda_app" {
  statement {
    sid     = "S3Objects"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"]
    resources = [
      var.bucket_arn,
      "${var.bucket_arn}/*",
    ]
  }
  statement {
    sid     = "DDBLedger"
    actions = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query", "dynamodb:BatchGetItem"]
    resources = [
      var.table_arn,
      "${var.table_arn}/index/*",
    ]
  }
  statement {
    sid       = "Secrets"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [var.secret_arn]
  }
}

resource "aws_iam_role_policy" "lambda_app" {
  name   = "${var.name_prefix}-lambda-app"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_app.json
}

# ── Step Functions execution role ──────────────────────────────────────────────

data "aws_iam_policy_document" "sfn_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sfn" {
  name               = "${var.name_prefix}-sfn"
  assume_role_policy = data.aws_iam_policy_document.sfn_assume.json
  tags               = var.tags
}

variable "lambda_arns" {
  type        = list(string)
  description = "ARNs of the four handler Lambdas the state machine invokes."
}

data "aws_iam_policy_document" "sfn_invoke" {
  statement {
    sid       = "InvokeLambdas"
    actions   = ["lambda:InvokeFunction"]
    resources = var.lambda_arns
  }
}

resource "aws_iam_role_policy" "sfn_invoke" {
  name   = "${var.name_prefix}-sfn-invoke"
  role   = aws_iam_role.sfn.id
  policy = data.aws_iam_policy_document.sfn_invoke.json
}

output "lambda_role_arn" {
  value = aws_iam_role.lambda.arn
}

output "sfn_role_arn" {
  value = aws_iam_role.sfn.arn
}
