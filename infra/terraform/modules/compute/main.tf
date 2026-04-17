variable "name_prefix" {
  type = string
}

variable "image_uri" {
  type        = string
  description = "Fully qualified ECR image URI, e.g. <acct>.dkr.ecr.<region>.amazonaws.com/auto-research-handlers:<sha>"
}

variable "lambda_role_arn" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}

resource "aws_ecr_repository" "handlers" {
  name                 = "${var.name_prefix}-handlers"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = var.tags
}

# One Lambda per handler, all pointing at the same image, distinguished by HANDLER env var.
locals {
  handlers = {
    propose = {
      memory_mb  = 512
      timeout_s  = 60
    }
    train = {
      memory_mb  = 10240
      timeout_s  = 870 # 14m30s — leave margin under the 15m hard cap
    }
    evaluate = {
      memory_mb  = 4096
      timeout_s  = 300
    }
    decide = {
      memory_mb  = 512
      timeout_s  = 60
    }
  }
}

resource "aws_lambda_function" "handler" {
  for_each = local.handlers

  function_name = "${var.name_prefix}-${each.key}"
  package_type  = "Image"
  image_uri     = var.image_uri
  role          = var.lambda_role_arn
  memory_size   = each.value.memory_mb
  timeout       = each.value.timeout_s
  architectures = ["x86_64"]

  ephemeral_storage {
    size = 10240 # 10 GB of /tmp — required so train_handler can materialize data + artifacts
  }

  environment {
    variables = {
      HANDLER = each.key
    }
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "handler" {
  for_each          = local.handlers
  name              = "/aws/lambda/${var.name_prefix}-${each.key}"
  retention_in_days = 14
  tags              = var.tags
}

output "ecr_repository_url" {
  value = aws_ecr_repository.handlers.repository_url
}

output "propose_arn" {
  value = aws_lambda_function.handler["propose"].arn
}
output "train_arn" {
  value = aws_lambda_function.handler["train"].arn
}
output "evaluate_arn" {
  value = aws_lambda_function.handler["evaluate"].arn
}
output "decide_arn" {
  value = aws_lambda_function.handler["decide"].arn
}

output "all_lambda_arns" {
  value = [for fn in aws_lambda_function.handler : fn.arn]
}
