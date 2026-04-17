variable "name_prefix" {
  type = string
}

variable "sfn_role_arn" {
  type = string
}

variable "propose_arn" {
  type = string
}

variable "train_arn" {
  type = string
}

variable "evaluate_arn" {
  type = string
}

variable "decide_arn" {
  type = string
}

variable "max_concurrency" {
  type        = number
  default     = 10
  description = "Maximum parallel Map branches. Matches spec.parallelism ceiling; must be <= 40 for Standard Map."
}

variable "tags" {
  type    = map(string)
  default = {}
}

# ASL source of truth lives at modules/orchestration/asl.json.tftpl.
# Regenerate with `make asl` (see infra/DEPLOY.md) which runs
# auto_research_aws.orchestrator.build_state_machine_definition and writes it here.
locals {
  definition = templatefile("${path.module}/asl.json.tftpl", {
    propose_arn     = var.propose_arn
    train_arn       = var.train_arn
    evaluate_arn    = var.evaluate_arn
    decide_arn      = var.decide_arn
    max_concurrency = var.max_concurrency
  })
}

resource "aws_sfn_state_machine" "loop" {
  name       = "${var.name_prefix}-loop"
  role_arn   = var.sfn_role_arn
  definition = local.definition
  type       = "STANDARD"

  logging_configuration {
    include_execution_data = true
    level                  = "ERROR"
    log_destination        = "${aws_cloudwatch_log_group.sfn.arn}:*"
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "sfn" {
  name              = "/aws/stepfunctions/${var.name_prefix}-loop"
  retention_in_days = 14
  tags              = var.tags
}

resource "aws_cloudwatch_metric_alarm" "executions_failed" {
  alarm_name          = "${var.name_prefix}-executions-failed"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ExecutionsFailed"
  namespace           = "AWS/States"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  dimensions = {
    StateMachineArn = aws_sfn_state_machine.loop.arn
  }
  treat_missing_data = "notBreaching"
  tags               = var.tags
}

output "state_machine_arn" {
  value = aws_sfn_state_machine.loop.arn
}
