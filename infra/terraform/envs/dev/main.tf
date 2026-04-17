terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
  }
}

provider "aws" {
  region = var.region
}

variable "team" {
  type        = string
  description = "Team slug; becomes part of the resource name prefix."
}

variable "region" {
  type    = string
  default = "eu-west-1"
}

variable "image_uri" {
  type        = string
  description = "ECR image URI for the shared Lambda container (output of `docker build && push`)."
}

variable "max_concurrency" {
  type        = number
  default     = 10
  description = "Upper bound on parallel Map branches; also the ceiling on spec.parallelism."
}

locals {
  name_prefix = "auto-research-${var.team}-dev"
  tags = {
    Project     = "auto-research"
    Environment = "dev"
    Team        = var.team
  }
}

module "storage" {
  source      = "../../modules/storage"
  name_prefix = local.name_prefix
  tags        = local.tags
}

module "secrets" {
  source      = "../../modules/secrets"
  name_prefix = local.name_prefix
  tags        = local.tags
}

# IAM must know about the Lambda ARNs to scope the Step Functions invoke policy,
# but the Lambdas need the Lambda role — two-pass is handled by constructing the
# ARN string from the function name we're about to create.
locals {
  lambda_arns = [
    for name in ["propose", "train", "evaluate", "decide"] :
    "arn:aws:lambda:${var.region}:${data.aws_caller_identity.current.account_id}:function:${local.name_prefix}-${name}"
  ]
}

data "aws_caller_identity" "current" {}

module "iam" {
  source      = "../../modules/iam"
  name_prefix = local.name_prefix
  bucket_arn  = module.storage.bucket_arn
  table_arn   = module.storage.table_arn
  secret_arn  = module.secrets.secret_arn
  lambda_arns = local.lambda_arns
  tags        = local.tags
}

module "compute" {
  source          = "../../modules/compute"
  name_prefix     = local.name_prefix
  image_uri       = var.image_uri
  lambda_role_arn = module.iam.lambda_role_arn
  tags            = local.tags
}

module "orchestration" {
  source          = "../../modules/orchestration"
  name_prefix     = local.name_prefix
  sfn_role_arn    = module.iam.sfn_role_arn
  propose_arn     = module.compute.propose_arn
  train_arn       = module.compute.train_arn
  evaluate_arn    = module.compute.evaluate_arn
  decide_arn      = module.compute.decide_arn
  max_concurrency = var.max_concurrency
  tags            = local.tags
}

# ── Outputs for the analyst notebook + integration tests ──────────────────────

output "state_machine_arn" {
  value = module.orchestration.state_machine_arn
}

output "s3_bucket" {
  value = module.storage.bucket_name
}

output "ddb_table" {
  value = module.storage.table_name
}

output "openai_secret_id" {
  value = module.secrets.secret_id
}

output "ecr_repository_url" {
  value = module.compute.ecr_repository_url
}
