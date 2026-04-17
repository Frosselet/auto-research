variable "name_prefix" {
  type = string
}

variable "tags" {
  type    = map(string)
  default = {}
}

# The secret value is NOT set by Terraform — the enablement team writes it
# once out-of-band after apply:
#
#   aws secretsmanager put-secret-value \
#     --secret-id <name> --secret-string sk-...
#
# This keeps the OpenAI key out of Terraform state.
resource "aws_secretsmanager_secret" "openai" {
  name                    = "${var.name_prefix}/openai"
  description             = "OpenAI API key for auto-research proposer Lambda"
  recovery_window_in_days = 7
  tags                    = var.tags
}

output "secret_arn" {
  value = aws_secretsmanager_secret.openai.arn
}

output "secret_id" {
  value = aws_secretsmanager_secret.openai.id
}
