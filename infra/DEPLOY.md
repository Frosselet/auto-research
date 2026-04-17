# Deploying auto-research to AWS (MVP-2)

Runbook for the enablement team. One Terraform apply per team + environment.
After this, quants point `auto_research.submit()` at the output ARNs and go.

## One-time account setup

1. Log into the target AWS account with admin-equivalent credentials.
2. Install: `terraform >= 1.6`, `aws` CLI v2, `docker`, `uv` (for the ASL regen step).

## Per-team deploy

All paths are relative to the repo root.

### 1. Build and push the Lambda container image

```bash
export AWS_REGION=eu-west-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=auto-research-<team>-dev-handlers
export IMAGE_TAG=$(git rev-parse --short HEAD)

# One-time: create the ECR repo (or let Terraform do it on first apply and rerun).
aws ecr describe-repositories --repository-names $ECR_REPO >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name $ECR_REPO --image-tag-mutability IMMUTABLE

# Build (amd64 to match Lambda's x86_64 architecture; on Apple Silicon add --platform).
docker build --platform linux/amd64 -f infra/docker/Dockerfile -t $ECR_REPO:$IMAGE_TAG .

# Push.
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag $ECR_REPO:$IMAGE_TAG $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG

export IMAGE_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
```

### 2. Apply Terraform

```bash
cd infra/terraform/envs/dev
terraform init
terraform apply \
  -var "team=<team>" \
  -var "region=$AWS_REGION" \
  -var "image_uri=$IMAGE_URI" \
  -var "max_concurrency=10"
```

Apply outputs:

- `state_machine_arn` — pass to `auto_research.submit(state_machine_arn=...)`
- `s3_bucket` — pass as `s3_bucket=...`
- `ddb_table` — pass as `ddb_table=...`
- `openai_secret_id` — pass as `openai_secret_id=...`
- `ecr_repository_url` — for subsequent image pushes

### 3. Populate the OpenAI secret

Terraform creates the empty secret. Write the key value once:

```bash
aws secretsmanager put-secret-value \
  --secret-id "$(terraform output -raw openai_secret_id)" \
  --secret-string "sk-..."
```

### 4. Smoke-test the stack

```bash
cd ../../../..   # back to repo root
export AWS_AUTORESEARCH_INTEGRATION=1
export AUTO_RESEARCH_STATE_MACHINE_ARN=$(cd infra/terraform/envs/dev && terraform output -raw state_machine_arn)
export AUTO_RESEARCH_S3_BUCKET=$(cd infra/terraform/envs/dev && terraform output -raw s3_bucket)
export AUTO_RESEARCH_DDB_TABLE=$(cd infra/terraform/envs/dev && terraform output -raw ddb_table)
export AUTO_RESEARCH_OPENAI_SECRET_ID=$(cd infra/terraform/envs/dev && terraform output -raw openai_secret_id)

uv run pytest tests/aws/test_integration_sandbox.py -m integration -v
```

Expect ~5 minutes, ~$0.05 OpenAI spend, four trials in the DynamoDB ledger, and
a promoted `best/train.py` in the S3 bucket.

### 5. Hand off to the analyst

Share the four output values. In the notebook:

```python
import auto_research
handle = auto_research.submit(
    "./spec.yaml",
    state_machine_arn="<state_machine_arn>",
    s3_bucket="<s3_bucket>",
    ddb_table="<ddb_table>",
    openai_secret_id="<openai_secret_id>",
    region="eu-west-1",
)
for snap in auto_research.watch(handle):
    print(snap)
print(auto_research.aws_results(handle))
```

## Regenerating the ASL template

The state machine ASL lives at
`infra/terraform/modules/orchestration/asl.json.tftpl`. It is a hand-maintained
Terraform template that mirrors
`auto_research_aws.orchestrator.build_state_machine_definition`. If you change
the Python builder, regenerate the template and commit:

```bash
uv run python -c "
import json
from auto_research_aws.orchestrator import build_state_machine_definition
d = build_state_machine_definition(
    propose_lambda_arn='\${propose_arn}',
    train_lambda_arn='\${train_arn}',
    evaluate_lambda_arn='\${evaluate_arn}',
    decide_lambda_arn='\${decide_arn}',
    max_concurrency=int('\${max_concurrency}') if '\${max_concurrency}'.isdigit() else 10,
)
print(json.dumps(d, indent=2))
"
```

Review the diff against `asl.json.tftpl` — remember that the template
substitutes `$${max_concurrency}` (numeric, unquoted) rather than a string.

## Common issues

- **`docker build` on Apple Silicon** — always pass `--platform linux/amd64`;
  Lambda does not run arm64 images from an x86_64 function (and vice versa).
- **`parallelism > 40`** — submit() rejects this up front. Standard Map caps at
  40 concurrent branches. Distributed Map (10k) is MVP-3.
- **Image push denied** — ECR is `IMMUTABLE`; you can't overwrite a tag. Bump
  the tag or use a fresh git SHA.
- **Slow cold start** — the first invocation per fresh container pulls ~1 GB of
  wheels. Subsequent invocations are warm. If this bites, pre-warm via
  `ProvisionedConcurrency` on the train Lambda.
