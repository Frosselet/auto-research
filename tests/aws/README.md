# tests/aws/

Two tiers of AWS-flavour tests.

## Unit tests (run in CI)

```bash
uv run pytest tests/aws/ -m "not integration"
```

Everything under `tests/aws/` except `test_integration_sandbox.py` uses `moto`'s
`@mock_aws` and never touches a real AWS account. Handler chain, store, ASL
builder, submit flow — all verified in <5 seconds.

## Integration test (run by enablement team after `terraform apply`)

```bash
export AWS_AUTORESEARCH_INTEGRATION=1
export AUTO_RESEARCH_STATE_MACHINE_ARN=arn:aws:states:eu-west-1:...
export AUTO_RESEARCH_S3_BUCKET=auto-research-<team>-dev
export AUTO_RESEARCH_DDB_TABLE=auto-research-<team>-dev-ledger
export AUTO_RESEARCH_OPENAI_SECRET_ID=openai/<team>/dev
aws sso login --profile <team>-dev  # or equivalent
uv run pytest tests/aws/test_integration_sandbox.py -m integration
```

Runs one short execution (K=2, max_iter=4, budget $0.05) end-to-end against the
real stack. Expect ~5 minutes wall-clock and ~$0.05 OpenAI spend. Use this after
any infra/ change to verify the deployment.
