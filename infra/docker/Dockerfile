# Single container image shared by all four Step Functions Lambdas.
# Dispatch to the right handler via the HANDLER env var
# (set per Lambda function in Terraform: propose | train | evaluate | decide).
#
# Size budget: < 1.5 GB compressed (hard limit is 10 GB). scipy+sklearn+pandas
# account for ~800 MB; headroom for analysts' own train.py imports.
FROM public.ecr.aws/lambda/python:3.12

# Ensure a modern pip so scipy wheels resolve cleanly on Python 3.12.
RUN pip install --no-cache-dir --upgrade "pip>=24"

# Runtime deps — bake the scipy/ML stack so user train.py scripts "just work"
# out of the box. auto_research's own runtime deps (openai, pydantic, pyyaml)
# are pulled in explicitly; boto3 is pre-installed on the lambda base image
# but we pin a minimum for the SDK features we use (Distributed Map etc.
# are MVP-3 territory, but we keep a recent floor for safety).
RUN pip install --no-cache-dir \
      "numpy>=2.0.0" \
      "pandas>=2.2.0" \
      "scikit-learn>=1.5.0" \
      "scipy>=1.13.0" \
      "openai>=1.50.0" \
      "pydantic>=2.8.0" \
      "pyyaml>=6.0.0" \
      "boto3>=1.35.0"

# Copy the two packages directly (simpler + smaller than building a wheel).
COPY packages/auto_research        ${LAMBDA_TASK_ROOT}/auto_research
COPY packages/auto_research_aws    ${LAMBDA_TASK_ROOT}/auto_research_aws
COPY infra/docker/entrypoint.py    ${LAMBDA_TASK_ROOT}/entrypoint.py

# Default CMD points at the dispatcher; the Lambda invoker calls
# entrypoint.dispatch(event, context).
CMD ["entrypoint.dispatch"]
