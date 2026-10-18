# Model Wheel: Databricks Asset Bundles

This repository packages a Python LangChain agent as a wheel, logs it as an MLflow PyFunc model, registers the model in Unity Catalog, promotes a `Champion` alias, and creates or updates a Model Serving endpoint.

The deployment resources are managed with **Databricks Asset Bundles (DAB)**. GitHub Actions runs linting and tests, validates the bundle, deploys the declared job, and invokes it on Databricks serverless job compute.

## Architecture

1. GitHub Actions runs Ruff, builds the wheel, installs it, and runs pytest.
2. `databricks bundle validate` validates the selected target.
3. `databricks bundle deploy` deploys the bundle-managed job.
4. `databricks bundle run` starts the job on serverless compute.
5. The job logs the agent as an MLflow PyFunc model and registers it in Unity Catalog.
6. The job creates or updates the target Model Serving endpoint, then assigns the `Champion` alias to the newly registered model version.

The MLflow model artifact contains a copy of the `model_wheel_demo` Python package. The wheel is also installed as a job dependency. This avoids relying on a local runner path or a long-lived DBFS file path when the serving endpoint loads the model.

## Prerequisites

- A Databricks workspace that supports serverless jobs and serverless Python script tasks.
- Unity Catalog enabled, with the selected catalog/schema available to the deployment identity.
- The LLM serving endpoint `databricks-meta-llama-3-3-70b-instruct` already deployed and accessible to the agent.
- Permissions to create registered models and model versions, update model aliases, create/update serving endpoints, and invoke the LLM endpoint.
- A GitHub OIDC federation trust configured between GitHub Actions and a Databricks service principal.

Create the schema once if it does not already exist:

```sql
CREATE SCHEMA IF NOT EXISTS main.mlflow_demo;
```

### GitHub environments and secrets

Create GitHub environments named:

- `development`
- `production`

Configure the following environment secrets in both environments:

- `DATABRICKS_HOST`: workspace URL, for example `https://<workspace-host>`
- `DATABRICKS_CLIENT_ID`: application/client ID for the Databricks service principal configured for GitHub OIDC federation

The workflow uses `DATABRICKS_AUTH_TYPE=github-oidc`. Configure the federation issuer, subject, and audience in Databricks to match your repository and GitHub environment. Do not add a personal access token to the repository.

Protect the GitHub `production` environment with required reviewers. The production deployment is manually triggered from Actions and must pass that environment's approval rules.

## Bundle targets

The bundle defines two targets in `databricks.yml`:

| Target | Model | Serving endpoint |
|---|---|---|
| `dev` | `main.mlflow_demo.demo_agent` | `model-wheel-dev-agent` |
| `prod` | `main.mlflow_demo.demo_agent` | `model-wheel-agent` |

Development uses DAB development mode, which can prefix or otherwise isolate deployed resources. Production uses production mode. Both currently point at the same model name/schema; for strict isolation, use a separate production schema or catalog and configure target-specific `catalog`/ `schema` variables before enabling production deployments.

## Local commands

Install Python 3.12 and the Databricks CLI, then authenticate using your configured identity. Set `DATABRICKS_HOST` and configure Databricks CLI OAuth authentication.

```bash
python -m pip install --upgrade build pytest ruff
python -m build
python -m pip install dist/*.whl
pytest -q
ruff check .

databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run -t dev model_wheel_deploy
```

For production, use the protected GitHub Actions `production` environment or run the equivalent commands only with the production service principal and workspace configuration:

```bash
databricks bundle validate -t prod
databricks bundle deploy -t prod
databricks bundle run -t prod model_wheel_deploy
```

Do not deploy the production target until its environment points to the intended production workspace and the model/endpoint naming and permissions have been verified.

## GitHub Actions flow

- Pull requests to `main`: lint, build, install the wheel, and run tests.
- Pushes to `main`: run CI and deploy to development.
- Manual workflow dispatch: select `dev` or `prod`. Production uses the protected `production` environment.

## Cleanup

The `Delete development serving endpoint` workflow can delete only the hard-coded development endpoint `model-wheel-dev-agent`. It requires the exact confirmation phrase `DELETE model-wheel-dev-agent`. It does not delete the Unity Catalog model, model versions, aliases, or the production endpoint.

Do not use `databricks bundle destroy -t prod` as a general cleanup command. Bundle destruction deletes managed resources and should only be performed after reviewing the plan and confirming the target.

## Serverless compatibility notes

The job uses a serverless environment specification and installs the built wheel plus the pinned LangChain dependencies. Serverless job support and available Python versions vary by workspace/cloud/region. If `bundle validate` or the workspace rejects the Python script task/environment specification, use a supported serverless task type/runtime for that workspace or switch the job to a bundle-managed job cluster; do not silently fall back to an unrelated existing cluster.

The workflow has been authored in this branch but must still be validated against your actual workspace, service-principal permissions, and serverless support before it can be considered production-ready.
