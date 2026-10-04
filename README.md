# Databricks MLflow Model + Wheel Demo

End-to-end lifecycle: GitHub Actions -> Python wheel -> Databricks Job -> MLflow agent -> Unity Catalog -> Champion alias -> Model Serving.

Default LLM: `databricks-meta-llama-3-3-70b-instruct`.

## Deploy

Create the Unity Catalog schema:

```sql
CREATE SCHEMA IF NOT EXISTS main.mlflow_demo;
```

Add GitHub environment `databricks` with secrets:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

Then run **Actions -> Build and Deploy Databricks Agent -> Run workflow**.

The workflow builds and tests the wheel, uploads the wheel and registration script to Databricks Workspace Files, submits a one-time Databricks Job with the wheel attached, and waits for completion.

The Databricks job:
1. Builds a small LangChain agent using the Databricks-hosted LLM.
2. Logs it with MLflow.
3. Registers it in Unity Catalog.
4. Sets the `Champion` alias to the new model version.
5. Creates or updates the Model Serving endpoint.

Default result:

```
main.mlflow_demo.demo_agent
        |
     Version 1
        |
    Champion
        |
        v
model-wheel-agent
```

Cleanup workflow: **Actions -> Delete Databricks Agent Serving Endpoint**.

The wheel is the Python package. The MLflow model is the versioned deployable artifact. Unity Catalog stores/versions the model and the Champion alias points to the promoted version.
