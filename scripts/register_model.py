"""Register the agent in Unity Catalog and deploy it."""

from __future__ import annotations

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

from model_wheel_demo.agent import DEFAULT_MODEL, build_agent

MODEL_NAME = os.getenv("UC_MODEL_NAME", "main.mlflow_demo.demo_agent")
SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT", "model-wheel-agent")
LLM_MODEL = os.getenv("DATABRICKS_MODEL", DEFAULT_MODEL)


def main() -> None:
    mlflow.set_registry_uri("databricks-uc")
    agent = build_agent(LLM_MODEL)

    with mlflow.start_run() as run:
        model_info = mlflow.langchain.log_model(
            lc_model=agent,
            artifact_path="agent",
            registered_model_name=MODEL_NAME,
            input_example={"input": "What is MLflow?"},
        )
        print(f"MLflow run: {run.info.run_id}")

    version = model_info.registered_model_version
    print(f"Registered: {MODEL_NAME} version {version}")

    client = WorkspaceClient()

    client.registered_models.set_alias(
        name=MODEL_NAME,
        alias="Champion",
        version=version,
    )

    served_model = ServedModelInput(
        model_name=MODEL_NAME,
        model_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )

    endpoint_names = [e.name for e in client.serving_endpoints.list()]

    if SERVING_ENDPOINT in endpoint_names:
        client.serving_endpoints.update_config(
            name=SERVING_ENDPOINT,
            served_models=[served_model],
        )
    else:
        client.serving_endpoints.create(
            name=SERVING_ENDPOINT,
            config=EndpointCoreConfigInput(
                name=SERVING_ENDPOINT,
                served_models=[served_model],
            ),
        )

    print(f"Champion -> version {version}")
    print(f"Serving endpoint -> {SERVING_ENDPOINT}")


if __name__ == "__main__":
    main()
