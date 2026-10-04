from __future__ import annotations

import argparse

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

from model_wheel_demo.agent import build_agent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--llm-model", required=True)
    p.add_argument("--serving-endpoint", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    mlflow.set_registry_uri("databricks-uc")

    agent = build_agent(args.llm_model)

    with mlflow.start_run() as run:
        info = mlflow.langchain.log_model(
            lc_model=agent,
            artifact_path="agent",
            registered_model_name=args.model_name,
            input_example={"input": "What is MLflow?"},
        )
        print(f"MLflow run: {run.info.run_id}")

    version = info.registered_model_version
    client = WorkspaceClient()

    client.registered_models.set_alias(
        name=args.model_name,
        alias="Champion",
        version=version,
    )

    served = ServedModelInput(
        model_name=args.model_name,
        model_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )

    endpoint_names = [e.name for e in client.serving_endpoints.list()]
    if args.serving_endpoint in endpoint_names:
        client.serving_endpoints.update_config(
            name=args.serving_endpoint,
            served_models=[served],
        )
    else:
        client.serving_endpoints.create(
            name=args.serving_endpoint,
            config=EndpointCoreConfigInput(
                name=args.serving_endpoint,
                served_models=[served],
            ),
        )

    print(f"Registered {args.model_name} version {version}")
    print(f"Champion -> version {version}")
    print(f"Serving endpoint -> {args.serving_endpoint}")


if __name__ == "__main__":
    main()
