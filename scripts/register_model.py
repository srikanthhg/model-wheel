from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

from model_wheel_demo.agent import build_agent


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around the packaged Databricks agent."""

    def __init__(self, llm_model: str):
        self.llm_model = llm_model

    def predict(self, context, model_input):
        agent = build_agent(self.llm_model)

        if isinstance(model_input, dict):
            prompt = model_input.get("input", "")
        else:
            prompt = model_input

        if not prompt:
            raise ValueError("model input must contain a non-empty 'input' value")

        result = agent.invoke({"input": prompt})
        return result.get("output", result)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True)
    p.add_argument("--llm-model", required=True)
    p.add_argument("--serving-endpoint", required=True)
    p.add_argument("--wheel-path", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    wheel_path = Path(args.wheel_path)

    if not wheel_path.is_file():
        raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    mlflow.set_registry_uri("databricks-uc")

    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=AgentModel(args.llm_model),
            registered_model_name=args.model_name,
            input_example={"input": "What is MLflow?"},
            code_paths=[str(wheel_path)],
            extra_pip_requirements=[f"code/{wheel_path.name}"],
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
