from __future__ import annotations

import argparse
import importlib.util
import shutil
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)
from mlflow.models.resources import DatabricksServingEndpoint

from model_wheel_demo.agent import build_agent


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around the packaged Databricks agent."""

    def __init__(self, llm_model: str):
        self.llm_model = llm_model

    def predict(
        self,
        context: Any,
        model_input: Any,
    ) -> list[str]:
        agent = build_agent(self.llm_model)
        records = (
            model_input.to_dict(orient="records")
            if hasattr(model_input, "to_dict")
            else model_input
        )
        responses: list[str] = []
        for record in records:
            prompt = record.get("input", "")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "each model input must contain a non-empty 'input' string"
                )
            result = agent.invoke({"input": prompt})
            responses.append(str(result.get("output", result)))
        return responses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--serving-endpoint", required=True)
    return parser.parse_args()


def package_source_path() -> str:
    """Copy only this project's import package into a compact MLflow code path."""
    spec = importlib.util.find_spec("model_wheel_demo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("model_wheel_demo is not installed in the job environment")

    package_dir = Path(next(iter(spec.submodule_search_locations)))
    if not package_dir.is_dir():
        raise RuntimeError(f"Package source directory not found: {package_dir}")

    code_root = Path(tempfile.mkdtemp(prefix="model-wheel-code-"))
    shutil.copytree(package_dir, code_root / "model_wheel_demo")
    return str(code_root)


def main() -> None:
    args = parse_args()
    print(f"Registering {args.model_name} using LLM endpoint {args.llm_model}")

    mlflow.set_registry_uri("databricks-uc")
    with mlflow.start_run() as run:
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=AgentModel(args.llm_model),
            registered_model_name=args.model_name,
            input_example=[{"input": "What is MLflow?"}],
            code_paths=[package_source_path()],
            resources=[
                DatabricksServingEndpoint(endpoint_name=args.llm_model),
            ],
            pip_requirements=[
                "databricks-langchain==0.6.0",
                "langchain==0.3.14",
                "langchain-core==0.3.29",
            ],
        )
        print(f"MLflow run: {run.info.run_id}")

    version = info.registered_model_version
    if version is None:
        raise RuntimeError("MLflow did not return a registered model version")

    client = WorkspaceClient()
    served_model = ServedModelInput(
        model_name=args.model_name,
        model_version=version,
        workload_size=ServedModelInputWorkloadSize.SMALL,
        scale_to_zero_enabled=True,
    )

    endpoint_names = {endpoint.name for endpoint in client.serving_endpoints.list()}
    if args.serving_endpoint in endpoint_names:
        client.serving_endpoints.update_config(
            name=args.serving_endpoint,
            served_models=[served_model],
        )
    else:
        client.serving_endpoints.create(
            name=args.serving_endpoint,
            config=EndpointCoreConfigInput(
                name=args.serving_endpoint,
                served_models=[served_model],
            ),
        )

    # Promote only after the serving endpoint create/update request was accepted.
    client.registered_models.set_alias(args.model_name, "Champion", version)

    print(
        f"Deployment submitted: model={args.model_name}, version={version}, "
        f"Champion={version}, endpoint={args.serving_endpoint}"
    )


if __name__ == "__main__":
    main()
