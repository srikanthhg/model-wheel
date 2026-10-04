from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadSize,
)

from model_wheel_demo.agent import build_agent


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around the packaged Databricks agent."""

    def __init__(self, llm_model: str):
        self.llm_model = llm_model

    def predict(
        self,
        context: Any,
        model_input: list[dict[str, Any]],
    ) -> list[str]:
        print("Starting agent prediction...", flush=True)

        agent = build_agent(self.llm_model)

        # MLflow can provide a pandas DataFrame to predict() when the model
        # is invoked through the normal pyfunc interface. Normalize both
        # DataFrame and list-of-dicts inputs to the same internal format.
        if hasattr(model_input, "to_dict"):
            records = model_input.to_dict(orient="records")
        else:
            records = model_input

        responses: list[str] = []

        for record in records:
            prompt = record.get("input", "")

            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    "each model input must contain a non-empty 'input' string"
                )

            result = agent.invoke({"input": prompt})
            output = result.get("output", result)

            responses.append(str(output))

        print("Agent prediction completed.", flush=True)

        return responses


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-name", required=True)
    parser.add_argument("--llm-model", required=True)
    parser.add_argument("--serving-endpoint", required=True)
    parser.add_argument("--wheel-path", required=True)

    return parser.parse_args()


def get_parameters():
    """
    Get deployment parameters.

    Databricks executes this .py file as a notebook, so notebook_task
    parameters are exposed through dbutils.widgets rather than sys.argv.
    The argparse fallback keeps the script usable outside Databricks.
    """
    try:
        dbutils
    except NameError:
        return parse_args()

    dbutils.widgets.text("model-name", "")
    dbutils.widgets.text("llm-model", "")
    dbutils.widgets.text("serving-endpoint", "")
    dbutils.widgets.text("wheel-path", "")

    args = argparse.Namespace(
        model_name=dbutils.widgets.get("model-name"),
        llm_model=dbutils.widgets.get("llm-model"),
        serving_endpoint=dbutils.widgets.get("serving-endpoint"),
        wheel_path=dbutils.widgets.get("wheel-path"),
    )

    required = {
        "model-name": args.model_name,
        "llm-model": args.llm_model,
        "serving-endpoint": args.serving_endpoint,
        "wheel-path": args.wheel_path,
    }

    missing = [name for name, value in required.items() if not value]

    if missing:
        raise ValueError(
            "Missing Databricks notebook parameters: "
            + ", ".join(missing)
        )

    return args


def main():
    args = get_parameters()

    print("========================================", flush=True)
    print("Starting Databricks MLflow deployment", flush=True)
    print("========================================", flush=True)

    print(f"Model name       : {args.model_name}", flush=True)
    print(f"LLM model        : {args.llm_model}", flush=True)
    print(f"Serving endpoint : {args.serving_endpoint}", flush=True)
    print(f"Wheel path       : {args.wheel_path}", flush=True)

    # ------------------------------------------------------------------
    # 1. Validate wheel
    # ------------------------------------------------------------------
    wheel_path = Path(args.wheel_path)

    print("Checking wheel...", flush=True)

    if not wheel_path.is_file():
        raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    print(f"Wheel found: {wheel_path}", flush=True)

    # ------------------------------------------------------------------
    # 2. Configure Unity Catalog Model Registry
    # ------------------------------------------------------------------
    print("Configuring MLflow Unity Catalog registry...", flush=True)

    mlflow.set_registry_uri("databricks-uc")

    print(
        "MLflow registry URI configured: databricks-uc",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Log model to MLflow and register in Unity Catalog
    # ------------------------------------------------------------------
    print("Starting MLflow run...", flush=True)

    with mlflow.start_run() as run:
        print(f"MLflow run started: {run.info.run_id}", flush=True)

        print("Logging pyfunc model to MLflow...", flush=True)

        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=AgentModel(args.llm_model),
            registered_model_name=args.model_name,
            input_example=[
                {"input": "What is MLflow?"}
            ],
            code_paths=[str(wheel_path)],
            resources=[
                DatabricksServingEndpoint(endpoint_name=args.llm_model),
            ],
            pip_requirements=[
                f"code/{wheel_path.name}",
                "databricks-langchain==0.6.0",
                "langchain==0.3.14",
                "langchain-core==0.3.29",
            ],
        )

        print("MLflow model logging completed.", flush=True)
        print(f"MLflow run ID: {run.info.run_id}", flush=True)

    # ------------------------------------------------------------------
    # 4. Get registered model version
    # ------------------------------------------------------------------
    version = info.registered_model_version

    if version is None:
        raise RuntimeError(
            "MLflow did not return a registered model version."
        )

    print("Unity Catalog model registered successfully.", flush=True)
    print(f"Model   : {args.model_name}", flush=True)
    print(f"Version : {version}", flush=True)

    # ------------------------------------------------------------------
    # 5. Create Databricks Workspace client
    # ------------------------------------------------------------------
    print("Creating Databricks Workspace client...", flush=True)

    client = WorkspaceClient()

    print("Workspace client created.", flush=True)

    # ------------------------------------------------------------------
    # 6. Set Champion alias
    # ------------------------------------------------------------------
    print("Setting Champion alias...", flush=True)

    client.registered_models.set_alias(
        args.model_name,
        "Champion",
        version,
    )

    print(f"Champion -> version {version}", flush=True)

    # ------------------------------------------------------------------
    # 7. Prepare serving configuration
    # ------------------------------------------------------------------
    print("Preparing Model Serving configuration...", flush=True)

    served = ServedModelInput(
        model_name=args.model_name,
        model_version=version,
        workload_size=ServedModelInputWorkloadSize.SMALL,
        scale_to_zero_enabled=True,
    )

    # ------------------------------------------------------------------
    # 8. Check whether endpoint already exists
    # ------------------------------------------------------------------
    print("Checking existing serving endpoints...", flush=True)

    endpoint_names = [
        endpoint.name
        for endpoint in client.serving_endpoints.list()
    ]

    print(
        f"Found {len(endpoint_names)} serving endpoint(s).",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 9. Create or update serving endpoint
    # ------------------------------------------------------------------
    if args.serving_endpoint in endpoint_names:
        print(
            f"Serving endpoint '{args.serving_endpoint}' already exists.",
            flush=True,
        )

        print(
            "Updating serving endpoint configuration...",
            flush=True,
        )

        client.serving_endpoints.update_config(
            name=args.serving_endpoint,
            served_models=[served],
        )

        print(
            "Serving endpoint update request submitted.",
            flush=True,
        )

    else:
        print(
            f"Serving endpoint '{args.serving_endpoint}' does not exist.",
            flush=True,
        )

        print("Creating serving endpoint...", flush=True)

        client.serving_endpoints.create(
            name=args.serving_endpoint,
            config=EndpointCoreConfigInput(
                name=args.serving_endpoint,
                served_models=[served],
            ),
        )

        print(
            "Serving endpoint creation request submitted.",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 10. Deployment summary
    # ------------------------------------------------------------------
    print("========================================", flush=True)
    print("Deployment request completed", flush=True)
    print("========================================", flush=True)

    print(f"Registered model : {args.model_name}", flush=True)
    print(f"Model version    : {version}", flush=True)
    print(f"Champion         : version {version}", flush=True)
    print(f"Serving endpoint : {args.serving_endpoint}", flush=True)


if __name__ == "__main__":
    main()
