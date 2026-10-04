from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
)

from model_wheel_demo.agent import build_agent


class AgentModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper around the packaged Databricks agent."""

    def __init__(self, llm_model: str):
        self.llm_model = llm_model

    def predict(self, context, model_input):
        print("Starting agent prediction...", flush=True)

        agent = build_agent(self.llm_model)

        if isinstance(model_input, dict):
            prompt = model_input.get("input", "")
        else:
            prompt = model_input

        if not prompt:
            raise ValueError(
                "model input must contain a non-empty 'input' value"
            )

        result = agent.invoke({"input": prompt})

        print("Agent prediction completed.", flush=True)

        return result.get("output", result)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-name",
        required=True,
    )

    parser.add_argument(
        "--llm-model",
        required=True,
    )

    parser.add_argument(
        "--serving-endpoint",
        required=True,
    )

    parser.add_argument(
        "--wheel-path",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

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
        raise FileNotFoundError(
            f"Wheel not found: {wheel_path}"
        )

    print(
        f"Wheel found: {wheel_path}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. Configure Unity Catalog Model Registry
    # ------------------------------------------------------------------
    print(
        "Configuring MLflow Unity Catalog registry...",
        flush=True,
    )

    mlflow.set_registry_uri("databricks-uc")

    print(
        "MLflow registry URI configured: databricks-uc",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Log model to MLflow and register in Unity Catalog
    # ------------------------------------------------------------------
    print(
        "Starting MLflow run...",
        flush=True,
    )

    with mlflow.start_run() as run:
        print(
            f"MLflow run started: {run.info.run_id}",
            flush=True,
        )

        print(
            "Logging pyfunc model to MLflow...",
            flush=True,
        )

        info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=AgentModel(args.llm_model),
            registered_model_name=args.model_name,
            input_example={
                "input": "What is MLflow?"
            },
            code_paths=[
                str(wheel_path)
            ],
            extra_pip_requirements=[
                f"code/{wheel_path.name}"
            ],
        )

        print(
            "MLflow model logging completed.",
            flush=True,
        )

        print(
            f"MLflow run ID: {run.info.run_id}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 4. Get registered model version
    # ------------------------------------------------------------------
    version = info.registered_model_version

    print(
        "Unity Catalog model registered successfully.",
        flush=True,
    )

    print(
        "Model : {args.model_name}",
        flush=True,
    )

    print(
        "Version : {version}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 5. Create Databricks Workspace client
    # ------------------------------------------------------------------
    print(
        "Creating Databricks Workspace client...",
        flush=True,
    )

    client = WorkspaceClient()

    print(
        "Workspace client created.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 6. Set Champion alias
    # ------------------------------------------------------------------
    print(
        "Setting Champion alias...",
        flush=True,
    )

    client.registered_models.set_alias(
        name=args.model_name,
        alias="Champion",
        version=version,
    )

    print(
        "Champion -> version {version}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 7. Prepare serving configuration
    # ------------------------------------------------------------------
    print(
        "Preparing Model Serving configuration...",
        flush=True,
    )

    served = ServedModelInput(
        model_name=args.model_name,
        model_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
    )

    # ------------------------------------------------------------------
    # 8. Check whether endpoint already exists
    # ------------------------------------------------------------------
    print(
        "Checking existing serving endpoints...",
        flush=True,
    )

    endpoint_names = [
        endpoint.name
        for endpoint in client.serving_endpoints.list()
    ]

    print(
        "Found {len(endpoint_names)} serving endpoint(s).",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 9. Create or update serving endpoint
    # ------------------------------------------------------------------
    if args.serving_endpoint in endpoint_names:
        print(
            "Serving endpoint '{args.serving_endpoint}' already exists.",
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
            "Serving endpoint '{args.serving_endpoint}' does not exist.",
            flush=True,
        )

        print(
            "Creating serving endpoint...",
            flush=True,
        )

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

    print(
        "Registered model : {args.model_name}",
        flush=True,
    )

    print(
        "Model version    : {version}",
        flush=True,
    )

    print(
        "Champion         : version {version}",
        flush=True,
    )

    print(
        "Serving endpoint : {args.serving_endpoint}",
        flush=True,
    )


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import argparse
# from pathlib import Path

# import mlflow
# from databricks.sdk import WorkspaceClient
# from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# from model_wheel_demo.agent import build_agent


# class AgentModel(mlflow.pyfunc.PythonModel):
#     """MLflow pyfunc wrapper around the packaged Databricks agent."""

#     def __init__(self, llm_model: str):
#         self.llm_model = llm_model

#     def predict(self, context, model_input):
#         agent = build_agent(self.llm_model)

#         if isinstance(model_input, dict):
#             prompt = model_input.get("input", "")
#         else:
#             prompt = model_input

#         if not prompt:
#             raise ValueError("model input must contain a non-empty 'input' value")

#         result = agent.invoke({"input": prompt})
#         return result.get("output", result)


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--model-name", required=True)
#     p.add_argument("--llm-model", required=True)
#     p.add_argument("--serving-endpoint", required=True)
#     p.add_argument("--wheel-path", required=True)
#     return p.parse_args()


# def main():
#     args = parse_args()
#     wheel_path = Path(args.wheel_path)

#     if not wheel_path.is_file():
#         raise FileNotFoundError(f"Wheel not found: {wheel_path}")

#     mlflow.set_registry_uri("databricks-uc")

#     print("Starting MLflow model registration...", flush=True)
#     with mlflow.start_run() as run:
#         print("Logging model to MLflow...", flush=True)

#         info = mlflow.pyfunc.log_model(
#             artifact_path="agent",
#             python_model=AgentModel(args.llm_model),
#             registered_model_name=args.model_name,
#             input_example={"input": "What is MLflow?"},
#             code_paths=[str(wheel_path)],
#             extra_pip_requirements=[f"code/{wheel_path.name}"],
#         )
#         print("MLflow model logging completed.", flush=True)
#         print(f"MLflow run: {run.info.run_id}", flush=True)

#     version = info.registered_model_version

#     print(f"Registered model version: {version}", flush=True)
#     print("Setting Champion alias...", flush=True
    
#     client = WorkspaceClient()

#     client.registered_models.set_alias(
#         name=args.model_name,
#         alias="Champion",
#         version=version,
#     )
#     print("Champion alias set.", flush=True)
#     print("Creating/updating serving endpoint...", flush=True)
    
#     served = ServedModelInput(
#         model_name=args.model_name,
#         model_version=version,
#         workload_size="Small",
#         scale_to_zero_enabled=True,
#     )

#     endpoint_names = [e.name for e in client.serving_endpoints.list()]
#     if args.serving_endpoint in endpoint_names:
#         client.serving_endpoints.update_config(
#             name=args.serving_endpoint,
#             served_models=[served],
#         )
#     else:
#         client.serving_endpoints.create(
#             name=args.serving_endpoint,
#             config=EndpointCoreConfigInput(
#                 name=args.serving_endpoint,
#                 served_models=[served],
#             ),
#         )

#     print(f"Registered {args.model_name} version {version}")
#     print(f"Champion -> version {version}")
#     print(f"Serving endpoint -> {args.serving_endpoint}")


# if __name__ == "__main__":
#     main()
