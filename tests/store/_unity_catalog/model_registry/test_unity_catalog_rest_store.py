import json
from itertools import combinations
from unittest import mock

import pytest
import yaml

from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry import (
    RegisteredModelTag,
)
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.run import Run
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_tag import RunTag
from mlflow.exceptions import MlflowException, RestException
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature, Schema
from mlflow.prompt.constants import (
    PROMPT_MODEL_CONFIG_TAG_KEY,
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TAG_KEY,
    PROMPT_TYPE_TEXT,
    RESPONSE_FORMAT_TAG_KEY,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    DeploymentJobConnection,
    UcRegisteredModelInfo,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    LinkPromptsToTracesRequest,
    LinkPromptVersionsToModelsRequest,
    LinkPromptVersionsToRunsRequest,
)
from mlflow.store._unity_catalog.registry.enriched_rest_store import (
    UcEnrichedModelRegistryStore,
)
from mlflow.tracing.constant import TraceTagKey
from mlflow.types.schema import ColSpec, DataType
from mlflow.utils._unity_catalog_utils import (
    _ACTIVE_CATALOG_QUERY,
    _ACTIVE_SCHEMA_QUERY,
)
from mlflow.utils.proto_json_utils import message_to_json

from tests.helper_functions import mock_http_200
from tests.store._unity_catalog.conftest import (
    _REGISTRY_HOST_CREDS,
)


@pytest.fixture
def store(mock_databricks_uc_host_creds):
    with mock.patch("mlflow.utils.databricks_utils.get_databricks_host_creds"):
        yield UcEnrichedModelRegistryStore(store_uri="databricks-uc", tracking_uri="databricks")


@pytest.fixture
def spark_session(request):
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store._get_active_spark_session"
    ) as spark_session_getter:
        spark = mock.MagicMock()
        spark_session_getter.return_value = spark

        # Define a custom side effect function for spark sql queries
        def sql_side_effect(query):
            if query == _ACTIVE_CATALOG_QUERY:
                catalog_response_mock = mock.MagicMock()
                catalog_response_mock.collect.return_value = [{"catalog": request.param}]
                return catalog_response_mock
            elif query == _ACTIVE_SCHEMA_QUERY:
                schema_response_mock = mock.MagicMock()
                schema_response_mock.collect.return_value = [{"schema": "default"}]
                return schema_response_mock
            else:
                raise ValueError(f"Unexpected query: {query}")

        spark.sql.side_effect = sql_side_effect
        yield spark


def _args(endpoint, method, json_body, host_creds, extra_headers):
    res = {
        "host_creds": host_creds,
        "endpoint": f"/api/2.0/mlflow/unity-catalog/{endpoint}",
        "method": method,
        "extra_headers": extra_headers,
    }
    if extra_headers is None:
        del res["extra_headers"]
    if method == "GET":
        res["params"] = json.loads(json_body)
    else:
        res["json"] = json.loads(json_body)
    return res


def _verify_requests(
    http_request,
    endpoint,
    method,
    proto_message,
    host_creds=_REGISTRY_HOST_CREDS,
    extra_headers=None,
):
    json_body = message_to_json(proto_message)
    call_args = _args(endpoint, method, json_body, host_creds, extra_headers)
    http_request.assert_any_call(**(call_args))


def _expected_unsupported_method_error_message(method):
    return f"Method '{method}' is unsupported for models in the Unity Catalog"


def _expected_unsupported_arg_error_message(arg):
    return f"Argument '{arg}' is unsupported for models in the Unity Catalog"


def test_create_registered_model_three_level_name_hint(store):
    # The native create flow validates the three-level UC name client-side (before issuing any
    # request) and raises with a legacy-registry hint when the name is not catalog.schema.model.
    with pytest.raises(MlflowException, match="three levels") as exc_info:
        store.create_registered_model(name="invalid_model")

    error_message = str(exc_info.value)
    assert "Not a valid Unity Catalog model name: 'invalid_model'" in error_message
    assert "set the Model Registry URI to 'databricks' (legacy) instead of" in error_message


def test_create_registered_model_two_level_name_hint(store):
    # A two-level name is still not a valid UC (three-level) model name.
    with pytest.raises(MlflowException, match="three levels") as exc_info:
        store.create_registered_model(name="schema.invalid_model")

    error_message = str(exc_info.value)
    assert "Not a valid Unity Catalog model name: 'schema.invalid_model'" in error_message
    # Should not have double periods
    assert ". ." not in error_message


def test_create_registered_model_metastore_does_not_exist_hint(store):
    """
    Test that creating a registered model when the metastore doesn't exist
    provides a legacy registry hint. The name is three-level so it passes
    client-side validation and reaches the native endpoint call.
    """
    original_error_message = "METASTORE_DOES_NOT_EXIST: Metastore not found"
    rest_exception = RestException({
        "error_code": "METASTORE_DOES_NOT_EXIST",
        "message": original_error_message,
    })

    with mock.patch.object(store, "_edit_endpoint_and_call", side_effect=rest_exception):
        with pytest.raises(MlflowException, match="METASTORE_DOES_NOT_EXIST") as exc_info:
            store.create_registered_model(name="catalog.schema.test_model")

    # Verify the exception message includes the original error and the legacy registry hint
    expected_hint = (
        "If you are trying to use the Model Registry in a Databricks workspace"
        " that does not have Unity Catalog enabled, either enable Unity Catalog in"
        " the workspace (recommended) or set the Model Registry URI to 'databricks'"
        " to use the legacy Workspace Model Registry."
    )
    error_message = str(exc_info.value)
    assert original_error_message in error_message
    assert expected_hint in error_message


def test_create_registered_model_other_rest_exceptions_not_modified(store):
    """
    Test that RestExceptions unrelated to bad UC model names are not modified
    and are re-raised as-is.
    """
    original_error_message = "Some other error"
    rest_exception = RestException({
        "error_code": "INTERNAL_ERROR",
        "message": original_error_message,
    })

    with mock.patch.object(store, "_edit_endpoint_and_call", side_effect=rest_exception):
        with pytest.raises(RestException, match=original_error_message) as exc_info:
            store.create_registered_model(name="catalog.schema.some_model")

    # Verify the original RestException is re-raised without modification
    assert str(exc_info.value) == "INTERNAL_ERROR: Some other error"


@pytest.fixture
def local_model_dir(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.double)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


@pytest.fixture
def langchain_local_model_dir(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
        "flavors": {
            "langchain": {
                "databricks_dependency": {
                    "databricks_vector_search_index_name": ["index1", "index2"],
                    "databricks_embeddings_endpoint_name": ["embedding_endpoint"],
                    "databricks_llm_endpoint_name": ["llm_endpoint"],
                    "databricks_chat_endpoint_name": ["chat_endpoint"],
                }
            }
        },
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path


@pytest.fixture(params=[True, False])  # True tests with resources and False tests with auth policy
def langchain_local_model_dir_with_resources(request, tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    if request.param:
        fake_mlmodel_contents = {
            "artifact_path": "some-artifact-path",
            "run_id": "abc123",
            "signature": fake_signature.to_dict(),
            "resources": {
                "databricks": {
                    "serving_endpoint": [
                        {"name": "embedding_endpoint"},
                        {"name": "llm_endpoint"},
                        {"name": "chat_endpoint"},
                    ],
                    "vector_search_index": [{"name": "index1"}, {"name": "index2"}],
                    "function": [
                        {"name": "test.schema.test_function"},
                        {"name": "test.schema.test_function_2"},
                    ],
                    "uc_connection": [{"name": "test_connection"}],
                    "table": [
                        {"name": "test.schema.test_table"},
                        {"name": "test.schema.test_table_2"},
                    ],
                }
            },
        }
    else:
        fake_mlmodel_contents = {
            "artifact_path": "some-artifact-path",
            "run_id": "abc123",
            "signature": fake_signature.to_dict(),
            "auth_policy": {
                "system_auth_policy": {
                    "resources": {
                        "databricks": {
                            "serving_endpoint": [
                                {"name": "embedding_endpoint"},
                                {"name": "llm_endpoint"},
                                {"name": "chat_endpoint"},
                            ],
                            "vector_search_index": [{"name": "index1"}, {"name": "index2"}],
                            "function": [
                                {"name": "test.schema.test_function"},
                                {"name": "test.schema.test_function_2"},
                            ],
                            "uc_connection": [{"name": "test_connection"}],
                            "table": [
                                {"name": "test.schema.test_table"},
                                {"name": "test.schema.test_table_2"},
                            ],
                        }
                    },
                },
                "user_auth_policy": {
                    "api_scopes": [
                        "serving.serving-endpoints,vectorsearch.vector-search-endpoints,vectorsearch.vector-search-indexes"
                    ]
                },
            },
        }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)

    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
        {"type": "DATABRICKS_UC_FUNCTION", "name": "test.schema.test_function"},
        {"type": "DATABRICKS_UC_FUNCTION", "name": "test.schema.test_function_2"},
        {"type": "DATABRICKS_UC_CONNECTION", "name": "test_connection"},
        {"type": "DATABRICKS_TABLE", "name": "test.schema.test_table"},
        {"type": "DATABRICKS_TABLE", "name": "test.schema.test_table_2"},
    ]

    return (tmp_path, model_version_dependencies)


@pytest.fixture
def langchain_local_model_dir_with_invoker_resources(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
        "resources": {
            "databricks": {
                "serving_endpoint": [
                    {"name": "embedding_endpoint"},
                    {"name": "llm_endpoint"},
                    {"name": "chat_endpoint"},
                ],
                "vector_search_index": [
                    {"name": "index1", "on_behalf_of_user": False},
                    {"name": "index2"},
                ],
                "function": [
                    {"name": "test.schema.test_function", "on_behalf_of_user": True},
                    {"name": "test.schema.test_function_2"},
                ],
                "uc_connection": [{"name": "test_connection", "on_behalf_of_user": False}],
                "table": [
                    {"name": "test.schema.test_table", "on_behalf_of_user": True},
                    {"name": "test.schema.test_table_2"},
                ],
            }
        },
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)

    model_version_dependencies = [
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index1"},
        {"type": "DATABRICKS_VECTOR_INDEX", "name": "index2"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "embedding_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "llm_endpoint"},
        {"type": "DATABRICKS_MODEL_ENDPOINT", "name": "chat_endpoint"},
        {"type": "DATABRICKS_UC_FUNCTION", "name": "test.schema.test_function_2"},
        {"type": "DATABRICKS_UC_CONNECTION", "name": "test_connection"},
        {"type": "DATABRICKS_TABLE", "name": "test.schema.test_table_2"},
    ]

    return (tmp_path, model_version_dependencies)


@pytest.fixture
def langchain_local_model_dir_no_dependencies(tmp_path):
    fake_signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.string)]), outputs=Schema([ColSpec(DataType.string)])
    )
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": fake_signature.to_dict(),
        "flavors": {"langchain": {}},
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path




_TEST_SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec(DataType.double)]), outputs=Schema([ColSpec(DataType.double)])
)


@pytest.fixture
def feature_store_local_model_dir(tmp_path):
    fake_mlmodel_contents = {
        "artifact_path": "some-artifact-path",
        "run_id": "abc123",
        "signature": _TEST_SIGNATURE.to_dict(),
        "flavors": {"python_function": {"loader_module": "databricks.feature_store.mlflow_model"}},
    }
    with open(tmp_path.joinpath(MLMODEL_FILE_NAME), "w") as handle:
        yaml.dump(fake_mlmodel_contents, handle)
    return tmp_path






















def test_search_registered_models_invalid_args(store):
    params_list = [
        {"filter_string": "model = 'yo'"},
        {"order_by": ["x", "Y"]},
    ]
    # test all combination of invalid params
    for sz in range(1, 3):
        for combination in combinations(params_list, sz):
            params = {k: v for d in combination for k, v in d.items()}
            with pytest.raises(
                MlflowException, match="unsupported for models in the Unity Catalog"
            ):
                store.search_registered_models(**params)


def test_get_latest_versions_unsupported(store):
    name = "model_1"
    with pytest.raises(
        MlflowException,
        match=f"{_expected_unsupported_method_error_message('get_latest_versions')}. "
        "To load the latest version of a model in Unity Catalog, you can set "
        "an alias on the model version and load it by alias",
    ):
        store.get_latest_versions(name=name)
    with pytest.raises(
        MlflowException,
        match=f"{_expected_unsupported_method_error_message('get_latest_versions')}. "
        "Detected attempt to load latest model version in stages",
    ):
        store.get_latest_versions(name=name, stages=["Production"])






















def _get_workspace_id_for_run(run_id=None):
    return "123" if run_id is not None else None














@pytest.mark.parametrize("spark_session", ["main"], indirect=True)  # set the catalog name to "main"
def test_store_uses_catalog_and_schema_from_spark_session(spark_session, store):
    with mock.patch.object(
        store, "_edit_endpoint_and_call", return_value=UcRegisteredModelInfo()
    ) as native_call:
        store.get_registered_model(name="model_1")
    spark_session.sql.assert_any_call(_ACTIVE_CATALOG_QUERY)
    spark_session.sql.assert_any_call(_ACTIVE_SCHEMA_QUERY)
    assert spark_session.sql.call_count == 2
    assert native_call.call_args.kwargs["full_name_arg"] == "main.default.model_1"


@pytest.mark.parametrize("spark_session", ["main"], indirect=True)
def test_store_uses_catalog_from_spark_session(spark_session, store):
    with mock.patch.object(
        store, "_edit_endpoint_and_call", return_value=UcRegisteredModelInfo()
    ) as native_call:
        store.get_registered_model(name="default.model_1")
    spark_session.sql.assert_any_call(_ACTIVE_CATALOG_QUERY)
    assert spark_session.sql.call_count == 1
    assert native_call.call_args.kwargs["full_name_arg"] == "main.default.model_1"


@pytest.mark.parametrize("spark_session", ["hive_metastore", "spark_catalog"], indirect=True)
def test_store_ignores_hive_metastore_default_from_spark_session(spark_session, store):
    with mock.patch.object(
        store, "_edit_endpoint_and_call", return_value=UcRegisteredModelInfo()
    ) as native_call:
        store.get_registered_model(name="model_1")
    spark_session.sql.assert_any_call(_ACTIVE_CATALOG_QUERY)
    assert spark_session.sql.call_count == 1
    assert native_call.call_args.kwargs["full_name_arg"] == "model_1"






@mock_http_200
def test_create_and_update_registered_model_print_job_url(mock_http, store):
    # UC model names must be three-level; the native create/update flow rejects non-UC names.
    name = "catalog.schema.model_for_job_url_test"
    description = "test model with job id"
    deployment_job_id = "123"

    with (
        mock.patch(
            "mlflow.store._unity_catalog.registry.enriched_rest_store._print_databricks_deployment_job_url",
        ) as mock_print_url,
    ):
        # Should not print the job url when the deployment job id is None
        store.create_registered_model(name=name, description=description, deployment_job_id=None)
        store.create_registered_model(name=name, description=description, deployment_job_id="")
        mock_print_url.assert_not_called()
        # Should print the job url when the deployment job id is not None
        store.create_registered_model(
            name=name, description=description, deployment_job_id=deployment_job_id
        )
        mock_print_url.assert_called_once_with(model_name=name, job_id=deployment_job_id)
        mock_print_url.reset_mock()

        # Should not print the job url when the deployment job id is false-y
        store.update_registered_model(name=name, description=description, deployment_job_id=None)
        store.update_registered_model(name=name, description=description, deployment_job_id="")
        mock_print_url.assert_not_called()
        # Should print the job url when the deployment job id is not None
        store.update_registered_model(
            name=name, description=description, deployment_job_id=deployment_job_id
        )
        mock_print_url.assert_called_once_with(model_name=name, job_id=deployment_job_id)


@mock_http_200
def test_create_prompt_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    description = "A test prompt"
    tags = {"foo": "bar"}
    # Patch proto_info_to_mlflow_prompt_info to return a dummy Prompt
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_info_to_mlflow_prompt_info",
        return_value=Prompt(name=name, description=description, tags=tags),
    ) as proto_to_prompt:
        store.create_prompt(name=name, description=description, tags=tags)
        # Check that the endpoint was called correctly
        assert any(c[1]["endpoint"].endswith("/prompts") for c in mock_http.call_args_list)
        proto_to_prompt.assert_called()


@mock_http_200
def test_search_prompts_uc(mock_http, store, monkeypatch):
    # Patch proto_info_to_mlflow_prompt_info to return a dummy Prompt
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_info_to_mlflow_prompt_info",
        return_value=Prompt(name="prompt1", description="test prompt"),
    ) as proto_to_prompt:
        store.search_prompts(filter_string="catalog = 'test_catalog' AND schema = 'test_schema'")
        # Should call the correct endpoint for SearchPromptsRequest
        assert any("/prompts" in c[1]["endpoint"] for c in mock_http.call_args_list)
        # The utility function should NOT be called when there are no results (empty list)
        assert proto_to_prompt.call_count == 0  # Correct behavior for empty results


def test_search_prompts_with_results_uc(store, monkeypatch):
    # Create mock protobuf objects
    from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
        Prompt as ProtoPrompt,
    )
    from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
        PromptTag as ProtoPromptTag,
    )
    from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
        SearchPromptsResponse,
    )

    # Create mock prompt data
    mock_prompt_1 = ProtoPrompt(
        name="test_prompt_1",
        description="First test prompt",
        tags=[ProtoPromptTag(key="env", value="dev")],
    )
    mock_prompt_2 = ProtoPrompt(name="test_prompt_2", description="Second test prompt")

    # Create mock response
    mock_response = SearchPromptsResponse(
        prompts=[mock_prompt_1, mock_prompt_2], next_page_token="next_token_123"
    )

    # Expected conversion results
    expected_prompts = [
        Prompt(name="test_prompt_1", description="First test prompt", tags={"env": "dev"}),
        Prompt(name="test_prompt_2", description="Second test prompt", tags={}),
    ]

    with (
        mock.patch.object(store, "_call_endpoint", return_value=mock_response),
        mock.patch(
            "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_info_to_mlflow_prompt_info",
            side_effect=expected_prompts,
        ) as mock_converter,
    ):
        # Call search_prompts
        result = store.search_prompts(
            max_results=10, filter_string="catalog = 'test_catalog' AND schema = 'test_schema'"
        )

        # Verify conversion function was called twice (once for each result)
        assert mock_converter.call_count == 2

        # Verify the results
        assert len(result) == 2
        assert result.token == "next_token_123"
        assert result[0].name == "test_prompt_1"
        assert result[1].name == "test_prompt_2"


@mock_http_200
def test_delete_prompt_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    store.delete_prompt(name=name)
    # Should call the correct endpoint for DeletePromptRequest
    assert any("/prompts" in c[1]["endpoint"] for c in mock_http.call_args_list)


@mock_http_200
def test_set_prompt_tag_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    key = "env"
    value = "prod"
    store.set_prompt_tag(name=name, key=key, value=value)
    # Should call the exact endpoint for SetPromptTagRequest with substituted path
    expected_endpoint = f"/api/2.0/mlflow/unity-catalog/prompts/{name}/tags"
    assert any(c[1]["endpoint"] == expected_endpoint for c in mock_http.call_args_list)


@mock_http_200
def test_delete_prompt_tag_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    key = "env"
    store.delete_prompt_tag(name=name, key=key)
    # Should call the exact endpoint for DeletePromptTagRequest with substituted path
    expected_endpoint = f"/api/2.0/mlflow/unity-catalog/prompts/{name}/tags/{key}"
    assert any(c[1]["endpoint"] == expected_endpoint for c in mock_http.call_args_list)


@mock_http_200
def test_create_prompt_version_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = "Hello {name}!"
    description = "A greeting prompt"
    tags = {"env": "test"}
    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name, version=1, template=template, commit_message=description, tags=tags
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name, template=template, description=description, tags=tags
        )
        # Should call the correct endpoint for CreatePromptVersionRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


@mock_http_200
def test_create_prompt_version_with_response_format_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = "Generate a response for {query}"
    description = "A response generation prompt"
    tags = {"env": "test"}
    response_format = {"type": "object", "properties": {"answer": {"type": "string"}}}

    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name,
            version=1,
            template=template,
            commit_message=description,
            tags=tags,
            response_format=response_format,
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name,
            template=template,
            description=description,
            tags=tags,
            response_format=response_format,
        )

    # Verify the correct endpoint is called
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )
    proto_to_prompt.assert_called()

    # Verify the HTTP request body contains the response_format in tags
    http_call_args = [
        c
        for c in mock_http.call_args_list
        if "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
    ]
    assert len(http_call_args) == 1

    request_body = http_call_args[0][1]["json"]
    prompt_version = request_body["prompt_version"]

    tags_in_request = {tag["key"]: tag["value"] for tag in prompt_version.get("tags", [])}
    assert RESPONSE_FORMAT_TAG_KEY in tags_in_request

    expected_response_format = json.dumps(response_format)
    assert tags_in_request[RESPONSE_FORMAT_TAG_KEY] == expected_response_format
    assert tags_in_request["env"] == "test"
    assert tags_in_request[PROMPT_TYPE_TAG_KEY] == PROMPT_TYPE_TEXT


@mock_http_200
def test_create_prompt_version_with_multi_turn_template_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, can you help me with {task}?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help you with {task}."},
        {"role": "user", "content": "Please provide a detailed explanation."},
    ]
    description = "A multi-turn conversation prompt"
    tags = {"type": "conversation", "env": "test"}

    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name, version=1, template=template, commit_message=description, tags=tags
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name, template=template, description=description, tags=tags
        )

    # Verify the correct endpoint is called
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )
    proto_to_prompt.assert_called()

    # Verify the HTTP request body contains the multi-turn template
    http_call_args = [
        c
        for c in mock_http.call_args_list
        if "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
    ]
    assert len(http_call_args) == 1

    request_body = http_call_args[0][1]["json"]
    prompt_version = request_body["prompt_version"]

    # Verify template was JSON-encoded properly
    template_in_request = json.loads(prompt_version["template"])
    assert template_in_request == template

    tags_in_request = {tag["key"]: tag["value"] for tag in prompt_version.get("tags", [])}
    assert tags_in_request["type"] == "conversation"
    assert tags_in_request["env"] == "test"
    assert tags_in_request[PROMPT_TYPE_TAG_KEY] == PROMPT_TYPE_CHAT


@mock_http_200
def test_create_prompt_version_with_model_config_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    template = "Generate a response for {query}"
    description = "A prompt with model config"
    tags = {"env": "test"}
    model_config = {
        "model_name": "databricks-meta-llama-3-1-70b-instruct",
        "max_tokens": 100,
        "temperature": 0.7,
    }

    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(
            name=name,
            version=1,
            template=template,
            commit_message=description,
            tags=tags,
            model_config=model_config,
        ),
    ) as proto_to_prompt:
        store.create_prompt_version(
            name=name,
            template=template,
            description=description,
            tags=tags,
            model_config=model_config,
        )

    # Verify the correct endpoint is called
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )
    proto_to_prompt.assert_called()

    # Verify the HTTP request body contains the model_config in tags
    http_call_args = [
        c
        for c in mock_http.call_args_list
        if "/prompts/" in c[1]["endpoint"] and "/versions" in c[1]["endpoint"]
    ]
    assert len(http_call_args) == 1

    request_body = http_call_args[0][1]["json"]
    prompt_version = request_body["prompt_version"]

    tags_in_request = {tag["key"]: tag["value"] for tag in prompt_version.get("tags", [])}
    assert PROMPT_MODEL_CONFIG_TAG_KEY in tags_in_request

    expected_model_config = json.dumps(model_config)
    assert tags_in_request[PROMPT_MODEL_CONFIG_TAG_KEY] == expected_model_config
    assert tags_in_request["env"] == "test"
    assert tags_in_request[PROMPT_TYPE_TAG_KEY] == PROMPT_TYPE_TEXT


@mock_http_200
def test_get_prompt_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    # Patch proto_info_to_mlflow_prompt_info to return a dummy Prompt
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_info_to_mlflow_prompt_info",
        return_value=Prompt(name=name, description="test prompt", tags={}),
    ) as proto_to_prompt:
        store.get_prompt(name=name)
        # Should call the correct endpoint for GetPromptRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions/" not in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


@mock_http_200
def test_get_prompt_version_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    version = "1"
    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(name=name, version=1, template="Hello {name}!"),
    ) as proto_to_prompt:
        store.get_prompt_version(name=name, version=version)
        # Should call the correct endpoint for GetPromptVersionRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions/" in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


@mock_http_200
def test_delete_prompt_version_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    version = "1"
    store.delete_prompt_version(name=name, version=version)
    # Should call the correct endpoint for DeletePromptVersionRequest
    assert any(
        "/prompts/" in c[1]["endpoint"] and "/versions/" in c[1]["endpoint"]
        for c in mock_http.call_args_list
    )


@mock_http_200
def test_get_prompt_version_by_alias_uc(mock_http, store, monkeypatch):
    name = "prompt1"
    alias = "latest"
    # Patch proto_to_mlflow_prompt to return a dummy PromptVersion
    with mock.patch(
        "mlflow.store._unity_catalog.registry.enriched_rest_store.proto_to_mlflow_prompt",
        return_value=PromptVersion(name=name, version=1, template="Hello {name}!"),
    ) as proto_to_prompt:
        store.get_prompt_version_by_alias(name=name, alias=alias)
        # Should call the correct endpoint for GetPromptVersionByAliasRequest
        assert any(
            "/prompts/" in c[1]["endpoint"] and "/versions/by-alias/" in c[1]["endpoint"]
            for c in mock_http.call_args_list
        )
        proto_to_prompt.assert_called()


def test_link_prompt_version_to_model_success(store):
    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as mock_edit_call,
        mock.patch.object(store, "_get_endpoint_from_method") as mock_get_endpoint,
        mock.patch(
            "mlflow.store.model_registry.abstract_store.AbstractStore.link_prompt_version_to_model"
        ) as mock_super_call,
    ):
        # Setup
        mock_get_endpoint.return_value = (
            "/api/2.0/mlflow/unity-catalog/prompts/link-to-model",
            "POST",
        )

        # Execute
        store.link_prompt_version_to_model("test_prompt", "1", "model_123")

        # Verify parent method was called
        mock_super_call.assert_called_once_with(
            name="test_prompt", version="1", model_id="model_123"
        )

        # Verify API call was made
        mock_edit_call.assert_called_once()
        call_args = mock_edit_call.call_args

        assert call_args[1]["name"] == "test_prompt"
        assert call_args[1]["version"] == "1"
        assert call_args[1]["model_id"] == "model_123"
        assert call_args[1]["proto_name"] == LinkPromptVersionsToModelsRequest


def test_link_prompt_version_to_model_sets_tag(store):
    # Mock the prompt version
    mock_prompt_version = PromptVersion(
        name="test_prompt",
        version=1,
        template="Test template",
        creation_timestamp=1234567890,
    )

    with (
        mock.patch("mlflow.tracking._get_store") as mock_get_tracking_store,
        mock.patch.object(store, "get_prompt_version", return_value=mock_prompt_version),
    ):
        # Setup mocks
        mock_tracking_store = mock.Mock()
        mock_get_tracking_store.return_value = mock_tracking_store

        # Mock the logged model
        model_id = "model_123"
        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/artifacts",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Mock the UC-specific API call to avoid real API calls
        with (
            mock.patch.object(store, "_edit_endpoint_and_call"),
            mock.patch.object(
                store, "_get_endpoint_from_method", return_value=("/api/test", "POST")
            ),
        ):
            # Execute
            store.link_prompt_version_to_model("test_prompt", "1", model_id)

        # Verify the tag was set
        mock_tracking_store.set_logged_model_tags.assert_called_once()
        call_args = mock_tracking_store.set_logged_model_tags.call_args
        assert call_args[0][0] == model_id

        logged_model_tags = call_args[0][1]
        assert len(logged_model_tags) == 1
        logged_model_tag = logged_model_tags[0]
        assert isinstance(logged_model_tag, LoggedModelTag)
        assert logged_model_tag.key == TraceTagKey.LINKED_PROMPTS

        expected_value = [{"name": "test_prompt", "version": "1"}]
        assert json.loads(logged_model_tag.value) == expected_value


def test_link_prompts_to_trace_success(store):
    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as mock_edit_call,
        mock.patch.object(store, "_get_endpoint_from_method") as mock_get_endpoint,
    ):
        # Setup
        mock_get_endpoint.return_value = (
            "/api/2.0/mlflow/unity-catalog/prompt-versions/links-to-traces",
            "POST",
        )

        prompt_versions = [
            PromptVersion(name="test_prompt", version=1, template="test", creation_timestamp=123)
        ]
        trace_id = "trace_123"

        # Execute
        store.link_prompts_to_trace(prompt_versions, trace_id)

        # Verify API call was made
        mock_edit_call.assert_called_once()
        call_args = mock_edit_call.call_args

        assert call_args[1]["proto_name"] == LinkPromptsToTracesRequest


def test_link_prompt_version_to_run_success(store):
    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as mock_edit_call,
        mock.patch.object(store, "_get_endpoint_from_method") as mock_get_endpoint,
        mock.patch(
            "mlflow.store.model_registry.abstract_store.AbstractStore.link_prompt_version_to_run"
        ) as mock_super_call,
    ):
        # Setup
        mock_get_endpoint.return_value = (
            "/api/2.0/mlflow/unity-catalog/prompt-versions/links-to-runs",
            "POST",
        )

        # Execute
        store.link_prompt_version_to_run("test_prompt", "1", "run_123")

        # Verify parent method was called
        mock_super_call.assert_called_once_with(name="test_prompt", version="1", run_id="run_123")

        # Verify API call was made
        mock_edit_call.assert_called_once()
        call_args = mock_edit_call.call_args

        # Check that _edit_endpoint_and_call was called with correct parameters
        assert (
            call_args[1]["endpoint"]
            == "/api/2.0/mlflow/unity-catalog/prompt-versions/links-to-runs"
        )
        assert call_args[1]["method"] == "POST"
        assert call_args[1]["proto_name"] == LinkPromptVersionsToRunsRequest

        # Verify the request body contains correct prompt and run information
        req_body = json.loads(call_args[1]["req_body"])
        assert len(req_body["prompt_versions"]) == 1
        assert req_body["prompt_versions"][0]["name"] == "test_prompt"
        assert req_body["prompt_versions"][0]["version"] == "1"
        assert req_body["run_ids"] == ["run_123"]


def test_link_prompt_version_to_run_sets_tag(store):
    # Mock the prompt version
    mock_prompt_version = PromptVersion(
        name="test_prompt",
        version=1,
        template="Test template",
        creation_timestamp=1234567890,
    )

    with (
        mock.patch("mlflow.tracking._get_store") as mock_get_tracking_store,
        mock.patch.object(store, "get_prompt_version", return_value=mock_prompt_version),
    ):
        # Setup mocks
        mock_tracking_store = mock.Mock()
        mock_get_tracking_store.return_value = mock_tracking_store

        # Mock the run
        run_id = "run_123"
        run_data = RunData(metrics=[], params=[], tags={})
        run_info = RunInfo(
            run_id=run_id,
            experiment_id="exp_123",
            user_id="user_123",
            status="FINISHED",
            start_time=1234567890,
            end_time=1234567890,
            lifecycle_stage="active",
        )
        run = Run(run_info=run_info, run_data=run_data)
        mock_tracking_store.get_run.return_value = run

        # Mock the UC-specific API call to avoid real API calls
        with (
            mock.patch.object(store, "_edit_endpoint_and_call"),
            mock.patch.object(
                store, "_get_endpoint_from_method", return_value=("/api/test", "POST")
            ),
        ):
            # Execute
            store.link_prompt_version_to_run("test_prompt", "1", run_id)

        # Verify the tag was set
        mock_tracking_store.set_tag.assert_called_once()
        call_args = mock_tracking_store.set_tag.call_args
        assert call_args[0][0] == run_id

        run_tag = call_args[0][1]
        assert isinstance(run_tag, RunTag)
        assert run_tag.key == TraceTagKey.LINKED_PROMPTS

        expected_value = [{"name": "test_prompt", "version": "1"}]
        assert json.loads(run_tag.value) == expected_value


# ---------------------------------------------------------------------------
# Native (/api/2.1/unity-catalog/*) path: governance + enrichment. Selected when
# MLFLOW_ENABLE_UC_NATIVE_MODEL_REGISTRY is set; the legacy store is a separate class.
# ---------------------------------------------------------------------------


def test_get_registered_model_uses_native_endpoint(store):
    native_info = UcRegisteredModelInfo(
        name="model",
        catalog_name="catalog",
        schema_name="schema",
        full_name="catalog.schema.model",
        comment="d",
        deployment_job_id="42",
        deployment_job_state=DeploymentJobConnection.State.Value("CONNECTED"),
    )
    with mock.patch.object(
        store, "_edit_endpoint_and_call", return_value=native_info
    ) as native_call:
        result = store.get_registered_model("catalog.schema.model")
    native_call.assert_called_once()
    assert result.name == "catalog.schema.model"
    assert result.deployment_job_id == "42"
    assert result.deployment_job_state == "CONNECTED"










# ---------------------------------------------------------------------------
# Native write paths: the governance entity rides with the MLflow write inputs as
# siblings (deployment_job_id / model_id / feature_deps / run_tracking_server_id).
# ---------------------------------------------------------------------------


def test_create_registered_model_uses_native(store):
    native_info = UcRegisteredModelInfo(
        name="model",
        catalog_name="catalog",
        schema_name="schema",
        full_name="catalog.schema.model",
        deployment_job_id="9",
    )
    with (
        mock.patch.object(
            store, "_edit_endpoint_and_call", return_value=native_info
        ) as native_call,
        mock.patch.object(store, "_call_endpoint") as legacy_call,
    ):
        result = store.create_registered_model("catalog.schema.model", description="d")
    native_call.assert_called_once()
    legacy_call.assert_not_called()
    assert result.name == "catalog.schema.model"
    assert result.deployment_job_id == "9"


def test_create_registered_model_rejects_non_three_level_name(store):
    # UC model names must be three-level; a non-three-level name errors with guidance.
    # Validated client-side before any HTTP request, so no http mock is needed.
    # Disable the Spark session so the name is not auto-qualified to three levels.
    store.spark = None
    with pytest.raises(MlflowException, match="three levels"):
        store.create_registered_model(name="model_1", description="d")


def test_update_registered_model_uses_native(store):
    native_info = UcRegisteredModelInfo(
        name="model",
        catalog_name="catalog",
        schema_name="schema",
        full_name="catalog.schema.model",
        comment="new",
    )
    with (
        mock.patch.object(
            store, "_edit_endpoint_and_call", return_value=native_info
        ) as native_call,
        mock.patch.object(store, "_call_endpoint") as legacy_call,
    ):
        result = store.update_registered_model("catalog.schema.model", description="new")
    native_call.assert_called_once()
    legacy_call.assert_not_called()
    assert result.description == "new"


def test_rename_registered_model_uses_native(store):
    native_info = UcRegisteredModelInfo(
        name="newname",
        catalog_name="catalog",
        schema_name="schema",
        full_name="catalog.schema.newname",
    )
    with (
        mock.patch.object(
            store, "_edit_endpoint_and_call", return_value=native_info
        ) as native_call,
        mock.patch.object(store, "_call_endpoint") as legacy_call,
    ):
        result = store.rename_registered_model(
            "catalog.schema.model", new_name="catalog.schema.newname"
        )
    native_call.assert_called_once()
    legacy_call.assert_not_called()
    assert result.name == "catalog.schema.newname"










# ---------------------------------------------------------------------------
# Native passthrough (delete / alias), the generic UC tag API, and client-side
# download-uri derivation.
# ---------------------------------------------------------------------------


def test_delete_registered_model_uses_native(store):
    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as native_call,
        mock.patch.object(store, "_call_endpoint") as legacy_call,
    ):
        store.delete_registered_model("catalog.schema.model")
    native_call.assert_called_once()
    legacy_call.assert_not_called()








def test_set_registered_model_tag_uses_native_tag_api(store):
    with (
        mock.patch.object(store, "_edit_endpoint_and_call") as native_call,
        mock.patch.object(store, "_call_endpoint") as legacy_call,
    ):
        store.set_registered_model_tag(
            "catalog.schema.model", RegisteredModelTag(key="k", value="v")
        )
    native_call.assert_called_once()
    legacy_call.assert_not_called()
    # Targets the FUNCTION securable by full name via the generic UC tag API.
    kwargs = native_call.call_args.kwargs
    assert kwargs["securable_type"] == "FUNCTION"
    assert kwargs["securable_full_name"] == "catalog.schema.model"
    body = json.loads(kwargs["req_body"])
    assert body["changes"]["add_tags"] == [{"key": "k", "value": "v"}]




