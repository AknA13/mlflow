import functools
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import google.protobuf.empty_pb2
from pydantic import BaseModel

import mlflow
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import (
    PromptModelConfig,
    PromptVersion,
)
from mlflow.exceptions import MlflowException, RestException
from mlflow.prompt.constants import (
    PROMPT_MODEL_CONFIG_TAG_KEY,
    PROMPT_TYPE_CHAT,
    PROMPT_TYPE_TAG_KEY,
    PROMPT_TYPE_TEXT,
    RESPONSE_FORMAT_TAG_KEY,
)
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    DeleteModelVersion,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    GenerateTemporaryModelVersionCredential,
    SetRegisteredModelAlias,
    TagAssignmentsChange,
    TagKeyValue,
    TemporaryCredentials,
    UcCreateModelVersion,
    UcCreateRegisteredModel,
    UcFinalizeModelVersion,
    UcGetModelVersion,
    UcGetModelVersionByAlias,
    UcGetRegisteredModel,
    UcListModelVersions,
    UcListModelVersionsResponse,
    UcListRegisteredModels,
    UcListRegisteredModelsResponse,
    UcModelVersionInfo,
    UcRegisteredModelInfo,
    UcUpdateModelVersion,
    UcUpdateRegisteredModel,
    UpdateTagSecurableAssignments,
    UpdateTagSubentityAssignments,
)
from mlflow.protos.databricks_uc_registry_service_pb2 import (
    UcEnrichedModelRegistryService,
)
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    CreatePromptRequest,
    CreatePromptVersionRequest,
    DeletePromptAliasRequest,
    DeletePromptRequest,
    DeletePromptTagRequest,
    DeletePromptVersionRequest,
    DeletePromptVersionTagRequest,
    GetPromptRequest,
    GetPromptVersionByAliasRequest,
    GetPromptVersionRequest,
    LinkPromptsToTracesRequest,
    LinkPromptVersionsToModelsRequest,
    LinkPromptVersionsToRunsRequest,
    PromptVersionLinkEntry,
    SearchPromptsRequest,
    SearchPromptsResponse,
    SearchPromptVersionsRequest,
    SearchPromptVersionsResponse,
    SetPromptAliasRequest,
    SetPromptTagRequest,
    SetPromptVersionTagRequest,
    UnityCatalogSchema,
    UpdatePromptRequest,
    UpdatePromptVersionRequest,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    Prompt as ProtoPrompt,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptVersion as ProtoPromptVersion,
)
from mlflow.protos.unity_catalog_prompt_service_pb2 import UnityCatalogPromptService
from mlflow.store._unity_catalog.registry.utils import (
    mlflow_tags_to_proto,
    mlflow_tags_to_proto_version_tags,
    proto_info_to_mlflow_prompt_info,
    proto_to_mlflow_prompt,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
    enriched_registered_model_from_uc_proto as registered_model_from_uc_proto,
)
from mlflow.utils._unity_catalog_utils import (
    enriched_registered_model_search_from_uc_proto as registered_model_search_from_uc_proto,
)
from mlflow.utils._unity_catalog_utils import (
    get_full_name_from_sc,
)
from mlflow.utils.databricks_utils import (
    _print_databricks_deployment_job_url,
    get_databricks_host_creds,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    _UC_OSS_REST_API_PATH_PREFIX,
    call_endpoint,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
)

_TRACKING_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
# UC model-registry endpoints are served on the native /api/2.1/unity-catalog/* surface
# (UcEnrichedModelRegistryService); prompt endpoints remain on /api/2.0/mlflow/unity-catalog/*.
_METHOD_TO_INFO = {
    **extract_api_info_for_service(UnityCatalogPromptService, _REST_API_PATH_PREFIX),
    **extract_api_info_for_service(UcEnrichedModelRegistryService, _UC_OSS_REST_API_PATH_PREFIX),
}
_METHOD_TO_ALL_INFO = {
    **extract_all_api_info_for_service(UnityCatalogPromptService, _REST_API_PATH_PREFIX),
}

_logger = logging.getLogger(__name__)
_DELTA_TABLE = "delta_table"
_MAX_LINEAGE_DATA_SOURCES = 10

# Pre-compiled regex patterns for better performance in search operations
_CATALOG_PATTERN = re.compile(r"catalog\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_SCHEMA_PATTERN = re.compile(r"schema\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)


@dataclass
class _CatalogSchemaFilter:
    """Internal class to hold parsed catalog, schema, and remaining filter."""

    catalog_name: str
    schema_name: str
    remaining_filter: str | None


def _require_arg_unspecified(arg_name, arg_value, default_values=None, message=None):
    default_values = [None] if default_values is None else default_values
    if arg_value not in default_values:
        _raise_unsupported_arg(arg_name, message)


def _raise_unsupported_arg(arg_name, message=None):
    messages = [
        f"Argument '{arg_name}' is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    raise MlflowException(" ".join(messages))


def _raise_unsupported_method(method, message=None):
    messages = [
        f"Method '{method}' is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    raise MlflowException(" ".join(messages))


def _load_model(local_model_dir):
    # Import Model here instead of in the top level, to avoid circular import; the
    # mlflow.models.model module imports from MLflow tracking, which triggers an import of
    # this file during store registry initialization
    from mlflow.models.model import Model

    try:
        return Model.load(local_model_dir)
    except Exception as e:
        raise MlflowException(
            "Unable to load model metadata. Ensure the source path of the model "
            "being registered points to a valid MLflow model directory "
            "(see https://mlflow.org/docs/latest/models.html#storage-format) containing a "
            "model signature (https://mlflow.org/docs/latest/models.html#model-signature) "
            "specifying both input and output type specifications."
        ) from e


def get_feature_dependencies(model_dir):
    """
    Gets the features which a model depends on. This functionality is only implemented on
    Databricks. In OSS mlflow, the dependencies are always empty ("").
    """
    model = _load_model(model_dir)
    if (
        model.flavors.get("python_function", {}).get("loader_module")
        == mlflow.models.model._DATABRICKS_FS_LOADER_MODULE
    ):
        raise MlflowException(
            "This model was packaged by Databricks Feature Store and can only be registered on a "
            "Databricks cluster."
        )
    return ""


def get_model_version_dependencies(model_dir):
    """
    Gets the specified dependencies for a particular model version and formats them
    to be passed into UcCreateModelVersion.
    """
    from mlflow.models.resources import ResourceType

    model = _load_model(model_dir)
    dependencies = []

    # Try to get model.auth_policy.system_auth_policy.resources. If that is not found or empty,
    # then use model.resources.
    if model.auth_policy:
        databricks_resources = model.auth_policy.get("system_auth_policy", {}).get("resources", {})
    else:
        databricks_resources = model.resources

    if databricks_resources:
        databricks_dependencies = databricks_resources.get("databricks", {})
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.VECTOR_SEARCH_INDEX.value,
                "DATABRICKS_VECTOR_INDEX",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.SERVING_ENDPOINT.value,
                "DATABRICKS_MODEL_ENDPOINT",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.FUNCTION.value,
                "DATABRICKS_UC_FUNCTION",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.UC_CONNECTION.value,
                "DATABRICKS_UC_CONNECTION",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.TABLE.value,
                "DATABRICKS_TABLE",
            )
        )
    else:
        # These types of dependencies are required for old models that didn't use
        # resources so they can be registered correctly to UC
        _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY = "databricks_vector_search_index_name"
        _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY = "databricks_embeddings_endpoint_name"
        _DATABRICKS_LLM_ENDPOINT_NAME_KEY = "databricks_llm_endpoint_name"
        _DATABRICKS_CHAT_ENDPOINT_NAME_KEY = "databricks_chat_endpoint_name"
        _DB_DEPENDENCY_KEY = "databricks_dependency"

        databricks_dependencies = model.flavors.get("langchain", {}).get(_DB_DEPENDENCY_KEY, {})

        index_names = _fetch_langchain_dependency_from_model_info(
            databricks_dependencies, _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY
        )
        dependencies.extend(
            {"type": "DATABRICKS_VECTOR_INDEX", "name": index_name} for index_name in index_names
        )
        for key in (
            _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY,
            _DATABRICKS_LLM_ENDPOINT_NAME_KEY,
            _DATABRICKS_CHAT_ENDPOINT_NAME_KEY,
        ):
            endpoint_names = _fetch_langchain_dependency_from_model_info(
                databricks_dependencies, key
            )
            dependencies.extend(
                {"type": "DATABRICKS_MODEL_ENDPOINT", "name": endpoint_name}
                for endpoint_name in endpoint_names
            )
    return dependencies


def _fetch_langchain_dependency_from_model_resources(databricks_dependencies, key, resource_type):
    dependencies = databricks_dependencies.get(key, [])
    deps = []
    for dependency in dependencies:
        if dependency.get("on_behalf_of_user", False):
            continue
        deps.append({"type": resource_type, "name": dependency["name"]})
    return deps


def _fetch_langchain_dependency_from_model_info(databricks_dependencies, key):
    return databricks_dependencies.get(key, [])


class UcEnrichedModelRegistryStore(BaseRestStore):
    """
    Client for the Unity Catalog model registry accessed via the native (enriched)
    ``/api/2.1/unity-catalog/*`` REST endpoints (``UcEnrichedModelRegistryService``).

    Selected by the store registry when ``MLFLOW_ENABLE_UC_NATIVE_MODEL_REGISTRY`` is set;
    otherwise the legacy ``/api/2.0`` client (:class:`UcModelRegistryStore`) is used.

    Args:
        store_uri: URI with scheme 'databricks-uc'
        tracking_uri: URI of the Databricks MLflow tracking server from which to fetch
            run info and download run artifacts, when creating new model
            versions from source artifacts logged to an MLflow run.
    """

    def __init__(self, store_uri, tracking_uri):
        super().__init__(get_host_creds=functools.partial(get_databricks_host_creds, store_uri))
        self.store_uri = store_uri
        self.tracking_uri = tracking_uri
        self.get_tracking_host_creds = functools.partial(get_databricks_host_creds, tracking_uri)
        try:
            self.spark = _get_active_spark_session()
        except Exception:
            pass

    def _get_response_from_method(self, method):
        method_to_response = {
            GetRun: GetRun.Response,
            CreatePromptRequest: ProtoPrompt,
            SearchPromptsRequest: SearchPromptsResponse,
            DeletePromptRequest: google.protobuf.empty_pb2.Empty,
            SetPromptTagRequest: google.protobuf.empty_pb2.Empty,
            DeletePromptTagRequest: google.protobuf.empty_pb2.Empty,
            CreatePromptVersionRequest: ProtoPromptVersion,
            GetPromptVersionRequest: ProtoPromptVersion,
            DeletePromptVersionRequest: google.protobuf.empty_pb2.Empty,
            GetPromptVersionByAliasRequest: ProtoPromptVersion,
            UpdatePromptRequest: ProtoPrompt,
            GetPromptRequest: ProtoPrompt,
            SearchPromptVersionsRequest: SearchPromptVersionsResponse,
            SetPromptAliasRequest: google.protobuf.empty_pb2.Empty,
            DeletePromptAliasRequest: google.protobuf.empty_pb2.Empty,
            SetPromptVersionTagRequest: google.protobuf.empty_pb2.Empty,
            DeletePromptVersionTagRequest: google.protobuf.empty_pb2.Empty,
            UpdatePromptVersionRequest: ProtoPromptVersion,
            LinkPromptVersionsToModelsRequest: google.protobuf.empty_pb2.Empty,
            LinkPromptsToTracesRequest: google.protobuf.empty_pb2.Empty,
            LinkPromptVersionsToRunsRequest: google.protobuf.empty_pb2.Empty,
            UcGetRegisteredModel: UcRegisteredModelInfo,
            UcCreateRegisteredModel: UcRegisteredModelInfo,
            UcUpdateRegisteredModel: UcRegisteredModelInfo,
            UcListRegisteredModels: UcListRegisteredModelsResponse,
            DeleteRegisteredModel: DeleteRegisteredModel.Response,
            UcGetModelVersion: UcModelVersionInfo,
            UcGetModelVersionByAlias: UcModelVersionInfo,
            UcCreateModelVersion: UcModelVersionInfo,
            UcUpdateModelVersion: UcModelVersionInfo,
            UcFinalizeModelVersion: UcModelVersionInfo,
            UcListModelVersions: UcListModelVersionsResponse,
            DeleteModelVersion: DeleteModelVersion.Response,
            GenerateTemporaryModelVersionCredential: TemporaryCredentials,
            SetRegisteredModelAlias: SetRegisteredModelAlias.Response,
            DeleteRegisteredModelAlias: DeleteRegisteredModelAlias.Response,
            UpdateTagSecurableAssignments: UpdateTagSecurableAssignments.Response,
            UpdateTagSubentityAssignments: UpdateTagSubentityAssignments.Response,
        }
        return method_to_response[method]()

    def _get_endpoint_from_method(self, method):
        return _METHOD_TO_INFO[method]

    def _get_all_endpoints_from_method(self, method):
        return _METHOD_TO_ALL_INFO[method]

    # CRUD API for RegisteredModel objects

    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.
            deployment_job_id: Optional deployment job id.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.

        """
        full_name = get_full_name_from_sc(name, self.spark)
        parts = full_name.split(".")
        if len(parts) != 3 or not all(parts):
            raise MlflowException(
                f"Not a valid Unity Catalog model name: '{full_name}'. Unity Catalog model names "
                "must have three levels (catalog.schema.model). If you are trying to use the "
                "legacy Workspace Model Registry instead of the recommended Unity Catalog Model "
                "Registry, set the Model Registry URI to 'databricks' (legacy) instead of "
                "'databricks-uc'."
            )
        catalog, schema, model = parts
        req_body = message_to_json(
            UcCreateRegisteredModel(
                name=model,
                catalog_name=catalog,
                schema_name=schema,
                comment=description,
                tags=[TagKeyValue(key=t.key, value=t.value) for t in (tags or [])],
                deployment_job_id=str(deployment_job_id) if deployment_job_id else None,
            )
        )
        endpoint, method = self._get_endpoint_from_method(UcCreateRegisteredModel)
        try:
            resp = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                proto_name=UcCreateRegisteredModel,
            )
        except RestException as e:
            if "METASTORE_DOES_NOT_EXIST" in e.message:
                # The user is likely on a workspace without Unity Catalog enabled.
                raise MlflowException(
                    message=e.message.rstrip(".")
                    + ". If you are trying to use the Model Registry in a Databricks workspace"
                    " that does not have Unity Catalog enabled, either enable Unity Catalog in"
                    " the workspace (recommended) or set the Model Registry URI to 'databricks'"
                    " to use the legacy Workspace Model Registry.",
                    error_code=e.error_code,
                )
            raise
        if deployment_job_id:
            _print_databricks_deployment_job_url(
                model_name=full_name, job_id=str(deployment_job_id)
            )
        return registered_model_from_uc_proto(resp)

    def update_registered_model(self, name, description=None, deployment_job_id=None):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.
            deployment_job_id: Optional deployment job id.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UcUpdateRegisteredModel(
                full_name_arg=full_name,
                comment=description,
                deployment_job_id=(
                    str(deployment_job_id) if deployment_job_id is not None else None
                ),
            )
        )
        endpoint, method = self._get_endpoint_from_method(UcUpdateRegisteredModel)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UcUpdateRegisteredModel,
            full_name_arg=full_name,
        )
        if deployment_job_id:
            _print_databricks_deployment_job_url(
                model_name=full_name, job_id=str(deployment_job_id)
            )
        return registered_model_from_uc_proto(native_resp)

    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UcUpdateRegisteredModel(full_name_arg=full_name, new_name=new_name)
        )
        endpoint, method = self._get_endpoint_from_method(UcUpdateRegisteredModel)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UcUpdateRegisteredModel,
            full_name_arg=full_name,
        )
        return registered_model_from_uc_proto(native_resp)

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(DeleteRegisteredModel(full_name_arg=full_name))
        endpoint, method = self._get_endpoint_from_method(DeleteRegisteredModel)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=DeleteRegisteredModel,
            full_name_arg=full_name,
        )
        return

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        Args:
            filter_string: Filter query string, defaults to searching all registered models.
            max_results: Maximum number of registered models desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.

        """
        _require_arg_unspecified("filter_string", filter_string)
        _require_arg_unspecified("order_by", order_by)
        req_body = message_to_json(
            UcListRegisteredModels(max_results=max_results, page_token=page_token)
        )
        endpoint, method = self._get_endpoint_from_method(UcListRegisteredModels)
        resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            proto_name=UcListRegisteredModels,
        )
        registered_models = [
            registered_model_search_from_uc_proto(rm) for rm in resp.registered_models
        ]
        return PagedList(registered_models, resp.next_page_token)

    def get_registered_model(self, name):
        """
        Get registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(UcGetRegisteredModel(full_name_arg=full_name))
        # The server json_inlines the wrapper, so the response is a flat
        # UcRegisteredModelInfo (governance fields + enrichment) at the top level.
        endpoint, method = self._get_endpoint_from_method(UcGetRegisteredModel)
        native_resp = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UcGetRegisteredModel,
            full_name_arg=full_name,
        )
        return registered_model_from_uc_proto(native_resp)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                each stage.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        alias_doc_url = "https://mlflow.org/docs/latest/model-registry.html#deploy-and-organize-models-with-aliases-and-tags"
        if stages is None:
            message = (
                "To load the latest version of a model in Unity Catalog, you can "
                "set an alias on the model version and load it by alias. See "
                f"{alias_doc_url} for details."
            )
        else:
            message = (
                f"Detected attempt to load latest model version in stages {stages}. "
                "You may see this error because:\n"
                "1) You're attempting to load a model version by stage. Setting stages "
                "and loading model versions by stage is unsupported in Unity Catalog. Instead, "
                "use aliases for flexible model deployment. See "
                f"{alias_doc_url} for details.\n"
                "2) You're attempting to load a model version by alias. Use "
                "syntax 'models:/your_model_name@your_alias_name'\n"
                "3) You're attempting load a model version by version number. Verify "
                "that the version number is a valid integer"
            )

        _raise_unsupported_method(
            method="get_latest_versions",
            message=message,
        )

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateTagSecurableAssignments(
                changes=TagAssignmentsChange(add_tags=[TagKeyValue(key=tag.key, value=tag.value)])
            )
        )
        endpoint, method = self._get_endpoint_from_method(UpdateTagSecurableAssignments)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateTagSecurableAssignments,
            securable_type="FUNCTION",
            securable_full_name=full_name,
        )
        return

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        native_req = message_to_json(
            UpdateTagSecurableAssignments(changes=TagAssignmentsChange(remove=[key]))
        )
        endpoint, method = self._get_endpoint_from_method(UpdateTagSecurableAssignments)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=native_req,
            proto_name=UpdateTagSecurableAssignments,
            securable_type="FUNCTION",
            securable_full_name=full_name,
        )
        return

    # Prompt-related method overrides for UC

    def create_prompt(
        self,
        name: str,
        description: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> Prompt:
        """
        Create a new prompt in Unity Catalog (metadata only, no initial version).
        """
        # Create a Prompt object with the provided fields
        prompt_proto = ProtoPrompt()
        prompt_proto.name = name
        if description:
            prompt_proto.description = description
        if tags:
            prompt_proto.tags.extend(mlflow_tags_to_proto(tags))

        req_body = message_to_json(
            CreatePromptRequest(
                name=name,
                prompt=prompt_proto,
            )
        )
        response_proto = self._call_endpoint(CreatePromptRequest, req_body)
        return proto_info_to_mlflow_prompt_info(response_proto, tags or {})

    def search_prompts(
        self,
        filter_string: str | None = None,
        max_results: int | None = None,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts in Unity Catalog.

        Args:
            filter_string: Filter string that must include catalog and schema in the format:
                "catalog = 'catalog_name' AND schema = 'schema_name'"
            max_results: Maximum number of results to return
            order_by: List of fields to order by (not used in current implementation)
            page_token: Token for pagination
        """
        # Parse catalog and schema from filter string
        if filter_string:
            filter_string = self._parse_experiment_id_filter(filter_string)
            parsed_filter = self._parse_catalog_schema_from_filter(filter_string)
        else:
            raise MlflowException(
                "For Unity Catalog prompt registries, you must specify catalog and schema "
                "in the filter string: \"catalog = 'catalog_name' AND schema = 'schema_name'\"",
                INVALID_PARAMETER_VALUE,
            )

        # Build the request with Unity Catalog schema
        unity_catalog_schema = UnityCatalogSchema(
            catalog_name=parsed_filter.catalog_name,
            schema_name=parsed_filter.schema_name,
        )
        req_body = message_to_json(
            SearchPromptsRequest(
                catalog_schema=unity_catalog_schema,
                filter=parsed_filter.remaining_filter,
                max_results=max_results,
                page_token=page_token,
            )
        )

        response_proto = self._call_endpoint(SearchPromptsRequest, req_body)
        # For UC, only use the basic prompt info without extra tag fetching
        prompts = [
            proto_info_to_mlflow_prompt_info(prompt_info, {})
            for prompt_info in response_proto.prompts
        ]

        return PagedList(prompts, response_proto.next_page_token)

    def _parse_catalog_schema_from_filter(self, filter_string: str | None) -> _CatalogSchemaFilter:
        """
        Parse catalog and schema from filter string for Unity Catalog using regex.

        Expects filter format: "catalog = 'catalog_name' AND schema = 'schema_name'"

        Args:
            filter_string: Filter string containing catalog and schema

        Returns:
            _CatalogSchemaFilter object with catalog_name, schema_name, and remaining_filter

        Raises:
            MlflowException: If filter format is invalid for Unity Catalog
        """
        if not filter_string:
            raise MlflowException(
                "For Unity Catalog prompt registries, you must specify catalog and schema "
                "in the filter string: \"catalog = 'catalog_name' AND schema = 'schema_name'\"",
                INVALID_PARAMETER_VALUE,
            )

        # Use pre-compiled regex patterns for better performance
        catalog_match = _CATALOG_PATTERN.search(filter_string)
        schema_match = _SCHEMA_PATTERN.search(filter_string)

        if not catalog_match or not schema_match:
            raise MlflowException(
                "For Unity Catalog prompt registries, filter string must include both "
                "catalog and schema in the format: "
                "\"catalog = 'catalog_name' AND schema = 'schema_name'\". "
                f"Got: {filter_string}",
                INVALID_PARAMETER_VALUE,
            )

        catalog_name = catalog_match.group(1)
        schema_name = schema_match.group(1)

        # Remove catalog and schema from filter string to get remaining filters
        # First, normalize the filter by splitting on AND and rebuilding
        # without catalog/schema parts
        parts = re.split(r"\s+AND\s+", filter_string, flags=re.IGNORECASE)
        remaining_parts = []

        for part in parts:
            part = part.strip()
            # Skip parts that match catalog or schema patterns
            if not (_CATALOG_PATTERN.match(part) or _SCHEMA_PATTERN.match(part)):
                remaining_parts.append(part)

        # Rejoin the remaining parts
        remaining_filter = " AND ".join(remaining_parts) if remaining_parts else None

        return _CatalogSchemaFilter(catalog_name, schema_name, remaining_filter)

    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt from Unity Catalog.
        """
        req_body = message_to_json(DeletePromptRequest(name=name))
        endpoint, method = self._get_endpoint_from_method(DeletePromptRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            proto_name=DeletePromptRequest,
        )

    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt in Unity Catalog.
        """
        req_body = message_to_json(SetPromptTagRequest(name=name, key=key, value=value))
        endpoint, method = self._get_endpoint_from_method(SetPromptTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            key=key,
            proto_name=SetPromptTagRequest,
        )

    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt in Unity Catalog.
        """
        req_body = message_to_json(DeletePromptTagRequest(name=name, key=key))
        endpoint, method = self._get_endpoint_from_method(DeletePromptTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            key=key,
            proto_name=DeletePromptTagRequest,
        )

    def get_prompt(self, name: str) -> Prompt | None:
        """
        Get prompt by name from Unity Catalog.
        """
        try:
            req_body = message_to_json(GetPromptRequest(name=name))
            endpoint, method = self._get_endpoint_from_method(GetPromptRequest)
            response_proto = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                proto_name=GetPromptRequest,
            )
            return proto_info_to_mlflow_prompt_info(response_proto, {})
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise

    def create_prompt_version(
        self,
        name: str,
        template: str | list[dict[str, Any]],
        description: str | None = None,
        tags: dict[str, str] | None = None,
        response_format: type[BaseModel] | dict[str, Any] | None = None,
        model_config: "PromptModelConfig | dict[str, Any] | None" = None,
    ) -> PromptVersion:
        """
        Create a new prompt version in Unity Catalog.
        """
        # Create a PromptVersion object with the provided fields
        prompt_version_proto = ProtoPromptVersion()
        prompt_version_proto.name = name
        # JSON-encode the template for Unity Catalog server
        prompt_version_proto.template = json.dumps(template)

        # Note: version will be set by the backend when creating a new version
        # We don't set it here as it's generated server-side
        if description:
            prompt_version_proto.description = description

        final_tags = tags.copy() if tags else {}
        if response_format:
            final_tags[RESPONSE_FORMAT_TAG_KEY] = json.dumps(
                PromptVersion.convert_response_format_to_dict(response_format)
            )
        if model_config:
            # Convert ModelConfig to dict if needed
            if isinstance(model_config, PromptModelConfig):
                config_dict = model_config.to_dict()
            else:
                config_dict = model_config

            final_tags[PROMPT_MODEL_CONFIG_TAG_KEY] = json.dumps(config_dict)
        if isinstance(template, str):
            final_tags[PROMPT_TYPE_TAG_KEY] = PROMPT_TYPE_TEXT
        else:
            final_tags[PROMPT_TYPE_TAG_KEY] = PROMPT_TYPE_CHAT

        if final_tags:
            prompt_version_proto.tags.extend(mlflow_tags_to_proto_version_tags(final_tags))

        req_body = message_to_json(
            CreatePromptVersionRequest(
                name=name,
                prompt_version=prompt_version_proto,
            )
        )
        endpoint, method = self._get_endpoint_from_method(CreatePromptVersionRequest)
        response_proto = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            proto_name=CreatePromptVersionRequest,
        )
        return proto_to_mlflow_prompt(response_proto)

    def get_prompt_version(self, name: str, version: str | int) -> PromptVersion | None:
        """
        Get a specific prompt version from Unity Catalog.
        """
        try:
            req_body = message_to_json(GetPromptVersionRequest(name=name, version=str(version)))
            endpoint, method = self._get_endpoint_from_method(GetPromptVersionRequest)
            response_proto = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                version=version,
                proto_name=GetPromptVersionRequest,
            )

            # No longer fetch prompt-level tags - keep them completely separate
            return proto_to_mlflow_prompt(response_proto)
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise

    def delete_prompt_version(self, name: str, version: str | int) -> None:
        """
        Delete a prompt version from Unity Catalog.
        """
        # Delete the specific version only
        req_body = message_to_json(DeletePromptVersionRequest(name=name, version=str(version)))
        endpoint, method = self._get_endpoint_from_method(DeletePromptVersionRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            version=version,
            proto_name=DeletePromptVersionRequest,
        )

    def search_prompt_versions(
        self, name: str, max_results: int | None = None, page_token: str | None = None
    ) -> SearchPromptVersionsResponse:
        """
        Search prompt versions for a given prompt name in Unity Catalog.

        Note: Unity Catalog server uses a non-standard endpoint pattern for this operation.

        Args:
            name: Name of the prompt to search versions for
            max_results: Maximum number of versions to return
            page_token: Token for pagination

        Returns:
            SearchPromptVersionsResponse containing the list of versions
        """
        req_body = message_to_json(
            SearchPromptVersionsRequest(name=name, max_results=max_results, page_token=page_token)
        )
        endpoint, method = self._get_endpoint_from_method(SearchPromptVersionsRequest)
        return self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            proto_name=SearchPromptVersionsRequest,
        )

    def set_prompt_version_tag(self, name: str, version: str | int, key: str, value: str) -> None:
        """
        Set a tag on a prompt version in Unity Catalog.
        """
        req_body = message_to_json(
            SetPromptVersionTagRequest(name=name, version=str(version), key=key, value=value)
        )
        endpoint, method = self._get_endpoint_from_method(SetPromptVersionTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            version=version,
            key=key,
            proto_name=SetPromptVersionTagRequest,
        )

    def delete_prompt_version_tag(self, name: str, version: str | int, key: str) -> None:
        """
        Delete a tag from a prompt version in Unity Catalog.
        """
        req_body = message_to_json(
            DeletePromptVersionTagRequest(name=name, version=str(version), key=key)
        )
        endpoint, method = self._get_endpoint_from_method(DeletePromptVersionTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            version=version,
            key=key,
            proto_name=DeletePromptVersionTagRequest,
        )

    def get_prompt_version_by_alias(self, name: str, alias: str) -> PromptVersion | None:
        """
        Get a prompt version by alias from Unity Catalog.
        """
        try:
            req_body = message_to_json(GetPromptVersionByAliasRequest(name=name, alias=alias))
            endpoint, method = self._get_endpoint_from_method(GetPromptVersionByAliasRequest)
            response_proto = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                alias=alias,
                proto_name=GetPromptVersionByAliasRequest,
            )

            # No longer fetch prompt-level tags - keep them completely separate
            return proto_to_mlflow_prompt(response_proto)
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise

    def set_prompt_alias(self, name: str, alias: str, version: str | int) -> None:
        """
        Set an alias for a prompt version in Unity Catalog.
        """
        req_body = message_to_json(
            SetPromptAliasRequest(name=name, alias=alias, version=str(version))
        )
        endpoint, method = self._get_endpoint_from_method(SetPromptAliasRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            alias=alias,
            version=version,
            proto_name=SetPromptAliasRequest,
        )

    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete an alias from a prompt in Unity Catalog.
        """
        req_body = message_to_json(DeletePromptAliasRequest(name=name, alias=alias))
        endpoint, method = self._get_endpoint_from_method(DeletePromptAliasRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            alias=alias,
            proto_name=DeletePromptAliasRequest,
        )

    def link_prompt_version_to_model(self, name: str, version: str, model_id: str) -> None:
        """
        Link a prompt version to a model in Unity Catalog.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to link.
            model_id: ID of the model to link to.
        """
        # Call the default implementation, since the LinkPromptVersionsToModels API
        # will initially be a no-op until the Databricks backend supports it
        super().link_prompt_version_to_model(name=name, version=version, model_id=model_id)

        prompt_version_entry = PromptVersionLinkEntry(name=name, version=version)
        req_body = message_to_json(
            LinkPromptVersionsToModelsRequest(
                prompt_versions=[prompt_version_entry], model_ids=[model_id]
            )
        )
        endpoint, method = self._get_endpoint_from_method(LinkPromptVersionsToModelsRequest)
        try:
            # NB: This will not raise an exception if the backend does not support linking.
            # We do this to prioritize reduction in errors and log spam while the prompt
            # registry remains experimental
            self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                version=version,
                model_id=model_id,
                proto_name=LinkPromptVersionsToModelsRequest,
            )
        except Exception:
            _logger.debug("Failed to link prompt version to model in unity catalog", exc_info=True)

    def link_prompts_to_trace(self, prompt_versions: list[PromptVersion], trace_id: str) -> None:
        """
        Link multiple prompt versions to a trace in Unity Catalog.

        Args:
            prompt_versions: List of PromptVersion objects to link.
            trace_id: Trace ID to link to each prompt version.
        """
        prompt_version_entries = [
            PromptVersionLinkEntry(name=pv.name, version=str(pv.version)) for pv in prompt_versions
        ]

        batch_size = 25
        endpoint, method = self._get_endpoint_from_method(LinkPromptsToTracesRequest)

        for i in range(0, len(prompt_version_entries), batch_size):
            batch = prompt_version_entries[i : i + batch_size]
            req_body = message_to_json(
                LinkPromptsToTracesRequest(prompt_versions=batch, trace_ids=[trace_id])
            )
            try:
                self._edit_endpoint_and_call(
                    endpoint=endpoint,
                    method=method,
                    req_body=req_body,
                    proto_name=LinkPromptsToTracesRequest,
                )
            except Exception:
                _logger.debug("Failed to link prompts to traces in unity catalog", exc_info=True)

    def link_prompt_version_to_run(self, name: str, version: str, run_id: str) -> None:
        """
        Link a prompt version to a run in Unity Catalog.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to link.
            run_id: ID of the run to link to.
        """
        super().link_prompt_version_to_run(name=name, version=version, run_id=run_id)

        prompt_version_entry = PromptVersionLinkEntry(name=name, version=version)
        endpoint, method = self._get_endpoint_from_method(LinkPromptVersionsToRunsRequest)

        req_body = message_to_json(
            LinkPromptVersionsToRunsRequest(
                prompt_versions=[prompt_version_entry], run_ids=[run_id]
            )
        )
        try:
            self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                proto_name=LinkPromptVersionsToRunsRequest,
            )
        except Exception:
            _logger.debug("Failed to link prompt version to run in unity catalog", exc_info=True)

    def _edit_endpoint_and_call(
        self, endpoint, method, req_body, proto_name, extra_headers=None, **kwargs
    ):
        """
        Edit endpoint URL with parameters and make the call.

        Args:
            endpoint: URL template with placeholders like {name}, {key}
            method: HTTP method
            req_body: Request body
            proto_name: Protobuf message class for response
            extra_headers: Optional extra request headers (e.g. lineage headers).
            **kwargs: Parameters to substitute in the endpoint template
        """
        # Replace placeholders in endpoint with actual values
        for key, value in kwargs.items():
            if value is not None:
                endpoint = endpoint.replace(f"{{{key}}}", str(value))

        # Make the API call
        return call_endpoint(
            self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=self._get_response_from_method(proto_name),
            extra_headers=extra_headers,
        )
