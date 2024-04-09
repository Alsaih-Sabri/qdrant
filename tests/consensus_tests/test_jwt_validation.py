import json
import pathlib
import random
import string
from tabnanny import check
import tempfile
import time
from inspect import isfunction
from typing import Callable, List, Optional, Tuple, Union

import grpc
import grpc_requests
import pytest
import requests
from grpc_interceptor import ClientCallDetails, ClientInterceptor

from consensus_tests import fixtures

from .utils import encode_jwt, make_peer_folder, start_first_peer, wait_for


def random_str():
    return "".join(random.choices(string.ascii_lowercase, k=10))


PORT_SEED = 10000
REST_URI = f"http://127.0.0.1:{PORT_SEED + 2}"
GRPC_URI = f"127.0.0.1:{PORT_SEED + 1}"

SECRET = "my_top_secret_key"

API_KEY_HEADERS = {"Api-Key": SECRET}
API_KEY_METADATA = [("api-key", SECRET)]

COLL_NAME = "primary_test_collection"

# Global read access token
TOKEN_R = encode_jwt({"access": "r"}, SECRET)

# Collection read access token
TOKEN_COLL_R = encode_jwt({"access": [{"collections": [COLL_NAME], "access": "r"}]}, SECRET)

# Collection read-write access token
TOKEN_COLL_RW = encode_jwt({"access": [{"collections": [COLL_NAME], "access": "rw"}]}, SECRET)

# Global manage access token
TOKEN_M = encode_jwt({"access": "m"}, SECRET)

RENAMABLE_ALIASES = [random_str() for _ in range(10)]
MOVABLE_SHARD_IDS = [i + 2 for i in range(10)]
SHARD_ID = 1
SNAPSHOT_NAME = "test_snapshot"
POINT_ID = 0
FIELD_NAME = "test_field"
PEER_ID = 0
DELETABLE_SHARD_KEYS = [random_str() for _ in range(2)]
MOVABLE_SHARD_KEYS = [random_str() for _ in range(2)]
SHARD_KEY = "existing_shard_key"
DELETABLE_COLLECTION_SNAPSHOTS = []

_cached_clients = None


class Access:
    def __init__(self, r, coll_rw, m=True, coll_r=None):
        self.read = r
        self.coll_rw = coll_rw
        self.manage = m
        self.coll_r = r if coll_r is None else coll_r


class AccessStub:
    def __init__(
        self,
        read,
        read_write,
        manage,
        rest_req=None,
        grpc_req=None,
        collection_name=COLL_NAME,
        snapshot_name=SNAPSHOT_NAME,
    ):
        self.access = Access(read, read_write, manage)
        self.rest_req = rest_req
        self.grpc_req = grpc_req
        self.collection_name = collection_name
        self.snapshot_name = snapshot_name


### operation stubs to use for update_collection_cluster_setup tests
# move_shard_operation = AccessStub(
#     False,
#     False,
#     True,
#     {
#         "move_shard": {
#             "shard_id": SHARD_ID,
#             "from_peer_id": PEER_ID,
#             "to_peer_id": PEER_ID + 1,
#         }
#     },
#     {
#         "collection_name": COLL_NAME,
#         "move_shard": {
#             "shard_id": SHARD_ID,
#             "from_peer_id": PEER_ID,
#             "to_peer_id": PEER_ID + 1,
#         },
#     },
# )
# replicate_shard_operation = AccessStub(
#     False,
#     False,
#     True,
#     {
#         "replicate_shard": {
#             "shard_id": SHARD_ID,
#             "from_peer_id": PEER_ID,
#             "to_peer_id": PEER_ID,
#         }
#     },
# )
# abort_shard_transfer_operation = AccessStub(
#     False,
#     False,
#     True,
#     {
#         "abort_transfer": {
#             "shard_id": SHARD_ID,
#             "from_peer_id": PEER_ID,
#             "to_peer_id": PEER_ID,
#         }
#     },
# )
# drop_shard_replica_operation = AccessStub(
#     False,
#     False,
#     True,
#     {
#         "drop_replica": {
#             "shard_id": SHARD_ID,
#             "peer_id": PEER_ID,
#         }
#     },
# )
# default_create_shard_key_operation = AccessStub(
#     False,
#     True,
#     True,
#     {
#         "create_sharding_key": default_shard_key_config,
#     },
# )
# custom_create_shard_key_operation = AccessStub(
#     False,
#     False,
#     True,
#     {
#         "create_sharding_key": None,
#     },
# )
# drop_shard_key_operation = AccessStub(
#     False,
#     True,
#     True,
#     {
#         "drop_sharding_key": {
#             "shard_key": "a",
#         }
#     },
# )
# restart_transfer_operation = AccessStub(
#     False,
#     False,
#     True,
#     {
#         "restart_transfer": {
#             "shard_id": SHARD_ID,
#             "from_peer_id": PEER_ID,
#             "to_peer_id": PEER_ID,
#             "method": "stream_records",
#         }
#     },
# )


class EndpointAccess:
    def __init__(self, r, coll_rw, m, rest_endpoint, grpc_endpoint=None, **kwargs):
        self.access = Access(r, coll_rw, m, **kwargs)
        self.rest_endpoint = rest_endpoint
        self.grpc_endpoint = grpc_endpoint


ACTION_ACCESS = {
    ### Collections ###
    "list_collections": EndpointAccess(
        True, True, True, "GET /collections", "qdrant.Collections/List"
    ),
    "get_collection": EndpointAccess(
        True, True, True, "GET /collections/{collection_name}", "qdrant.Collections/Get"
    ),
    "create_collection": EndpointAccess(
        False, False, True, "PUT /collections/{collection_name}", "qdrant.Collections/Create"
    ),
    "delete_collection": EndpointAccess(
        False, False, True, "DELETE /collections/{collection_name}", "qdrant.Collections/Delete"
    ),
    "update_collection_params": EndpointAccess(
        False, False, True, "PATCH /collections/{collection_name}", "qdrant.Collections/Update"
    ),
    "get_collection_cluster_info": EndpointAccess(
        True,
        True,
        True,
        "GET /collections/{collection_name}/cluster",
        "qdrant.Collections/CollectionClusterInfo",
    ),  # TODO: are these the expected permissions for coll cluster info?
    "collection_exists": EndpointAccess(
        True,
        True,
        True,
        "GET /collections/{collection_name}/exists",
        "qdrant.Collections/CollectionExists",
    ),
    # TODO: also test these actions for update cluster setup:
    # "move_shard_operation": EndpointAccess(
    #     False,
    #     False,
    #     True,
    #     "POST /collections/{collection_name}/cluster",
    #     "qdrant.Collections/UpdateCollectionClusterSetup",
    # ),
    # replicate_shard_operation,
    # abort_shard_transfer_operation,
    # drop_shard_replica_operation,
    # restart_transfer_operation,
    # default_create_shard_key_operation,
    # custom_create_shard_key_operation,
    # drop_shard_key_operation
    ### Aliases ###
    "create_alias": EndpointAccess(
        False,
        False,
        True,
        "POST /collections/aliases",
        "qdrant.Collections/UpdateAliases",
    ),
    "rename_alias": EndpointAccess(
        False,
        False,
        True,
        "POST /collections/aliases",
        "qdrant.Collections/UpdateAliases",
    ),
    "delete_alias": EndpointAccess(
        False,
        False,
        True,
        "POST /collections/aliases",
        "qdrant.Collections/UpdateAliases",
    ),
    "list_collection_aliases": EndpointAccess(
        True,
        True,
        True,
        "GET /collections/{collection_name}/aliases",
        "qdrant.Collections/ListCollectionAliases",
    ),
    "list_aliases": EndpointAccess(
        True, True, True, "GET /aliases", "qdrant.Collections/ListAliases"
    ),
    ### Shard Keys ###
    "create_default_shard_key": EndpointAccess(
        False,
        True,
        True,
        "PUT /collections/{collection_name}/shards",
        "qdrant.Collections/CreateShardKey",
    ),
    "create_custom_shard_key": EndpointAccess(
        False,
        False,
        True,
        "PUT /collections/{collection_name}/shards",
        "qdrant.Collections/CreateShardKey",
    ),
    "delete_shard_key": EndpointAccess(
        False,
        True,
        True,
        "POST /collections/{collection_name}/shards/delete",
        "qdrant.Collections/DeleteShardKey",
    ),
    ### Payload Indexes ###
    "create_index": EndpointAccess(
        False,
        True,
        True,
        "PUT /collections/{collection_name}/index",
        "qdrant.Points/CreateFieldIndex",
    ),
    "delete_index": EndpointAccess(
        False,
        True,
        True,
        "DELETE /collections/{collection_name}/index/{field_name}",
        "qdrant.Points/DeleteFieldIndex",
    ),
    ### Collection Snapshots ###
    "list_collection_snapshots": EndpointAccess(
        True,
        True,
        True,
        "GET /collections/{collection_name}/snapshots",
        "qdrant.Snapshots/List",
    ),  # TODO: this should not be allowed with payload constraints
    "create_collection_snapshot": EndpointAccess(
        False,
        True,
        True,
        "POST /collections/{collection_name}/snapshots",
        "qdrant.Snapshots/Create",
    ),
    "delete_collection_snapshot": EndpointAccess(
        False,
        True,
        True,
        "DELETE /collections/{collection_name}/snapshots/{snapshot_name}",
        "qdrant.Snapshots/Delete",
    ),
    "download_collection_snapshot": EndpointAccess(
        True, True, True, "GET /collections/{collection_name}/snapshots/{snapshot_name}"
    ),  # TODO: confirm access rights
    "upload_collection_snapshot": EndpointAccess(
        False, False, True, "POST /collections/{collection_name}/snapshots/upload"
    ),
    "recover_collection_snapshot": EndpointAccess(
        False,
        False,
        True,
        "PUT /collections/{collection_name}/snapshots/recover",
    ),
    ### Shard Snapshots ###
    "upload_shard_snapshot": EndpointAccess(
        False,
        False,
        True,
        "POST /collections/{collection_name}/shards/{shard_id}/snapshots/upload",
    ),
    "recover_shard_snapshot": EndpointAccess(
        False,
        False,
        True,
        "PUT /collections/{collection_name}/shards/{shard_id}/snapshots/recover",
    ),
    "create_shard_snapshot": EndpointAccess(
        False, True, True, "POST /collections/{collection_name}/shards/{shard_id}/snapshots"
    ),
    "list_shard_snapshots": EndpointAccess(
        True, True, True, "GET /collections/{collection_name}/shards/{shard_id}/snapshots"
    ),
    "delete_shard_snapshot": EndpointAccess(
        False,
        True,
        True,
        "DELETE /collections/{collection_name}/shards/{shard_id}/snapshots/{snapshot_name}",
    ),
    "download_shard_snapshot": EndpointAccess(
        True,
        True,
        True,
        "GET /collections/{collection_name}/shards/{shard_id}/snapshots/{snapshot_name}",
    ),
    ### Full Snapshots ###
    "list_full_snapshots": EndpointAccess(
        True, False, True, "GET /snapshots", "qdrant.Snapshots/ListFull", coll_r=False
    ),
    "create_full_snapshot": EndpointAccess(
        False, False, True, "POST /snapshots", "qdrant.Snapshots/CreateFull"
    ),
    "delete_full_snapshot": EndpointAccess(
        False, False, True, "DELETE /snapshots/{snapshot_name}", "qdrant.Snapshots/DeleteFull"
    ),
    "download_full_snapshot": EndpointAccess(
        True, False, True, "GET /snapshots/{snapshot_name}", coll_r=False
    ),
    ### Cluster ###
    "get_cluster": EndpointAccess(True, False, True, "GET /cluster", coll_r=False),
    "recover_raft_state": EndpointAccess(False, False, True, "POST /cluster/recover"),
    "delete_peer": EndpointAccess(False, False, True, "DELETE /cluster/peer/{peer_id}"),
    ### Points ###
    # TODO: add tests for these actions
    # "get_point": EndpointAccess(
    #     True, True, True, "GET /collections/{collection_name}/points/{id}"
    # ),
    # "get_points": EndpointAccess(
    #     True, True, True, "POST /collections/{collection_name}/points", "qdrant.Points/Get"
    # ),
    # "upsert_points": EndpointAccess(
    #     False, True, True, "PUT /collections/{collection_name}/points", "qdrant.Points/Upsert"
    # ),
    # "update_points_batch": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/batch",
    #     "qdrant.Points/UpdateBatch",
    # ),
    # "delete_points": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/delete",
    #     "qdrant.Points/Delete",
    # ),
    # "update_vectors": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "PUT /collections/{collection_name}/points/vectors",
    #     "qdrant.Points/UpdateVectors",
    # ),
    # "delete_vectors": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/vectors/delete",
    #     "qdrant.Points/DeleteVectors",
    # ),
    # "set_payload": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/payload",
    #     "qdrant.Points/SetPayload",
    # ),
    # "overwrite_payload": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "PUT /collections/{collection_name}/points/payload",
    #     "qdrant.Points/OverwritePayload",
    # ),
    # "delete_payload": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/payload/delete",
    #     "qdrant.Points/DeletePayload",
    # ),
    # "clear_payload": EndpointAccess(
    #     False,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/payload/clear",
    #     "qdrant.Points/ClearPayload",
    # ),
    # "scroll_points": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/scroll",
    #     "qdrant.Points/Scroll",
    # ),
    # "search_points": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/search",
    #     "qdrant.Points/Search",
    # ),
    # "search_points_batch": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/search/batch",
    #     "qdrant.Points/SearchBatch",
    # ),
    # "search_point_groups": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/search/groups",
    #     "qdrant.Points/SearchGroups",
    # ),
    # "recommend_points": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/recommend",
    #     "qdrant.Points/Recommend",
    # ),
    # "recommend_points_batch": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/recommend/batch",
    #     "qdrant.Points/RecommendBatch",
    # ),
    # "recommend_point_groups": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/recommend/groups",
    #     "qdrant.Points/RecommendGroups",
    # ),
    # "discover_points": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/discover",
    #     "qdrant.Points/Discover",
    # ),
    # "discover_points_batch": EndpointAccess(
    #     True,
    #     True,
    #     True,
    #     "POST /collections/{collection_name}/points/discover/batch",
    #     "qdrant.Points/DiscoverBatch",
    # ),
    # "count_points": EndpointAccess(
    #     True, True, True, "POST /collections/{collection_name}/points/count", "qdrant.Points/Count"
    # ),
    ### Service ###
    # TODO: add tests for these actions
    # "get_root": EndpointAccess(True, True, True, "GET /", "qdrant.Qdrant/HealthCheck"),
    # "readyz": EndpointAccess(True, True, True, "GET /readyz", "grpc.health.v1.Health/Check"),
    # "healthz": EndpointAccess(True, True, True, "GET /healthz", "grpc.health.v1.Health/Check"),
    # "livez": EndpointAccess(True, True, True, "GET /livez", "grpc.health.v1.Health/Check"),
    # "telemetry": EndpointAccess(True, False, True, "GET /telemetry"),
    # "metrics": EndpointAccess(True, False, True, "GET /metrics"),
    # "set_lock_options": EndpointAccess(False, False, True, "POST /locks"),
    # "get_lock_options": EndpointAccess(True, True, True, "GET /locks"),
}


def test_all_actions_have_tests():
    # for each action
    for action_name in ACTION_ACCESS.keys():
        # a test_{action_name} exists in this file
        test_name = f"test_{action_name}"
        assert (
            test_name in globals()
        ), f"An action is not tested: `{test_name}` was not found in this file"


def test_all_rest_endpoints_are_covered():
    # Load the JSON content from the openapi.json file
    with open("./docs/redoc/master/openapi.json", "r") as file:
        openapi_data = json.load(file)

    # Extract all endpoint paths
    endpoint_paths = []
    for path in openapi_data["paths"].keys():
        for method in openapi_data["paths"][path]:
            method = method.upper()
            endpoint_paths.append(f"{method} {path}")

    # check that all endpoints are covered in ACTION_ACCESS
    covered_endpoints = set(v.rest_endpoint for v in ACTION_ACCESS.values())
    for endpoint in endpoint_paths:
        assert (
            endpoint in covered_endpoints
        ), f"REST endpoint `{endpoint}` not found in any of the `ACTION_ACCESS` REST endpoints"


class MetadataInterceptor(ClientInterceptor):
    """A test interceptor that injects invocation metadata."""

    def __init__(self, metadata: List[Tuple[str, str]]):
        self._metadata = metadata

    def intercept(self, method, request_or_iterator, call_details: ClientCallDetails):
        """Add invocation metadata to request."""
        new_details = call_details._replace(metadata=self._metadata)
        return method(request_or_iterator, new_details)


def test_all_grpc_endpoints_are_covered():
    # read grpc services from the reflection server
    client: grpc_requests.Client = grpc_requests.Client(
        GRPC_URI, interceptors=[MetadataInterceptor(API_KEY_METADATA)]
    )

    # check that all endpoints are covered in GRPC_TO_REST_MAPPING
    covered_endpoints = set(v.grpc_endpoint for v in ACTION_ACCESS.values())

    for service_name in client.service_names:
        service = client.service(service_name)
        for method in service.method_names:
            grpc_endpoint = f"{service_name}/{method}"
            assert (
                grpc_endpoint in covered_endpoints
            ), f"gRPC endpoint `{grpc_endpoint}` not found in ACTION_ACCESS gRPC endpoints"


def start_api_key_instance(tmp_path: pathlib.Path) -> Tuple[str, str]:
    extra_env = {
        "QDRANT__SERVICE__API_KEY": SECRET,
        "QDRANT__SERVICE__JWT_RBAC": "true",
        "QDRANT__STORAGE__WAL__WAL_CAPACITY_MB": "1",
    }

    peer_dir = make_peer_folder(tmp_path, 0)

    (rest_uri, _bootstrap_uri) = start_first_peer(
        peer_dir, "api_key_peer.log", port=PORT_SEED, extra_env=extra_env
    )

    assert rest_uri == REST_URI

    time.sleep(0.5)

    def check_readyz(uri: str) -> bool:
        res = requests.get(f"{uri}/readyz")
        return res.ok

    wait_for(check_readyz, REST_URI)

    return REST_URI, GRPC_URI


@pytest.fixture(scope="module", autouse=True)
def uris(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("api_key_instance")

    rest_uri, grpc_uri = start_api_key_instance(tmp_path)

    fixtures.create_collection(
        rest_uri,
        collection=COLL_NAME,
        sharding_method="custom",
        headers=API_KEY_HEADERS,
    )

    requests.put(
        f"{rest_uri}/collections/{COLL_NAME}/shards",
        json={"shard_key": SHARD_KEY},
        headers=API_KEY_HEADERS,
    ).raise_for_status()

    fixtures.upsert_random_points(
        rest_uri, 100, COLL_NAME, shard_key=SHARD_KEY, headers=API_KEY_HEADERS
    )

    yield rest_uri, grpc_uri

    fixtures.drop_collection(rest_uri, COLL_NAME, headers=API_KEY_HEADERS)


def create_validation_collection(
    collection: str, timeout=10
):
    res = requests.put(
        f"{REST_URI}/collections/{collection}?timeout={timeout}",
        json={},
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()


def scroll_with_token(uri: str, collection: str, token: str) -> requests.Response:
    res = requests.post(
        f"{uri}/collections/{collection}/points/scroll",
        json={
            "limit": 10,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    res.raise_for_status()
    return res


def test_value_exists_claim(uris: Tuple[str, str]):
    rest_uri, grpc_uri = uris

    validation_collection = "secondary_test_collection"

    key = "tokenId"
    value = "token_42"

    claims = {
        "value_exists": {
            "collection": validation_collection,
            "matches": [{"key": key, "value": value}],
        },
    }
    token = encode_jwt(claims, SECRET)

    # Check that token does not work with unexisting collection
    with pytest.raises(requests.HTTPError):
        scroll_with_token(rest_uri, COLL_NAME, token)

    # Create collection
    create_validation_collection(validation_collection)

    # Check it does not work now
    with pytest.raises(requests.HTTPError):
        res = scroll_with_token(rest_uri, COLL_NAME, token)

    # Upload validation point
    res = requests.put(
        f"{rest_uri}/collections/{validation_collection}/points?wait=true",
        json={
            "points": [
                {
                    "id": 42,
                    "vectors": {},
                    "payload": {key: value},
                }
            ]
        },
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()

    # Check that token works now
    res = scroll_with_token(rest_uri, COLL_NAME, token)
    assert len(res.json()["result"]["points"]) == 10

    # Delete validation point
    res = requests.post(
        f"{rest_uri}/collections/{validation_collection}/points/delete?wait=true",
        json={"points": [42]},
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()

    # Check it does not work now
    with pytest.raises(requests.HTTPError):
        scroll_with_token(rest_uri, COLL_NAME, token)

    fixtures.drop_collection(rest_uri, validation_collection, headers=API_KEY_HEADERS)
    fixtures.drop_collection(rest_uri, validation_collection, headers=API_KEY_HEADERS)


def check_rest_access(
    method: str,
    path: str,
    body: Optional[Union[dict, Callable[[], dict]]],
    should_succeed: bool,
    token: str,
    path_params: dict = {},
    request_kwargs: dict = {},
):
    if isfunction(body):
        body = body()

    concrete_path_params = {}
    for key, value in path_params.items():
        concrete_path_params[key] = value() if isfunction(value) else value

    path = path.format(**concrete_path_params)

    res = requests.request(
        method,
        f"{REST_URI}{path}",
        headers={"authorization": f"Bearer {token}"},
        json=body,
        **request_kwargs,
    )

    if should_succeed:
        assert res.ok, f"{method} {path} failed with {res.status_code}: {res.text}"
    else:
        assert res.status_code in [
            401,
            403,
        ], f"{method} {path} failed with {res.status_code}: {res.text}"


def check_grpc_access(
    client: grpc_requests.Client,
    service: str,
    method: str,
    request: Optional[dict],
    should_succeed: bool,
):
    if isfunction(request):
        request = request()

    try:
        _res = client.request(service=service, method=method, request=request)
    except grpc.RpcError as e:
        if should_succeed:
            pytest.fail(f"{service}/{method} failed with {e.code()}: {e.details()}")
        else:
            assert e.code() == grpc.StatusCode.PERMISSION_DENIED


class GrpcClients:
    def __init__(self):
        self.r = grpc_requests.Client(
            GRPC_URI, interceptors=[MetadataInterceptor([("authorization", f"Bearer {TOKEN_R}")])]
        )
        self.coll_r = grpc_requests.Client(
            GRPC_URI,
            interceptors=[MetadataInterceptor([("authorization", f"Bearer {TOKEN_COLL_R}")])],
        )
        self.coll_rw = grpc_requests.Client(
            GRPC_URI,
            interceptors=[MetadataInterceptor([("authorization", f"Bearer {TOKEN_COLL_RW}")])],
        )
        self.m = grpc_requests.Client(
            GRPC_URI, interceptors=[MetadataInterceptor([("authorization", f"Bearer {TOKEN_M}")])]
        )


def get_auth_grpc_clients() -> GrpcClients:
    global _cached_clients
    if _cached_clients is None:
        _cached_clients = GrpcClients()

    return _cached_clients


def check_access(
    action_name: str, rest_request=None, grpc_request=None, path_params={}, rest_req_kwargs={}
):
    action_access: EndpointAccess = ACTION_ACCESS[action_name]

    ## Check Rest
    assert isinstance(action_access, EndpointAccess)

    method, path = action_access.rest_endpoint.split(" ")

    allowed_for = action_access.access

    check_rest_access(
        method, path, rest_request, allowed_for.read, TOKEN_R, path_params, rest_req_kwargs
    )
    check_rest_access(
        method, path, rest_request, allowed_for.coll_r, TOKEN_COLL_R, path_params, rest_req_kwargs
    )
    check_rest_access(
        method,
        path,
        rest_request,
        allowed_for.coll_rw,
        TOKEN_COLL_RW,
        path_params,
        rest_req_kwargs,
    )
    check_rest_access(
        method, path, rest_request, allowed_for.manage, TOKEN_M, path_params, rest_req_kwargs
    )

    ## Check GRPC
    grpc_endpoint = action_access.grpc_endpoint
    if grpc_endpoint is not None:
        service = grpc_endpoint.split("/")[0]
        method = grpc_endpoint.split("/")[1]

        allowed_for = action_access.access

        grpc = get_auth_grpc_clients()

        check_grpc_access(grpc.r, service, method, grpc_request, allowed_for.read)
        check_grpc_access(grpc.coll_r, service, method, grpc_request, allowed_for.coll_r)
        check_grpc_access(grpc.coll_rw, service, method, grpc_request, allowed_for.coll_rw)
        check_grpc_access(grpc.m, service, method, grpc_request, allowed_for.manage)


def test_list_collections():
    check_access("list_collections")


def test_get_collection():
    check_access(
        "get_collection",
        grpc_request={"collection_name": COLL_NAME},
        path_params={"collection_name": COLL_NAME},
    )


def test_create_collection():
    def grpc_req():
        return {"collection_name": random_str()}

    check_access(
        "create_collection",
        rest_request={},
        grpc_request=grpc_req,
        path_params={"collection_name": lambda: random_str()},
    )


def test_delete_collection():
    # create collections
    coll_names = [random_str() for _ in range(10)]
    for collection_name in coll_names:
        requests.put(
            f"{REST_URI}/collections/{collection_name}", json={}, headers=API_KEY_HEADERS
        ).raise_for_status()

    coll_names_iter = iter(coll_names)

    def grpc_req():
        return {"collection_name": next(coll_names_iter)}

    check_access(
        "delete_collection",
        grpc_request=grpc_req,
        path_params={"collection_name": lambda: next(coll_names_iter)},
    )

    # teardown
    for collection_name in coll_names:
        requests.delete(f"{REST_URI}/collections/{collection_name}", headers=API_KEY_HEADERS)


def test_update_collection_params():
    check_access(
        "update_collection_params",
        rest_request={},
        grpc_request={"collection_name": COLL_NAME},
        path_params={"collection_name": COLL_NAME},
    )


def test_create_alias():
    def req():
        return {
            "actions": [
                {
                    "create_alias": {
                        "collection_name": COLL_NAME,
                        "alias_name": random_str(),
                    }
                }
            ]
        }

    check_access(
        "create_alias",
        rest_request=req,
        grpc_request=req,
    )


def test_rename_alias():
    alias_names = [random_str() for _ in range(10)]

    for alias in alias_names:
        requests.post(
            f"{REST_URI}/collections/aliases",
            json={
                "actions": [
                    {
                        "create_alias": {
                            "collection_name": COLL_NAME,
                            "alias_name": alias,
                        }
                    }
                ]
            },
            headers=API_KEY_HEADERS,
        ).raise_for_status()

    names_iter = iter(alias_names)

    def req():
        return {
            "actions": [
                {
                    "rename_alias": {
                        "old_alias_name": next(names_iter),
                        "new_alias_name": random_str(),
                    }
                }
            ]
        }

    check_access(
        "rename_alias",
        rest_request=req,
        grpc_request=req,
    )


def test_delete_alias():
    alias_names = [random_str() for _ in range(10)]
    deletable_aliases = iter(alias_names)

    for alias in alias_names:
        requests.post(
            f"{REST_URI}/collections/aliases",
            json={
                "actions": [
                    {
                        "create_alias": {
                            "collection_name": COLL_NAME,
                            "alias_name": alias,
                        }
                    }
                ]
            },
            headers=API_KEY_HEADERS,
        ).raise_for_status()

    def req():
        return {"actions": [{"delete_alias": {"alias_name": next(deletable_aliases)}}]}

    check_access(
        "delete_alias",
        rest_request=req,
        grpc_request=req,
    )


def test_list_collection_aliases():
    check_access(
        "list_collection_aliases",
        grpc_request={"collection_name": COLL_NAME},
        path_params={"collection_name": COLL_NAME},
    )


def test_list_aliases():
    check_access("list_aliases")


def test_get_collection_cluster_info():
    check_access(
        "get_collection_cluster_info",
        path_params={"collection_name": COLL_NAME},
        grpc_request={"collection_name": COLL_NAME},
    )


def test_collection_exists():
    check_access(
        "collection_exists",
        path_params={"collection_name": COLL_NAME},
        grpc_request={"collection_name": COLL_NAME},
    )


def test_create_default_shard_key():
    def rest_req():
        return {"shard_key": random_str()}

    def grpc_req():
        return {
            "collection_name": COLL_NAME,
            "request": {"shard_key": {"keyword": random_str()}},
        }

    check_access(
        "create_default_shard_key",
        rest_request=rest_req,
        path_params={"collection_name": COLL_NAME},
        grpc_request=grpc_req,
    )


def test_create_custom_shard_key():
    def rest_req():
        return {"shard_key": random_str(), "replication_factor": 3}

    def grpc_req():
        return {
            "collection_name": COLL_NAME,
            "request": {"shard_key": {"keyword": random_str()}, "replication_factor": 3},
        }

    check_access(
        "create_custom_shard_key",
        rest_request=rest_req,
        path_params={"collection_name": COLL_NAME},
        grpc_request=grpc_req,
    )


def test_delete_shard_key():
    deletable_shard_keys = [random_str() for _ in range(10)]

    for shard_key in deletable_shard_keys:
        requests.put(
            f"{REST_URI}/collections/{COLL_NAME}/shards",
            json={"shard_key": shard_key},
            headers=API_KEY_HEADERS,
        ).raise_for_status()

    keys_iter = iter(deletable_shard_keys)

    def rest_req():
        return {"shard_key": next(keys_iter)}

    def grpc_req():
        return {
            "collection_name": COLL_NAME,
            "request": {"shard_key": {"keyword": next(keys_iter)}},
        }

    check_access(
        "delete_shard_key",
        rest_request=rest_req,
        path_params={"collection_name": COLL_NAME},
        grpc_request=grpc_req,
    )


def test_create_index():
    check_access(
        "create_index",
        rest_request={"field_name": FIELD_NAME, "field_schema": "keyword"},
        path_params={"collection_name": COLL_NAME},
        grpc_request={"collection_name": COLL_NAME, "field_name": FIELD_NAME, "field_type": 0},
    )


def test_delete_index():
    check_access(
        "delete_index",
        path_params={"collection_name": COLL_NAME, "field_name": "fake_field_name"},
        grpc_request={"collection_name": COLL_NAME, "field_name": "fake_field_name"},
    )


def test_list_collection_snapshots():
    check_access(
        "list_collection_snapshots",
        path_params={"collection_name": COLL_NAME},
        grpc_request={"collection_name": COLL_NAME},
    )


def test_create_collection_snapshot():
    check_access(
        "create_collection_snapshot",
        path_params={"collection_name": COLL_NAME},
        grpc_request={"collection_name": COLL_NAME},
    )


def test_delete_collection_snapshot():
    # create snapshots
    snapshot_names = []
    for _ in range(8):
        res = requests.post(
            f"{REST_URI}/collections/{COLL_NAME}/snapshots?wait=true",
            headers=API_KEY_HEADERS,
        )
        res.raise_for_status()
        filename = res.json()["result"]["name"]
        snapshot_names.append(filename)
        # names are only different if they are 1 second apart
        time.sleep(1)

    snapshot_names_iter = iter(snapshot_names)

    def grpc_req():
        return {"collection_name": COLL_NAME, "snapshot_name": next(snapshot_names_iter)}

    check_access(
        "delete_collection_snapshot",
        path_params={
            "collection_name": COLL_NAME,
            "snapshot_name": lambda: next(snapshot_names_iter),
        },
        grpc_request=grpc_req,
    )


def test_download_collection_snapshot():
    res = requests.post(
        f"{REST_URI}/collections/{COLL_NAME}/snapshots?wait=true",
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()
    filename = res.json()["result"]["name"]

    check_access(
        "download_collection_snapshot",
        path_params={"collection_name": COLL_NAME, "snapshot_name": filename},
    )


@pytest.fixture(scope="module")
def collection_snapshot():
    res = requests.post(
        f"{REST_URI}/collections/{COLL_NAME}/snapshots?wait=true",
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()
    filename = res.json()["result"]["name"]

    res = requests.get(
        f"{REST_URI}/collections/{COLL_NAME}/snapshots/{filename}",
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()

    return res.content


def test_upload_collection_snapshot(collection_snapshot: bytes):
    check_access(
        "upload_collection_snapshot",
        rest_req_kwargs={"files": {"snapshot": collection_snapshot}},
        path_params={"collection_name": COLL_NAME},
    )


def test_recover_collection_snapshot(collection_snapshot: bytes):
    # Save file to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix=".snapshot")
    temp_file.write(collection_snapshot)
    temp_file.seek(0)
    file = temp_file.name

    check_access(
        "recover_collection_snapshot",
        rest_request={"location": f"file://{file}"},
        path_params={"collection_name": COLL_NAME},
    )


@pytest.fixture(scope="module")
def shard_snapshot_name():
    res = requests.post(
        f"{REST_URI}/collections/{COLL_NAME}/shards/{SHARD_ID}/snapshots?wait=true",
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()
    return res.json()["result"]["name"]


@pytest.fixture(scope="module")
def shard_snapshot(shard_snapshot_name):
    res = requests.get(
        f"{REST_URI}/collections/{COLL_NAME}/shards/{SHARD_ID}/snapshots/{shard_snapshot_name}",
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()
    return res.content


def test_upload_shard_snapshot(shard_snapshot: bytes):
    check_access(
        "upload_shard_snapshot",
        rest_req_kwargs={"files": {"snapshot": shard_snapshot}},
        path_params={"collection_name": COLL_NAME, "shard_id": SHARD_ID},
    )


def test_recover_shard_snapshot(shard_snapshot_name: str):
    check_access(
        "recover_shard_snapshot",
        rest_request={"location": shard_snapshot_name},
        path_params={"collection_name": COLL_NAME, "shard_id": SHARD_ID},
    )


def test_list_shard_snapshots():
    check_access(
        "list_shard_snapshots",
        path_params={"collection_name": COLL_NAME, "shard_id": SHARD_ID},
    )


def test_create_shard_snapshot():
    check_access(
        "create_shard_snapshot",
        path_params={"collection_name": COLL_NAME, "shard_id": SHARD_ID},
    )


def test_delete_shard_snapshot():
    shard_snapshot_names = []
    for _ in range(8):
        res = requests.post(
            f"{REST_URI}/collections/{COLL_NAME}/shards/{SHARD_ID}/snapshots?wait=true",
            headers=API_KEY_HEADERS,
        )
        res.raise_for_status()
        filename = res.json()["result"]["name"]
        shard_snapshot_names.append(filename)
        # names are only different if they are 1 second apart
        time.sleep(1)

    snapshot_names_iter = iter(shard_snapshot_names)

    def grpc_req():
        return {
            "collection_name": COLL_NAME,
            "shard_id": SHARD_ID,
            "snapshot_name": next(snapshot_names_iter),
        }

    check_access(
        "delete_shard_snapshot",
        path_params={
            "collection_name": COLL_NAME,
            "shard_id": SHARD_ID,
            "snapshot_name": lambda: next(snapshot_names_iter),
        },
        grpc_request=grpc_req,
    )


def test_download_shard_snapshot(shard_snapshot_name: str):
    check_access(
        "download_shard_snapshot",
        path_params={
            "collection_name": COLL_NAME,
            "shard_id": SHARD_ID,
            "snapshot_name": shard_snapshot_name,
        },
    )


def test_list_full_snapshots():
    check_access("list_full_snapshots")


def test_create_full_snapshot():
    check_access("create_full_snapshot")


def test_delete_full_snapshot():
    snapshot_names = []
    for _ in range(8):
        res = requests.post(
            f"{REST_URI}/snapshots?wait=true",
            headers=API_KEY_HEADERS,
        )
        res.raise_for_status()
        filename = res.json()["result"]["name"]
        snapshot_names.append(filename)
        # names are only different if they are 1 second apart
        time.sleep(1)

    snapshot_names_iter = iter(snapshot_names)

    def grpc_req():
        return {"snapshot_name": next(snapshot_names_iter)}

    check_access(
        "delete_full_snapshot",
        path_params={"snapshot_name": lambda: next(snapshot_names_iter)},
        grpc_request=grpc_req,
    )


def test_download_full_snapshot():
    res = requests.post(
        f"{REST_URI}/snapshots?wait=true",
        headers=API_KEY_HEADERS,
    )
    res.raise_for_status()
    name = res.json()["result"]["name"]

    check_access(
        "download_full_snapshot",
        path_params={"snapshot_name": name},
    )


def test_get_cluster():
    check_access("get_cluster")


def test_recover_raft_state():
    check_access("recover_raft_state")


def test_delete_peer():
    check_access("delete_peer", path_params={"peer_id": "2000"})
