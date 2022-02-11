import pytest

from ..openapi_integration.helpers import request_with_validation
from ..openapi_integration.collection_setup import basic_collection_setup, drop_collection

collection_name = 'test_collection_payload_indexing'


@pytest.fixture(autouse=True)
def setup():
    basic_collection_setup(collection_name=collection_name)
    yield
    drop_collection(collection_name=collection_name)


def test_payload_indexing_operations():
    # create payload
    response = request_with_validation(
        api='/collections/{name}/points/payload',
        method="POST",
        path_params={'name': collection_name},
        query_params={'wait': 'true'},
        body={
            "payload": {"test_payload": "keyword"},
            "points": [6]
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{name}',
        method="GET",
        path_params={'name': collection_name},
    )
    assert response.ok
    assert not response.json()['result']['payload_schema']['test_payload']['indexed']

    # Create index
    response = request_with_validation(
        api='/collections/{name}/index',
        method="PUT",
        path_params={'name': collection_name},
        query_params={'wait': 'true'},
        body={
            "field_name": "test_payload"
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{name}',
        method="GET",
        path_params={'name': collection_name},
    )
    assert response.ok
    assert response.json()['result']['payload_schema']['test_payload']['indexed']

    # Delete index
    response = request_with_validation(
        api='/collections/{name}/index/{field_name}',
        method="DELETE",
        path_params={'name': collection_name, 'field_name': 'test_payload'},
        query_params={'wait': 'true'},
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{name}',
        method="GET",
        path_params={'name': collection_name},
    )
    assert response.ok
    assert not response.json()['result']['payload_schema']['test_payload']['indexed']

