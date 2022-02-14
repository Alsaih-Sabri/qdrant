import pytest

from openapi_integration.helpers.helpers import request_with_validation
from openapi_integration.helpers.collection_setup import basic_collection_setup, drop_collection

collection_name = 'test_collection_uuid'


@pytest.fixture(autouse=True)
def setup():
    basic_collection_setup(collection_name=collection_name)
    yield
    drop_collection(collection_name=collection_name)


def test_collection_update():
    response = request_with_validation(
        api='/collections/{name}',
        method="PATCH",
        path_params={'name': collection_name},
        body={
            "optimizers_config": {
                "default_segment_number": 6,
                "indexing_threshold": 10_000,
            }
        }
    )
    assert response.ok

    response = request_with_validation(
        api='/collections/{name}/points/{id}',
        method="GET",
        path_params={'name': collection_name, 'id': 6},
    )
    assert response.ok
