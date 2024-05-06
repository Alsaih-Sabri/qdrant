#!/usr/bin/env bash

set -ex

# Ensure current path is project root
cd "$(dirname "$0")/../"

QDRANT_HOST='localhost:6334'

docker_grpcurl="docker run --rm --network=host -v ${PWD}/lib/api/src/grpc/proto:/proto fullstorydev/grpcurl -plaintext -import-path /proto -proto qdrant.proto"

$docker_grpcurl -d '{
   "collection_name": "test_multivector_collection"
}' $QDRANT_HOST qdrant.Collections/Delete

$docker_grpcurl -d '{
   "collection_name": "test_multivector_collection",
   "vectors_config": {
      "params_map": {
        "map": {
          "my-multivec": {
            "size": 4,
            "distance": "Dot",
            "multivector_config" : { 
              "comparator": "MaxSim"
            }
          }
        }
      }
   }
}' $QDRANT_HOST qdrant.Collections/Create

$docker_grpcurl -d '{}' $QDRANT_HOST qdrant.Collections/List

$docker_grpcurl -d '{
  "collection_name": "test_multivector_collection",
  "wait": true,
  "ordering": null,
  "points": [
    {
      "id": { "num": 1 },
      "vectors": {
        "vectors": {
          "vectors": {
            "my-multivec": {
              "data": [0.05, 0.61, 0.76, 0.74],
              "vector_count": 1
            }
          }
        }
      },
      "payload": {
        "city": { "string_value": "Berlin" },
        "country":  { "string_value": "Germany" },
        "population": { "integer_value":  1000000 },
        "square": { "double_value": 12.5 },
        "coords": { "struct_value": { "fields": { "lat": { "double_value": 1.0 }, "lon": { "double_value": 2.0 } } } }
      }
    },
    {
      "id": { "num": 2 },
       "vectors": {
         "vectors": {
           "vectors": {
             "my-multivec": {
               "data": [0.19, 0.81, 0.75, 0.11],
               "vector_count": 1
             }
           }
         }
       },
       "payload": {"city": {"list_value": {"values": [{ "string_value": "Berlin" }, { "string_value": "London" }]}}}
    },
    {
      "id": { "num": 3 },
      "vectors": {
        "vectors": {
          "vectors": {
            "my-multivec": {
               "data": [0.36, 0.55, 0.47, 0.94],
               "vector_count": 1
            }
          }
        }
      },
      "payload": {"city": {"list_value": {"values": [{ "string_value": "Berlin" }, { "string_value": "Moscow" }]}}}
    },
    {
      "id": { "num": 4 },
      "vectors": {
        "vectors": {
          "vectors": {
            "my-multivec": {
              "data": [0.18, 0.01, 0.85, 0.80],
              "vector_count": 1
            }
          }
        }
      },
      "payload": {"city": {"list_value": {"values": [{ "string_value": "London" }, { "string_value": "Moscow" }]}}}
    },
    {
      "id": { "uuid": "98a9a4b1-4ef2-46fb-8315-a97d874fe1d7" },
      "vectors": {
        "vectors": {
          "vectors": {
            "my-multivec": {
              "data": [0.24, 0.18, 0.22, 0.44],
              "vector_count": 1
            }
          }
        }
      },
      "payload": {"count":{"list_value": {"values": [{ "integer_value": 0 }]}}}
    },
    {
      "id": { "uuid": "f0e09527-b096-42a8-94e9-ea94d342b925" },
      "vectors": {
        "vectors": {
          "vectors": {
            "my-multivec": {
              "data": [0.35, 0.08, 0.11, 0.44],
              "vector_count": 1
            }
          }
        }
      }
    }
  ]
}' $QDRANT_HOST qdrant.Points/Upsert

$docker_grpcurl -d '{ "collection_name": "test_multivector_collection" }' $QDRANT_HOST qdrant.Collections/Get

$docker_grpcurl -d '{
  "collection_name": "test_multivector_collection",
  "limit": 2,
  "with_vectors": {"enable": true},
  "filter": {
    "should": [
      {
        "field": {
          "key": "city",
          "match": {
            "keyword": "London"
          }
        }
      }
    ]
  }
}' $QDRANT_HOST qdrant.Points/Scroll

$docker_grpcurl -d '{
  "collection_name": "test_multivector_collection",
  "with_vectors": {"enable": true},
  "ids": [{ "num": 2 }, { "num": 3 }, { "num": 4 }]
}' $QDRANT_HOST qdrant.Points/Get
