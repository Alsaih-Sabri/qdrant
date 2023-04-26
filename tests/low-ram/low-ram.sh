#!/bin/bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

docker build ../../ --tag=qdrant_consensus

if [[ ! -e storage ]]
then
    git lfs pull
    tar -xf storage.tar.xz
fi

declare container && container=$(
    docker run --rm -d \
        -m 128m \
        -p 127.0.0.1:6333:6333 \
        -p 127.0.0.1:6334:6334 \
        -v $PWD/storage:/qdrant/storage \
        -e QDRANT__STORAGE__HANDLE_COLLECTION_LOAD_ERRORS=true \
        qdrant_consensus
)

function cleanup {
    docker stop $container
}

trap cleanup EXIT

# Wait (up to ~30 seconds) for the service to start
declare retry=0
while [[ $(curl -sS localhost:6333 -w ''%{http_code}'' -o /dev/null) != 200 ]]
do
    if (( retry++ < 30 ))
    then
        sleep 1
    else
        echo "Service failed to start in ~30 seconds" >&2
        exit 1
    fi
done

# Wait (up to ~10 seconds) until `low-ram` collection is loaded
declare retry=0
while ! curl -sS --fail-with-body localhost:6333/collections | jq -e '.result.collections | index({"name": "low-ram"})' &>/dev/null
do
    if (( retry++ < 10 ))
    then
        sleep 1
    else
        echo "Collection failed to load in ~10 seconds" >&2
        exit 2
    fi
done

# Check that there's a "dummy" shard log message in service logs
declare DUMMY_SHARD_MSG='initializing "dummy" shard'

if ! docker logs "$container" 2>&1 | grep "$DUMMY_SHARD_MSG"
then
    echo "'$DUMMY_SHARD_MSG' log message not found in $container container logs" >&2
    exit 3
fi

# Check that there's a "low RAM" log message in service logs
declare LOW_RAM_MSG='segment load aborted to prevent OOM'

if ! docker logs "$container" 2>&1 | grep "$LOW_RAM_MSG"
then
    echo "'$LOW_RAM_MSG' log message not found in $container container logs" >&2
    exit 4
fi

# Check that `low-ram` collection initialized as a "dummy" shard
# (e.g., collection info request returns HTTP status 500)
declare status; status="$(curl -sS localhost:6333/collections/low-ram -w ''%{http_code}'' -o /dev/null)"

if (( status != 500 ))
then
    echo "Collection info request returned an unexpected HTTP status: expected 500, but received $STATUS" >&2
    exit 5
fi
