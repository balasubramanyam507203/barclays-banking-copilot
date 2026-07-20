#!/bin/sh
set -eu

echo "Applying Alembic database migrations..."
alembic upgrade head

build_index_on_startup="${
    BUILD_INDEX_ON_STARTUP:-true
}"
force_rebuild_index="${
    FORCE_REBUILD_INDEX:-false
}"
index_directory="${
    LOCAL_INDEX_DIRECTORY:-local_indexes
}"

should_build_index="false"

if [ "$force_rebuild_index" = "true" ]; then
    should_build_index="true"
elif [ "$build_index_on_startup" = "true" ]; then
    if [ ! -d "$index_directory" ]; then
        should_build_index="true"
    elif ! find "$index_directory" \
        -type f \
        -print \
        -quit \
        | grep -q .; then
        should_build_index="true"
    fi
fi

if [ "$should_build_index" = "true" ]; then
    echo "Building the local FAISS policy index..."
    python -m app.build_index
else
    echo "Using the existing local FAISS policy index."
fi

echo "Starting the FastAPI backend..."

exec uvicorn app.api.main:app \
    --host 0.0.0.0 \
    --port 8000
