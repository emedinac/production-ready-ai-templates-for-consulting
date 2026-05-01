#!/bin/bash
PROJECT_NAME=$1

if [ -z "$PROJECT_NAME" ]; then
    echo "Usage: ./scripts/setup.sh <project_name>"
    exit 1
fi

find . -type f \( -name "*.py" -o -name "*.toml" -o -name "*.yaml" -o -name "Makefile" \) \
    -exec sed -i "s/CHANGE_ME/$PROJECT_NAME/g" {} +

echo "Done. Project name set to: $PROJECT_NAME"