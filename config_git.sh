#!/bin/sh
set -e -x

# Verify our environment variables are set
[ -z "${DVC_BUCKET}" ] && { echo "Need to set GIT_REPO"; exit 1; }
[ -z "${DVC_S3_ENDPOINT_URL}" ] && { echo "Need to set GIT_BRANCH"; exit 1; }
[ -z "${AWS_ACCESS_KEY_ID}" ] && { echo "Need to set COMMIT_USER"; exit 1; }
[ -z "${AWS_SECRET_ACCESS_KEY}" ] && { echo "Need to set COMMIT_EMAIL"; exit 1; }


dvc remote add minio ${DVC_BUCKET} --global
dvc config core.remote minio 
dvc remote modify minio endpointurl ${DVC_S3_ENDPOINT_URL} --global 
dvc remote modify minio access_key_id "${AWS_ACCESS_KEY_ID}" --global
dvc remote modify minio secret_access_key "${AWS_SECRET_ACCESS_KEY}" --global
dvc pull

