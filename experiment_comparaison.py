import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
import os
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
import subprocess
import shutil


def main():
    # Initialize the mlflow client to get the artifact
    client = MlflowClient()

    # Set the expirement by name
    expirement_name = "cats-and-dogs"
    model_name = "cats-and-dogs"
    experiment = mlflow.get_experiment_by_name(expirement_name)

    # Get infos about runs such as stage of the run and validation accuracy, and return the best run id
    def print_auto_logged_info(run_infos):
        best_run = None
        for r in run_infos:
            print(
                "- run_id: {}, lifecycle_stage: {}".format(r.run_id, r.lifecycle_stage)
            )
            print(
                "val_accuracy : ", mlflow.get_run(r.run_id).data.metrics["val_accuracy"]
            )
            best_run = r.run_id
        return best_run

    def get_model_s3_src(best_run):
        run_uri = "runs:/{}/model/data/model".format(best_run)
        model_src = RunsArtifactRepository.get_underlying_uri(run_uri)
        return model_src

    def get_model_deployment_s3_src(best_run):
        run_uri = "runs:/{}/deployment-model/".format(best_run, model_name)
        model_src = RunsArtifactRepository.get_underlying_uri(run_uri)
        return model_src

    def create_registered_model_version(model_name):
        try:
            registered_model = client.create_registered_model(model_name)
        except:
            print("Model already registered")
            return False
        return True

    def get_latest_model_version_info(model_name):
        try:
            model_latest_version = client.get_latest_versions(
                model_name, stages=["None"]
            )
            run_id = model_latest_version[0].run_id
            version = model_latest_version[0].version
            return model_latest_version[0].run_id, model_latest_version[0].version
        except:
            run_id = -1
            version = -1
            return run_id, version

    def check_accuracy_improvment(latest_run_id, best_run):
        improvment = False if best_run == latest_run_id else True
        return improvment

    def register_the_best_model(best_run):
        result = None
        try:
            result = client.create_model_version(
                name=expirement_name,
                source=get_model_deployment_s3_src(best_run),
                run_id=best_run,
            )
        except:
            return "Something went wrong"
        return result

    def save_best_model_artifact_uri(model_name, model_version):
        artifact_uri = client.get_model_version_download_uri(model_name, model_version)
        file = open("/tmp/best_model_artifact_uri.txt", "w+")
        file.write(artifact_uri)
        file.close()
        file = open("/tmp/model_name_version.txt","w+")
        file.write("{}-{}".format(model_name,model_version))
        file.close()
        return artifact_uri

    # Return the best val accuracy run of The choosen exepiremnt
    best_run = print_auto_logged_info(
        mlflow.list_run_infos(
            experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1,
            order_by=["metric.val_accuracy DESC"],
        )
    )
    # df = mlflow.search_runs([experiment_id], order_by=["metrics.m DESC"]) we can do same using this function

    # We gonna create a tmp folder to store the artifacts into
    local_dir = "/tmp/artifact_downloads"
    # local_dir = os.path.join(os.getcwd(), "tmp" )
    if not os.path.exists(local_dir):
        os.mkdir(local_dir)

    models_dir = "/tmp/models/model-store"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Get the model Artifact from the storage bucket
    local_path = client.download_artifacts(best_run, "model", local_dir)

    register_model = create_registered_model_version(model_name)
    latest_run_id, latest_model_version = get_latest_model_version_info(model_name)

    if check_accuracy_improvment(latest_run_id, best_run) or latest_model_version == -1:
        registered_model = register_the_best_model(best_run)
        if latest_model_version == -1 :
            latest_model_version = 1
    else:
        print("There is no improvement in the model")

    model_file = "/tmp/artifact_downloads/model/data/cats-and-dogs/model.py"
    serialized_file = "/tmp/artifact_downloads/model/data/cats-and-dogs/model.pt"
    extra_files = "/tmp/artifact_downloads/model/data/cats-and-dogs/index_to_name.json"
    handler_file = "/tmp/artifact_downloads/model/data/cats-and-dogs/handler.py"

    result = os.system("torch-model-archiver --model-name {} --version 2.0 --model-file {} --serialized-file {} --extra-files {} --handler {} --export-path {} ".format(model_name,model_file,serialized_file,extra_files,handler_file,models_dir))


    config_dir = "/tmp/models/config"
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    shutil.copy2("./config.properties", "/tmp/models/config/")

    client.log_artifacts(best_run, "/tmp/models/", "deployment-model")

    artifact_uri = save_best_model_artifact_uri(model_name, int(latest_model_version))
    print(artifact_uri)

    # client.download_artifacts(best_run, "deployment-model", "/Users/safoinpers/MLArgo/mnist_example/")
    # /tmp/artifact_downloads/model/data/cats-and-dogs


if __name__ == "__main__":
    # mlflow.mlflow.set_tracking_uri("http://127.0.0.1:5000")
    main()
