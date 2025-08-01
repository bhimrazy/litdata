import datetime
import sys
from pathlib import Path
from unittest import mock

import pytest
from lightning_sdk.lightning_cloud import login
from lightning_sdk.lightning_cloud.openapi import (
    Externalv1Cluster,
    V1AwsDataConnection,
    V1AWSDirectV1,
    V1CloudSpace,
    V1ClusterSpec,
    V1DataConnection,
    V1ListCloudSpacesResponse,
    V1ListClustersResponse,
    V1ListDataConnectionsResponse,
    V1S3FolderDataConnection,
)

from litdata.streaming import resolver


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_s3_connections(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    with pytest.raises(
        RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables."
    ):
        resolver._resolve_dir("/teamspace/s3_connections/imagenet")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[V1DataConnection(name="imagenet", aws=V1AwsDataConnection(source="s3://imagenet-bucket"))],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    assert resolver._resolve_dir("/teamspace/s3_connections/imagenet").url == "s3://imagenet-bucket"
    assert resolver._resolve_dir("/teamspace/s3_connections/imagenet/train").url == "s3://imagenet-bucket/train"

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(ValueError, match="name `imagenet`"):
        assert resolver._resolve_dir("/teamspace/s3_connections/imagenet")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_studios(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLUSTER_ID`"):
        resolver._resolve_dir("/teamspace/studios/other_studio")

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID`"):
        resolver._resolve_dir("/teamspace/studios/other_studio")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="other_studio", id="other_studio_id", cluster_id="cluster_id_of_other_studio")],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[
            Externalv1Cluster(
                id="cluster_id_of_other_studio", spec=V1ClusterSpec(aws_v1=V1AWSDirectV1(bucket_name="my_bucket"))
            )
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "s3://my_bucket/projects/project_id/cloudspaces/other_studio_id/code/content"
    assert resolver._resolve_dir("/teamspace/studios/other_studio").url == expected
    assert resolver._resolve_dir("/teamspace/studios/other_studio/train").url == f"{expected}/train"

    datetime_mock = mock.MagicMock()
    now_mock = mock.MagicMock()
    called = False

    def fn(pattern):
        nonlocal called
        called = True
        assert pattern == "%Y-%m-%d-%H-%M-%S"
        import datetime

        return datetime.datetime(2023, 12, 1, 10, 37, 40, 281942).strftime(pattern)

    now_mock.strftime = fn
    datetime_mock.datetime.now.return_value = now_mock
    monkeypatch.setattr(resolver, "datetime", datetime_mock)
    assert (
        resolver._resolve_dir("/teamspace/studios/other_studio/{%Y-%m-%d-%H-%M-%S}").path
        == "/teamspace/studios/other_studio/2023-12-01-10-37-40"
    )
    assert called

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(ValueError, match="other_studio`"):
        resolver._resolve_dir("/teamspace/studios/other_studio")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_s3_folders(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(name="debug_folder", s3_folder=V1S3FolderDataConnection(source="s3://imagenet-bucket"))
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "s3://imagenet-bucket"
    assert resolver._resolve_dir("/teamspace/s3_folders/debug_folder").url == expected
    assert resolver._resolve_dir("/teamspace/s3_folders/debug_folder/a/b/c").url == expected + "/a/b/c"
    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_datasets(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    assert resolver._resolve_dir("s3://bucket_name").url == "s3://bucket_name"

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLUSTER_ID`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    with pytest.raises(RuntimeError, match="`LIGHTNING_CLOUD_SPACE_ID`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "cloud_space_id")

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="other_studio", id="cloud_space_id", cluster_id="cluster_id_of_other_studio")],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[
            Externalv1Cluster(
                id="cluster_id_of_other_studio", spec=V1ClusterSpec(aws_v1=V1AWSDirectV1(bucket_name="my_bucket"))
            )
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "s3://my_bucket/projects/project_id/datasets/imagenet"
    assert resolver._resolve_dir("/teamspace/datasets/imagenet").url == expected
    assert resolver._resolve_dir("/teamspace/datasets/imagenet/train").url == f"{expected}/train"

    client_mock = mock.MagicMock()
    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[],
    )

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    with pytest.raises(ValueError, match="cloud_space_id`"):
        resolver._resolve_dir("/teamspace/datasets/imagenet")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_dst_resolver_dataset_path(monkeypatch, lightning_cloud_mock):
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    assert resolver._resolve_dir("something").url is None

    monkeypatch.setenv("LIGHTNING_CLUSTER_ID", "cluster_id")
    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")
    monkeypatch.setenv("LIGHTNING_CLOUD_SPACE_ID", "cloud_space_id")

    client_mock = mock.MagicMock()

    client_mock.cluster_service_list_project_clusters.return_value = V1ListClustersResponse(
        clusters=[
            Externalv1Cluster(id="cluster_id", spec=V1ClusterSpec(aws_v1=V1AWSDirectV1(bucket_name="my_bucket")))
        ],
    )

    client_mock.cloud_space_service_list_cloud_spaces.return_value = V1ListCloudSpacesResponse(
        cloudspaces=[V1CloudSpace(name="other_studio", id="cloud_space_id", cluster_id="cluster_id")],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    boto3 = mock.MagicMock()
    client_s3_mock = mock.MagicMock()
    client_s3_mock.list_objects_v2.return_value = {"KeyCount": 1, "Contents": []}
    boto3.client.return_value = client_s3_mock
    resolver.boto3 = boto3

    assert resolver._resolve_dir("something").url is None

    client_s3_mock.list_objects_v2.return_value = {"KeyCount": 0, "Contents": []}

    assert (
        resolver._resolve_dir("/teamspace/datasets/something/else").url
        == "s3://my_bucket/projects/project_id/datasets/something/else"
    )

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
@pytest.mark.parametrize("phase", ["LIGHTNINGAPP_INSTANCE_STATE_STOPPED", "LIGHTNINGAPP_INSTANCE_STATE_COMPLETED"])
def test_execute(phase, monkeypatch, lightning_sdk_mock):
    studio = mock.MagicMock()
    studio._studio.id = "studio_id"
    studio._teamspace.id = "teamspace_id"
    studio._studio.cluster_id = "cluster_id"
    studio._studio_api.get_machine.return_value = "cpu"
    studio.owner.name = "username"
    studio._teamspace.name = "teamspace_name"
    studio.name = "studio_name"
    studio.name = "studio_name"
    job = mock.MagicMock()
    job.name = "job_name"
    job.id = "job_id"
    job.status.phase = phase
    studio._studio_api.create_data_prep_machine_job.return_value = job
    studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance.return_value = job

    monkeypatch.setattr(resolver, "_LIGHTNING_SDK_AVAILABLE", True)
    lightning_sdk_mock.Studio = mock.MagicMock(return_value=studio)

    called = False

    def print_fn(msg, file=None):
        nonlocal called
        assert (
            msg
            == "Find your job at https://lightning.ai/username/teamspace_name/studios/studio_name/app?app_id=litdata&app_tab=Runs&job_name=job_name"
        )
        called = True

    original_print = __builtins__["print"]
    monkeypatch.setattr(sys, "argv", ["test.py", "--dummy"])
    monkeypatch.setitem(__builtins__, "print", print_fn)
    assert not called
    resolver._execute("dummy", 2)
    assert called
    monkeypatch.setitem(__builtins__, "print", original_print)

    generated_args = studio._studio_api.create_data_prep_machine_job._mock_call_args_list[0].args
    assert "&& python test.py --dummy" in generated_args[0]

    generated_kwargs = studio._studio_api.create_data_prep_machine_job._mock_call_args_list[0].kwargs
    assert generated_kwargs == {
        "name": "dummy",
        "num_instances": 2,
        "studio_id": "studio_id",
        "teamspace_id": "teamspace_id",
        "cloud_account": "cluster_id",
        "machine": "cpu",
        "interruptible": False,
    }

    generated_kwargs = (
        studio._studio_api._client.lightningapp_instance_service_get_lightningapp_instance._mock_call_args_list[
            0
        ].kwargs
    )
    assert generated_kwargs == {"project_id": "teamspace_id", "id": "job_id"}


def test_assert_dir_is_empty(monkeypatch):
    fs_provider = mock.MagicMock()
    fs_provider.is_empty = mock.MagicMock(return_value=False)
    monkeypatch.setattr(resolver, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    with pytest.raises(RuntimeError, match="The provided output_dir"):
        resolver._assert_dir_is_empty(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.is_empty = mock.MagicMock(return_value=True)

    resolver._assert_dir_is_empty(resolver.Dir(path="/teamspace/...", url="s3://"))


def test_assert_dir_has_index_file(monkeypatch):
    fs_provider = mock.MagicMock()
    fs_provider.is_empty = mock.MagicMock(return_value=False)
    monkeypatch.setattr(resolver, "_get_fs_provider", mock.MagicMock(return_value=fs_provider))

    with pytest.raises(RuntimeError, match="The provided output_dir"):
        resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.is_empty = mock.MagicMock(return_value=True)

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.exists = mock.MagicMock(return_value=False)

    fs_provider.is_empty = mock.MagicMock(return_value=True)

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"))

    fs_provider.exists = mock.MagicMock(return_value=True)

    fs_provider.is_empty = mock.MagicMock(return_value=False)
    fs_provider.delete_file_or_directory = mock.MagicMock()

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"), mode="overwrite")

    resolver._assert_dir_has_index_file(resolver.Dir(path="/teamspace/...", url="s3://"), mode="append")

    assert fs_provider.delete_file_or_directory.call_count == 1


def test_resolve_dir_absolute(tmp_path, monkeypatch):
    """Test that the directory gets resolved to an absolute path and symlinks are followed."""
    # relative path
    monkeypatch.chdir(tmp_path)
    relative = "relative"
    resolved_dir = resolver._resolve_dir(str(relative))
    assert resolved_dir.path == str(tmp_path / relative)
    assert Path(resolved_dir.path).is_absolute()
    monkeypatch.undo()

    # symlink
    src = tmp_path / "src"
    src.mkdir()
    link = tmp_path / "link"
    link.symlink_to(src)
    assert link.resolve() == src
    assert resolver._resolve_dir(str(link)).path == str(src)


def test_resolve_time_template():
    path_1 = "/logs/log_{%Y-%m}"
    path_2 = "/logs/my_logfile"
    path_3 = "/logs/log_{%Y-%m}/important"

    current_datetime = datetime.datetime.now()
    curr_year = current_datetime.year
    curr_month = current_datetime.month

    assert resolver._resolve_time_template(path_1) == f"/logs/log_{curr_year}-{curr_month:02d}"
    assert resolver._resolve_time_template(path_2) == path_2
    assert resolver._resolve_time_template(path_3) == f"/logs/log_{curr_year}-{curr_month:02d}/important"


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_gcs_connections(monkeypatch, lightning_cloud_mock):
    """Test GCS connections resolver."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    with pytest.raises(
        RuntimeError, match="`LIGHTNING_CLOUD_PROJECT_ID` couldn't be found from the environment variables."
    ):
        resolver._resolve_dir("/teamspace/gcs_connections/my_dataset")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[V1DataConnection(name="my_dataset", gcp=mock.MagicMock(source="gs://my-gcs-bucket"))],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    assert resolver._resolve_dir("/teamspace/gcs_connections/my_dataset").url == "gs://my-gcs-bucket"
    assert resolver._resolve_dir("/teamspace/gcs_connections/my_dataset/train").url == "gs://my-gcs-bucket/train"

    # Test missing data connection
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[],
    )

    with pytest.raises(ValueError, match="name `my_dataset`"):
        resolver._resolve_dir("/teamspace/gcs_connections/my_dataset")

    auth.clear()


@pytest.mark.skipif(sys.platform == "win32", reason="windows isn't supported")
def test_src_resolver_gcs_folders(monkeypatch, lightning_cloud_mock):
    """Test GCS folders resolver."""
    auth = login.Auth()
    auth.save(user_id="7c8455e3-7c5f-4697-8a6d-105971d6b9bd", api_key="e63fae57-2b50-498b-bc46-d6204cbf330e")

    monkeypatch.setenv("LIGHTNING_CLOUD_PROJECT_ID", "project_id")

    client_mock = mock.MagicMock()
    client_mock.data_connection_service_list_data_connections.return_value = V1ListDataConnectionsResponse(
        data_connections=[
            V1DataConnection(name="debug_folder", gcs_folder=mock.MagicMock(source="gs://my-gcs-bucket"))
        ],
    )

    client_cls_mock = mock.MagicMock()
    client_cls_mock.return_value = client_mock
    lightning_cloud_mock.rest_client.LightningClient = client_cls_mock

    expected = "gs://my-gcs-bucket"
    assert resolver._resolve_dir("/teamspace/gcs_folders/debug_folder").url == expected
    assert resolver._resolve_dir("/teamspace/gcs_folders/debug_folder/a/b/c").url == expected + "/a/b/c"

    auth.clear()
