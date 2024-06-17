from minio import Minio
import os
import glob
import fire

def get_minio_client(endpoint="bjsw-storage01.streamcomputing.com:9000", access_key="svc-ci", secret_key="Rdxt76*tx"):
    #"bjsw-storage01.streamcomputing.com:9000",
    # "172.16.11.18:9000",
    return Minio(endpoint,access_key, secret_key, secure=False)


def get_file(local_file, remote_file=None):
    if not os.path.exists(local_file):
        # print("Downloading remote file '{}' to local file '{}'".format(remote_file, local_file))
        if "/" in local_file:
            local_dir = local_file.rsplit("/", 1)[0]
            os.makedirs(local_dir, exist_ok=True)
        client = get_minio_client()
        client.fget_object("solution", remote_file, local_file)
    else:
        # print("A local file '{}' already exists!".format(local_file))
        return 


def get_files(local_dir, remote_dir):
    remote_files = []
    try:
        client = get_minio_client()
        res = client.list_objects("solution", prefix=remote_dir, recursive=True)
        for i in res:
            remote_files.append(i.object_name)
    except:
        remote_files = []
    local_files = []
    if remote_files:
        file_num = len(remote_files)
        for i in range(file_num):
            remote_file = remote_files[i]
            file_name = remote_file.split("/")[-1]
            local_file = os.path.join(local_dir, file_name)
            print("Progress {}/{}".format(i + 1, file_num))
            get_file(local_file, remote_file)
            local_files.append(local_file)
    else:
        local_dir = os.path.join(local_dir, remote_dir)
        local_files = os.listdir(local_dir)
        local_files = [os.path.join(local_dir, f) for f in local_files]
    return local_files


def upload_file(local_file, remote_file=None):
    print(local_file)
    if os.path.exists(local_file):
        print("Uploading  local file '{}' to remote file '{}'".format(local_file, remote_file))
        # client = get_minio_client("172.16.11.18:9000")
        client = get_minio_client()
        client.fput_object("solution", remote_file, local_file)
    else:
        print("local file is none")


def upload_files(local_dir, remote_dir):
    rel_paths = glob.glob(local_dir + '/**', recursive=True)
    for local_file in rel_paths:
        remote_path = f'{remote_dir}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            upload_file(local_file, remote_path)


if __name__ == '__main__':
    fire.Fire()