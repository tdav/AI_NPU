"""HuggingFace: разбор URL, скачивание нужного варианта квантизации, публикация в бакет."""
import re

from huggingface_hub import HfApi, hf_hub_download, snapshot_download


def repo_id_from_url(url):
    """https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX[/tree/main] | onnx-community/... -> repo id."""
    m = re.match(r"(?:https?://(?:huggingface\.co|hf-mirror\.com)/)?([\w.-]+/[\w.-]+)", url.strip())
    if not m:
        raise SystemExit(f"Не разобрать адрес модели: {url}")
    return m.group(1)


def fetch_config(repo_id, local_dir):
    return hf_hub_download(repo_id, "config.json", local_dir=local_dir)


def download(repo_id, patterns, local_dir):
    return snapshot_download(repo_id, allow_patterns=patterns, local_dir=local_dir)


def publish(package_dir, bucket_name, private=False):
    """Создаёт бакет <пользователь>/<bucket_name> (если нет) и синхронизирует в него каталог пакета."""
    api = HfApi()
    user = api.whoami()["name"]
    bucket = bucket_name if "/" in bucket_name else f"{user}/{bucket_name}"
    api.create_bucket(bucket, private=private, exist_ok=True)
    api.sync_bucket(package_dir, f"hf://buckets/{bucket}", quiet=True)
    return f"https://huggingface.co/buckets/{bucket}"
