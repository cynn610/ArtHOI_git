#!/usr/bin/env python3
"""Submit a Hunyuan3D image-to-3D job and download its textured GLB."""

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


HOST = "ai3d.tencentcloudapi.com"
SERVICE = "ai3d"
VERSION = "2025-05-13"
AUTO_REGION = "auto"
DEFAULT_API_REGION = "ap-guangzhou"

TOKENHUB_SUBMIT_URL = "https://tokenhub.tencentmaas.com/v1/api/3d/submit"
TOKENHUB_QUERY_URL = "https://tokenhub.tencentmaas.com/v1/api/3d/query"
TOKENHUB_MODEL = "hy-3d-3.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ENV_FILE = PROJECT_ROOT / "conf" / "api_keys.env"
DEFAULT_ENV_FILE = LOCAL_ENV_FILE


def load_env_file(path):
    """Load simple KEY=VALUE entries without evaluating shell expressions."""
    path = Path(path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _usable_secret(value):
    if not value:
        return False
    upper = value.strip().upper()
    return not upper.startswith(("REPLACE_", "CHANGE_ME", "INVALID"))


def _json_request(url, payload, headers=None):
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TokenHub HTTP {error.code}: {body}") from error
    if isinstance(result, dict) and result.get("error"):
        error = result["error"]
        if isinstance(error, dict):
            message = error.get("message_zh") or error.get("message") or str(error)
        else:
            message = str(error)
        raise RuntimeError(f"TokenHub API error: {message}")
    return result


def _find_named_value(value, names):
    """Find a named value recursively in a JSON-like response."""
    def normalize(name):
        return re.sub(r"[^a-z0-9]", "", name.lower())

    wanted = {normalize(name) for name in names}
    if isinstance(value, dict):
        for key, item in value.items():
            if normalize(key) in wanted and item not in (None, ""):
                return item
        for item in value.values():
            found = _find_named_value(item, names)
            if found not in (None, ""):
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_named_value(item, names)
            if found not in (None, ""):
                return found
    return None


def _find_glb_url(value, context=""):
    """Locate a GLB/download URL across common asynchronous API layouts."""
    if isinstance(value, dict):
        output_type = str(
            value.get("type") or value.get("format") or value.get("file_type") or ""
        ).lower()
        if "glb" in output_type:
            direct = _find_named_value(
                value, ("url", "file_url", "download_url", "result_url")
            )
            if isinstance(direct, str) and direct.startswith(("http://", "https://")):
                return direct
        for key, item in value.items():
            key_lower = key.lower()
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                if "glb" in key_lower or ".glb" in item.lower():
                    return item
        for key, item in value.items():
            found = _find_glb_url(item, key.lower())
            if found:
                return found
        if context in {"data", "output", "outputs", "result", "results"}:
            direct = _find_named_value(
                value, ("url", "file_url", "download_url", "result_url")
            )
            if isinstance(direct, str) and direct.startswith(("http://", "https://")):
                return direct
    elif isinstance(value, list):
        for item in value:
            found = _find_glb_url(item, context)
            if found:
                return found
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        if ".glb" in value.lower() or context in {
            "output",
            "result",
            "url",
            "file_url",
            "download_url",
            "result_url",
        }:
            return value
    return None


class TencentMaaSTokenHubClient:
    """Bearer-token client for Tencent MaaS TokenHub asynchronous 3D jobs."""

    def __init__(
        self,
        api_key,
        submit_url=TOKENHUB_SUBMIT_URL,
        query_url=TOKENHUB_QUERY_URL,
    ):
        if not _usable_secret(api_key):
            raise RuntimeError(
                "TENCENTMAAS_API_KEY is missing or still contains a placeholder"
            )
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.submit_url = submit_url
        self.query_url = query_url

    def submit(self, payload):
        return _json_request(self.submit_url, payload, self.headers)

    def query(self, payload):
        return _json_request(self.query_url, payload, self.headers)


def _tokenhub_model_name(model):
    if not model or model in {"3.0", "3.1"}:
        if model == "3.0":
            return "hy-3d-3.0"
        return os.environ.get("TENCENTMAAS_MODEL", TOKENHUB_MODEL)
    return model


def submit_tokenhub_job(client, image_path, model=TOKENHUB_MODEL, prompt=None):
    image_path = Path(image_path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    encoding = os.environ.get("TENCENTMAAS_IMAGE_ENCODING", "data_url").lower()
    if encoding == "base64":
        image_value = encoded
    elif encoding == "data_url":
        image_value = f"data:{mime_type};base64,{encoded}"
    else:
        raise RuntimeError(
            "TENCENTMAAS_IMAGE_ENCODING must be 'data_url' or 'base64'"
        )
    image_field = os.environ.get("TENCENTMAAS_IMAGE_FIELD", "image_url")
    payload = {
        "model": _tokenhub_model_name(model),
        image_field: image_value,
    }
    prompt = prompt if prompt is not None else os.environ.get("TENCENTMAAS_PROMPT", "")
    if prompt:
        payload["prompt"] = prompt
    result = client.submit(payload)
    job_id = _find_named_value(result, ("id", "job_id", "task_id"))
    if job_id in (None, ""):
        raise RuntimeError(f"TokenHub submit response has no task id: {result}")
    return str(job_id)


def wait_for_tokenhub_result(
    client,
    job_id,
    output_path,
    model=TOKENHUB_MODEL,
    interval=30,
    sleep=time.sleep,
    download=None,
):
    download = download or _download
    query_payload = {"model": _tokenhub_model_name(model), "id": job_id}
    pending = {
        "wait",
        "waiting",
        "pending",
        "queued",
        "queue",
        "run",
        "running",
        "processing",
        "submitted",
        "created",
        "in_progress",
        "generating",
    }
    succeeded = {
        "done",
        "success",
        "succeed",
        "succeeded",
        "completed",
        "finished",
    }
    failed = {
        "fail",
        "failed",
        "failure",
        "error",
        "cancelled",
        "canceled",
    }
    while True:
        result = client.query(query_payload)
        result_url = _find_glb_url(result)
        status_value = _find_named_value(result, ("status", "state", "task_status"))
        status = str(status_value or "").strip().lower()
        if result_url and (not status or status in succeeded):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            download(result_url, output_path)
            return output_path
        if status in pending:
            sleep(interval)
            continue
        if status in failed:
            message = _find_named_value(
                result, ("message_zh", "message", "error_message", "error")
            )
            raise RuntimeError(f"TokenHub job {job_id} failed: {message or result}")
        if result_url:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            download(result_url, output_path)
            return output_path
        raise RuntimeError(
            f"Unrecognized TokenHub query response for job {job_id}: {result}"
        )


def _hmac_sha256(key, message):
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def resolve_region(region):
    if region in (None, "", AUTO_REGION):
        return DEFAULT_API_REGION
    return region


class TencentAi3DClient:
    def __init__(self, secret_id, secret_key, region=AUTO_REGION):
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = resolve_region(region)

    def call(self, action, payload):
        timestamp = int(time.time())
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        body = json.dumps(payload, separators=(",", ":"))
        hashed_body = hashlib.sha256(body.encode("utf-8")).hexdigest()
        headers = (
            "content-type:application/json; charset=utf-8\n"
            f"host:{HOST}\n"
            f"x-tc-action:{action.lower()}\n"
        )
        signed_headers = "content-type;host;x-tc-action"
        canonical_request = (
            f"POST\n/\n\n{headers}\n{signed_headers}\n{hashed_body}"
        )
        scope = f"{date}/{SERVICE}/tc3_request"
        string_to_sign = (
            "TC3-HMAC-SHA256\n"
            f"{timestamp}\n"
            f"{scope}\n"
            f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        )
        secret_date = _hmac_sha256(("TC3" + self.secret_key).encode("utf-8"), date)
        secret_service = _hmac_sha256(secret_date, SERVICE)
        secret_signing = _hmac_sha256(secret_service, "tc3_request")
        signature = hmac.new(
            secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        authorization = (
            f"TC3-HMAC-SHA256 Credential={self.secret_id}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        request = urllib.request.Request(
            f"https://{HOST}",
            data=body.encode("utf-8"),
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json; charset=utf-8",
                "Host": HOST,
                "X-TC-Action": action,
                "X-TC-Region": self.region,
                "X-TC-Timestamp": str(timestamp),
                "X-TC-Version": VERSION,
            },
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            result = json.load(response)["Response"]
        if "Error" in result:
            error = result["Error"]
            raise RuntimeError(f"{error['Code']}: {error['Message']}")
        return result


def submit_job(client, image_path, model="3.1"):
    image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    result = client.call(
        "SubmitHunyuanTo3DProJob",
        {
            "ImageBase64": image_base64,
            "Model": model,
            "GenerateType": "Normal",
            "EnablePBR": False,
            "FaceCount": 50000,
        },
    )
    return result["JobId"]


def _download(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def wait_for_result(client, job_id, output_path, interval=600, sleep=time.sleep, download=_download):
    while True:
        result = client.call("QueryHunyuanTo3DProJob", {"JobId": job_id})
        status = result["Status"]
        if status in ("WAIT", "RUN"):
            sleep(interval)
            continue
        if status == "FAIL":
            raise RuntimeError(
                f"{result.get('ErrorCode', 'FAIL')}: {result.get('ErrorMessage', '')}"
            )
        if status == "DONE":
            for file_3d in result.get("ResultFile3Ds", []):
                if file_3d.get("Type", "").upper() == "GLB":
                    output_path = Path(output_path)
                    download(file_3d["Url"], output_path)
                    return output_path
            raise RuntimeError("DONE response has no GLB result")
        raise RuntimeError(f"Unknown job status: {status}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--im", type=Path, help="single input image path")
    parser.add_argument(
        "--output-path", type=Path, required=True, help="output .glb path"
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="local KEY=VALUE API configuration (default: conf/api_keys.env)",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "tokenhub", "tencentcloud"),
        default="auto",
        help="prefer TokenHub when its Bearer key is configured",
    )
    parser.add_argument(
        "--query-submitted",
        metavar="JOB_ID",
        help="skip submission and query an already submitted Hunyuan3D Pro job",
    )
    parser.add_argument(
        "--region",
        default=AUTO_REGION,
        help=(
            "Tencent Cloud Region header. Use 'auto' to keep the recommended "
            "nearest-access endpoint and send the current product region."
        ),
    )
    parser.add_argument(
        "--model",
        default="3.1",
        help="3.1/hy-3d-3.1 for TokenHub, or 3.0/3.1 for Tencent Cloud",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="optional TokenHub text prompt used together with the reference image",
    )
    parser.add_argument("--poll-interval", type=int, default=None, metavar="SECONDS")
    args = parser.parse_args(argv)
    load_env_file(args.env_file)

    if args.query_submitted is None and args.im is None:
        parser.error("--im is required unless --query-submitted is used")

    tokenhub_key = os.environ.get("TENCENTMAAS_API_KEY", "")
    secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID", "")
    secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY", "")
    provider = args.provider
    if provider == "auto":
        if _usable_secret(tokenhub_key):
            provider = "tokenhub"
        elif _usable_secret(secret_id) and _usable_secret(secret_key):
            provider = "tencentcloud"
        else:
            raise RuntimeError(
                f"No usable Hunyuan credentials found in {args.env_file}. "
                "Fill TENCENTMAAS_API_KEY for TokenHub."
            )

    if provider == "tokenhub":
        client = TencentMaaSTokenHubClient(
            tokenhub_key,
            submit_url=os.environ.get("TENCENTMAAS_SUBMIT_URL", TOKENHUB_SUBMIT_URL),
            query_url=os.environ.get("TENCENTMAAS_QUERY_URL", TOKENHUB_QUERY_URL),
        )
        if args.query_submitted is None:
            job_id = submit_tokenhub_job(client, args.im, args.model, args.prompt)
            print(f"Submitted TokenHub job: {job_id}")
        else:
            job_id = args.query_submitted
            print(f"Querying TokenHub job: {job_id}")
        interval = args.poll_interval
        if interval is None:
            interval = int(os.environ.get("TENCENTMAAS_POLL_INTERVAL", "30"))
        output = wait_for_tokenhub_result(
            client,
            job_id,
            args.output_path,
            model=args.model,
            interval=interval,
        )
    else:
        if not (_usable_secret(secret_id) and _usable_secret(secret_key)):
            raise RuntimeError(
                "TENCENTCLOUD_SECRET_ID and TENCENTCLOUD_SECRET_KEY are required "
                "for --provider tencentcloud"
            )
        client = TencentAi3DClient(secret_id, secret_key, args.region)
        if args.query_submitted is None:
            job_id = submit_job(client, args.im, args.model)
            print(f"Submitted Tencent Cloud job: {job_id}")
        else:
            job_id = args.query_submitted
            print(f"Querying Tencent Cloud job: {job_id}")
        interval = args.poll_interval if args.poll_interval is not None else 600
        output = wait_for_result(client, job_id, args.output_path, interval)
    print(f"Downloaded GLB: {output}")


if __name__ == "__main__":
    main()
