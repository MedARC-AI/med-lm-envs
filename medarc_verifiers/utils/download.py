import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from requests.adapters import HTTPAdapter
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util.retry import Retry


def download_file(
    url: str,
    dest: str | Path,
    connect_timeout: int = 5,
    read_timeout: int = 30,
    retries: int = 5,
    verify: bool | str = True,
) -> Path:
    """Download URL to 'dest' with retries/backoff, no overwrite, atomic write."""
    dest = Path(dest)
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    retry = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods={"GET"},
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    tmp_path: Path | None = None

    verify_param: bool | str = verify

    if verify_param is False:
        requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)  # type: ignore[attr-defined]

    try:
        with requests.Session() as session:
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            with session.get(
                url,
                stream=True,
                timeout=(connect_timeout, read_timeout),
                verify=verify_param,
            ) as response:
                response.raise_for_status()
                with NamedTemporaryFile(
                    dir=dest.parent, prefix=f"{dest.name}.", suffix=".part", delete=False
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)
                    for chunk in response.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            tmp_file.write(chunk)
        if tmp_path is None:
            msg = "Temporary file was not created during download."
            raise RuntimeError(msg)

        tmp_path.replace(dest)
        return dest
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def medarc_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        env_override = os.getenv("MEDARC_CACHE_DIR")
        if env_override:
            return Path(env_override)
        return Path.home() / ".cache" / "medarc"
    return Path(cache_dir)
