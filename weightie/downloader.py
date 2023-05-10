"""
Utility for downloading weights files uploaded as assets attached to GitHub
releases.

Quick usage example::

    >>> download(
    ...     repository="mossblaser/example",
    ...     asset_filenames=["foo.weights", "bar.weights"],
    ...     target_version="v1.2.3",
    ... )
    {
        "foo": Path("/path/to/downloaded/v1.1.0-foo.weights"),
        "bar": Path("/path/to/downloaded/v1.1.0-bar.weights"),
    }

See :py:func:`download` for full usage.


Why GitHub releases?
====================

GitHub releases include some pretty generous terms with regards to the size of
files you can include which according to
https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
on 25/04/2023 was:

    Each file included in a release must be under 2 GB. There is no limit on
    the total size of a release, nor bandwidth usage.

As such, this makes it a pretty good place to publish weights. This module
provides a collection of utilities for finding and downloading these release
files.
"""

from typing import Iterable, NamedTuple, BinaryIO, Callable, TypeVar

import os
import sys
import glob
import re

from functools import partial
from itertools import count, chain
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import requests
import platformdirs

import time


class Release(NamedTuple):
    """
    GitHub release information
    """

    name: str
    draft: bool
    prerelease: bool
    assets: dict[str, str]  # Filename -> URL


class RepositoryNotFoundError(ValueError):
    """
    Thrown when :py:class:`enumerate_github_releases` cannot find the named
    repository. This may also occur for private repositories when the
    GITHUB_TOKEN environment variable is not set to a suitable access token.
    """

    pass


def enumerate_github_releases(repository: str) -> Iterable[Release]:
    """
    Enumerate the GitHub releases published for a given repository.

    Parameters
    ==========
    repository : str
        A repository name, e.g. "mossblaser/weightie".

    Environment Variables
    =====================
    GITHUB_TOKEN
        If this environment variable is provided, the token will be passed to
        the GitHub API for authentication. This is required to probe private
        repositories.

    Yields
    ======
    Release
        Generates a series of Release objects describing a release and its
        assets.
    """
    headers = {
        "accept": "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
    }

    if github_token := os.environ.get("GITHUB_TOKEN"):
        headers["authorization"] = f"Bearer {github_token}"

    per_page = 100  # Maximum allowed by API

    for page in count(1):
        response = requests.get(
            f"https://api.github.com/repos/{repository}/releases?per_page={per_page}&page={page}",
            headers=headers,
        )
        if response.status_code == 404:
            raise RepositoryNotFoundError(repository)
        if response.status_code != 200:
            raise Exception(
                f"Got unexpected status {response.status_code} from GitHub API."
            )
        payload = response.json()

        for entry in payload:
            yield Release(
                name=entry["name"] or entry["tag_name"],
                draft=entry["draft"],
                prerelease=entry["prerelease"],
                assets={asset["name"]: asset["url"] for asset in entry["assets"]},
            )

        if len(payload) < per_page:
            break


def download_github_asset(
    url: str,
    file: BinaryIO,
    progress_callback: Callable[[int, int | None], None] | None = None,
    min_callback_interval: float = 0.5,
) -> None:
    """
    Download an asset from a GitHub release.

    Parameters
    ==========
    url : str
        The URL given in the Asset object.
    file : file
        A binary file opened for writing into which the asset will be
        downloaded.
    progress_callback : f(downloaded, total) or None
        A callback which will be called regularly during the download
        indicating progress. Gives the currently downloaded and total number of
        bytes. The total number of bytes may be None if no file size is
        indicated by the server.

        If None, no callbacks will be sent.
    min_callback_interval : float
        The minimum number of seconds to leave between calls to
        progress_callback.

    Environment Variables
    =====================
    GITHUB_TOKEN
        If this environment variable is provided, the token will be passed to
        the GitHub API for authentication. This is required to probe private
        repositories.
    """
    if progress_callback is None:
        progress_callback = lambda _downloaded, _total: None

    headers = {"accept": "application/octet-stream"}

    if github_token := os.environ.get("GITHUB_TOKEN"):
        headers["authorization"] = f"Bearer {github_token}"

    response = requests.get(url, headers=headers, stream=True)

    downloaded = 0
    total = None
    if "content-length" in response.headers:
        total = int(response.headers["content-length"])

    last_print = time.monotonic()
    progress_callback(downloaded, total)

    for data in response.iter_content(103 * 1024):
        file.write(data)
        downloaded += len(data)

        now = time.monotonic()
        if now - last_print >= min_callback_interval:
            progress_callback(downloaded, total)
            last_print = now

    progress_callback(downloaded, downloaded)


def parse_version(string: str) -> tuple[int, ...]:
    """
    Parse version numbers like "v1.2.3-beta" into a tuple (1, 2, 3) (where the
    non-numerical components are simply discarded).
    """
    string = string.lstrip("vV")

    match = re.match("[.0-9]*", string)
    assert match is not None
    numerical_part = match[0]

    return tuple(map(int, filter(None, numerical_part.split("."))))


T = TypeVar("T")


def find_max_suitable_version(
    target_version: tuple[int, ...],
    items: Iterable[T],
    key: Callable[[T], tuple[int, ...]],
) -> T:
    """
    Find the value in 'items' with the largest version number which is no
    greater than the target version and which has the same major version.

    Throws ValueError if no suitable versions were found.
    """
    return max(
        (key(item), -i, item)  # -i included to prevent sort ever using 'item'
        for i, item in enumerate(items)
        if (
            len(key(item)) >= 1
            and key(item)[0] == target_version[0]
            and key(item) <= target_version
        )
    )[2]


class NoSuitableReleaseError(ValueError):
    pass


def select_release(
    releases: Iterable[Release],
    target_version: tuple[int, ...],
    required_asset_filenames: list[str],
    include_prereleases: bool = False,
    include_drafts: bool = False,
) -> Release:
    """
    Selects a specific release based on various criteria.

    Parameters
    ==========
    releases : Iterable of Releases
    target_version : tuple e.g. (1, 2, 3)
        A simple version tuple

        Selects the release with the newest version which is:

        * No greater than this version number
        * Has the same major version number.
    required_asset_filenames : str
        Only consider relases which include assets with (at least) this set of
        filenames.
    include_prereleases : bool
        Consider prereleases (default False.)
    include_drafts : bool
        Consider drafts (default False.)
    """
    try:
        return find_max_suitable_version(
            target_version,
            (
                release
                for release in releases
                if (
                    (not release.prerelease or include_prereleases)
                    and (not release.draft or include_drafts)
                    and set(release.assets).issuperset(set(required_asset_filenames))
                )
            ),
            lambda r: parse_version(r.name),
        )
    except ValueError:
        raise NoSuitableReleaseError()


def find_downloaded_files(
    asset_filenames: list,
    target_version: tuple[int, ...],
    search_paths: list[Path],
) -> dict[str, Path] | None:
    """
    Find files with the given asset names compatible with the given version
    in the provided search paths.

    Parameters
    ==========
    asset_filenames : [filename, ...]
        The asset filenames to look for.
    target_version : (1, 2, 3)
        The target version number to find a file for.
    search_paths : [Path, ...]
        Directories to search in descending order of priority.

    Returns
    =======
    {asset_filename: Path} or None
        Gives the on-disk filenames to compatible versions of the named assets.
        If a complete set of files were not found, None is returned.

        Where multiple files are given, the version of all files returned will
        match exactly -- even if newer versions of some files may be available.
    """
    # Special (degenerate) case: If no asset filenames given, we can
    # successfully return no files!
    if not asset_filenames:
        return {}

    # Enumerate all currently available files, grouped by version, and where
    # several copies of an asset exist with the same version, preferring files
    # in higher-priority search paths.
    #
    # available_files[version][file] = Path
    available_files: dict[tuple[int, ...], dict[str, Path]] = defaultdict(dict)
    for search_path in reversed(search_paths):
        for file in asset_filenames:
            for candidate in search_path.glob(f"*{glob.escape(file)}"):
                if re.fullmatch(
                    r"[vV]([0-9]+\.)*[0-9]+-$", candidate.name[: -len(file)]
                ):
                    version = parse_version(candidate.name.partition("-")[0])
                    available_files[version][file] = candidate

    # Thin out the candidate versions to just those with the complete set of files
    available_files = {
        version: candidates
        for version, candidates in available_files.items()
        if set(candidates) == set(asset_filenames)
    }

    # Pick the most suitable version
    try:
        _selected_version, files = find_max_suitable_version(
            target_version,
            available_files.items(),
            lambda version_and_files: version_and_files[0],
        )
        return files
    except ValueError:
        return None


def print_status(
    files_to_download: list[str], file: str, downloaded: int, total: int | None
) -> None:
    """
    Default progress_callback for :py:func:`download`. Prints download
    progress on stderr.
    """
    max_filename_length = max(map(len, files_to_download))
    prefix = f"Downloading {file:{max_filename_length}s}"
    if len(files_to_download) > 1:
        prefix += (
            f" ({files_to_download.index(file) + 1:{len(str(len(files_to_download)))}d}"
        )
        prefix += f" of {len(files_to_download)})"

    if total is not None:
        formatted_total = f"{total / (1024 * 1024):0.1f}"
        print(
            "{}: {:3.0f}% ({:{}.1f} / {} MB)".format(
                prefix,
                (downloaded * 100) / total,
                downloaded / (1024 * 1024),
                len(formatted_total),
                formatted_total,
            ),
            file=sys.stderr,
        )
    else:
        print(
            f"{prefix}: {downloaded / (1024 * 1024):0.1f} MB (unknown size)",
            file=sys.stderr,
        )


def download(
    repository: str,
    asset_filenames: list[str],
    target_version: str,
    include_prereleases: bool = False,
    include_drafts: bool = False,
    search_paths: list[Path] | None = None,
    update: bool = False,
    redownload: bool = False,
    progress_callback: Callable[[list[str], str, int, int | None], None]
    | None = print_status,
    min_callback_interval: float = 0.5,
) -> dict[str, Path]:
    """
    Download a set of weights uploaded as a GitHub asset (if not already
    downloaded) and return the local filename.

    Parameters
    ==========
    repository : str
        Full GitHub reposotyr name, e.g. "mossblaser/weightie".

        For private repositories, you must set the GITHUB_TOKEN environment
        variable to a suitable GitHub access token.
    asset_filenames : [name, ...]
        List of asset file names to be downloaded.
    target_version : "v1.2.3"
        A string naming the target release version to download the weights for.
        Will fall back on earlier releases with the same major version until a
        suitable release is found.

        Unless update or redownload are given, this function will only check
        for newer releases if local files with compatible versions are not
        found.
    include_prereleases : bool
        If False, GitHub releases marked as prereleases will not be considered.
    include_drafts : bool
        If False, GitHub releases marked as drafts will not be considered.
    search_paths : [Path, ...] or None
        If not None, a list of directories to search for already downloaded
        files (in descending order of priority).

        If None, the system and user data directories (using the organisation and
        package in the repository arugment) are tried.

        When downloading a file, it will be downloaded to the first search path
        (i.e. the user data dir by default).
    update : bool
        If True, always check if a new version is available to download even if
        a local version is available..
    redownload : bool
        If True redownload files even if they have already been downloaded.
        Implicitly sets update=True.
    progress_callback : f(files_to_download, file, downloaded, total) or None
        A callback which will be called regularly during the download
        indicating progress. Gives the list of files due to be downloaded (in
        order), the currently downloading file and downloaded and total number
        of bytes for that file. The total number of bytes may be None if no
        file size is indicated by the server.

        Defaults to printing messages to stderr.

        If None, no callbacks will be sent.
    min_callback_interval : float
        The minimum number of seconds to leave between calls to
        progress_callback.

    Returns
    =======
    {asset_filename: Path, ...}
        A mapping from requested asset filename to local path where that file
        is stored on disk.

        Filenames have the format "<version>-<asset name>", e.g.
        "v1.0.0-foo.bin".

    Raises
    ======
    NoSuitableReleaseError
        If no release matching the requirements (e.g. compatible version
        number, prerelease/draft status or available assets) was found.
    PermissionError
        If none of the directories in the search path were writeable
    """
    parsed_target_version = parse_version(target_version)

    # Default search path to standard data directories for the application
    if search_paths is None:
        author, _, name = repository.partition("/")
        search_paths = [
            Path(d)
            for d in chain(
                [platformdirs.user_data_dir(name, author)],
                platformdirs.site_data_dir(name, author, multipath=True).split(
                    os.pathsep
                ),
            )
        ]

    if progress_callback is None:
        progress_callback = lambda _files_to_download, _file, _downloaded, _total: None

    existing_files = find_downloaded_files(
        asset_filenames, parsed_target_version, search_paths
    )

    # Check for (and download) updates or missing files.
    if update or redownload or existing_files is None:
        release = select_release(
            enumerate_github_releases(repository),
            parsed_target_version,
            asset_filenames,
            include_prereleases,
            include_drafts,
        )
        version = parse_version(release.name)

        # [(file, versioned_filename, url), ...]
        version_prefix = f"v{'.'.join(map(str, version))}-"
        downloads = [
            (file, version_prefix + file, release.assets[file])
            for file in asset_filenames
        ]

        if not redownload:
            # Filter download list to include only missing files
            downloads = [
                (file, versioned_filename, url)
                for file, versioned_filename, url in downloads
                if not any(
                    (search_path / versioned_filename).is_file()
                    for search_path in search_paths
                )
            ]

        # Download files
        files_to_download = [file for file, _versioned_filename, _url in downloads]
        download_dir = search_paths[0]
        for file, versioned_filename, url in downloads:
            download_dir.mkdir(parents=True, exist_ok=True)
            local_filename = download_dir / versioned_filename
            # NB: Atomically replace the local file with the downloaded
            # data, also downloading to a unique filename to ensure
            # that if multiple processes are trying to download the
            # file simultaneously they won't clash.
            download_filename = local_filename.with_stem(f".{uuid4()}")
            try:
                with download_filename.open("wb") as f:
                    download_github_asset(
                        url,
                        f,
                        partial(progress_callback, files_to_download, file),
                        min_callback_interval,
                    )
                download_filename.rename(local_filename)
            finally:
                download_filename.unlink(missing_ok=True)

        files = find_downloaded_files(
            asset_filenames, parsed_target_version, search_paths
        )
        assert files is not None
        return files
    else:
        return existing_files
