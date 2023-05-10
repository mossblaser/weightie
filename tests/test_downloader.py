import pytest

from unittest.mock import Mock, call

from typing import Iterable, Any, BinaryIO, Callable

from pathlib import Path

from weightie.downloader import (
    Release,
    RepositoryNotFoundError,
    NoSuitableReleaseError,
    enumerate_github_releases,
    download_github_asset,
    parse_version,
    find_max_suitable_version,
    select_release,
    find_downloaded_files,
    print_status,
    download,
)


def test_enumerate_github_releases(uses_github_api: None) -> None:
    releases = list(enumerate_github_releases("mossblaser/weightie_test_repo"))
    assert len(releases) == 3

    releases_dict = {release.name: release for release in releases}
    print(releases_dict)
    print(releases)
    assert len(releases_dict) == len(releases)

    assert releases_dict["v0.1"] == Release(
        "v0.1", draft=False, prerelease=True, assets={}
    )
    assert releases_dict["v1.0"] == Release(
        "v1.0", draft=False, prerelease=False, assets={}
    )

    assert releases_dict["v2.0"].draft is False
    assert releases_dict["v2.0"].prerelease is False
    assert list(releases_dict["v2.0"].assets) == ["test_asset.txt"]


def test_download_github_asset(tmp_path: Path, uses_github_api: None) -> None:
    releases = list(enumerate_github_releases("mossblaser/weightie_test_repo"))
    releases_dict = {release.name: release for release in releases}
    url = releases_dict["v2.0"].assets["test_asset.txt"]

    filename = tmp_path / "test_asset.txt"
    with filename.open("wb") as f:
        download_github_asset(url, f)

    assert filename.read_text() == "Testing.\n"


@pytest.mark.parametrize(
    "version, exp",
    [
        # Simple versions
        ("v1", (1,)),
        ("v1.2.3", (1, 2, 3)),
        ("v10.20", (10, 20)),
        # Versions with extra bits
        ("v1.2beta", (1, 2)),
        # Malformed (mostly to check it doesn't crash
        ("", ()),
        ("v", ()),
        ("vfoo", ()),
        ("v1..2", (1, 2)),
    ],
)
def test_parse_version(version: str, exp: tuple[int, ...]) -> None:
    assert parse_version(version) == exp


@pytest.mark.parametrize(
    "target, versions, exp",
    [
        # No versions to choose from
        ("v1.0", [], None),
        # Exact match available
        ("v1.0", ["v1.0"], "v1.0"),
        ("v1.0", ["v1.0", "v1.2"], "v1.0"),
        ("v1.0", ["v1.2", "v1.0"], "v1.0"),
        # Fall back to latest available if no exact match
        ("v1.2", ["v1.0"], "v1.0"),
        ("v1.2", ["v1.0", "v1.1"], "v1.1"),
        ("v1.2", ["v1.1", "v1.0"], "v1.1"),
        # Don't use versions with different major version
        ("v1.0", ["v0.1"], None),
        # Dont use versions from the future
        ("v1.0", ["v1.1"], None),
        # Malformed versions present
        ("v1.0", ["v1.0", ""], "v1.0"),
        ("v1.0", ["", "v1.0"], "v1.0"),
        # Only malformed options
        ("v1.0", [""], None),
    ],
)
def test_find_max_suitable_version(
    target: str, versions: list[str], exp: str | None
) -> None:
    if exp is not None:
        assert (
            find_max_suitable_version(
                parse_version(target),
                versions,
                key=parse_version,
            )
            == exp
        )
    else:
        with pytest.raises(ValueError):
            find_max_suitable_version(
                parse_version(target),
                versions,
                key=parse_version,
            )


class TestSelectRelease:
    def test_required_filenames(self) -> None:
        assert select_release(
            releases=[
                Release("v1.0", False, False, {}),
                Release("v1.0", False, False, {"foo": "..."}),
                Release(
                    "v1.0", False, False, {"foo": "...", "bar": "...", "baz": "..."}
                ),
            ],
            target_version=(1, 0),
            required_asset_filenames=["foo", "bar"],
        ) == Release("v1.0", False, False, {"foo": "...", "bar": "...", "baz": "..."})

    def test_prereleases(self) -> None:
        assert select_release(
            releases=[
                Release("v1.0", False, False, {}),
                Release("v1.2", False, True, {}),
            ],
            target_version=(1, 5),
            required_asset_filenames=[],
            include_prereleases=True,
        ) == Release("v1.2", False, True, {})

        assert select_release(
            releases=[
                Release("v1.0", False, False, {}),
                Release("v1.2", False, True, {}),
            ],
            target_version=(1, 5),
            required_asset_filenames=[],
            include_prereleases=False,
        ) == Release("v1.0", False, False, {})

    def test_drafts(self) -> None:
        assert select_release(
            releases=[
                Release("v1.0", False, False, {}),
                Release("v1.2", True, False, {}),
            ],
            target_version=(1, 5),
            required_asset_filenames=[],
            include_drafts=True,
        ) == Release("v1.2", True, False, {})

        assert select_release(
            releases=[
                Release("v1.0", False, False, {}),
                Release("v1.2", True, False, {}),
            ],
            target_version=(1, 5),
            required_asset_filenames=[],
            include_drafts=False,
        ) == Release("v1.0", False, False, {})

    def test_no_results_because_no_input(self) -> None:
        with pytest.raises(NoSuitableReleaseError):
            select_release(
                releases=[],
                target_version=(1, 5),
                required_asset_filenames=[],
            )

    def test_no_results_because_no_suitable_version(self) -> None:
        with pytest.raises(NoSuitableReleaseError):
            select_release(
                releases=[
                    Release("v1.0", False, False, {}),
                    Release("v1.2", True, False, {}),
                ],
                target_version=(2, 0),
                required_asset_filenames=[],
            )

    def test_no_results_because_no_assets(self) -> None:
        with pytest.raises(NoSuitableReleaseError):
            select_release(
                releases=[
                    Release("v1.0", False, False, {}),
                    Release("v1.2", True, False, {}),
                ],
                target_version=(1, 0),
                required_asset_filenames=["foo"],
            )


class TestFindDownloadedFiles:
    def test_no_assets(self) -> None:
        assert find_downloaded_files([], (1, 0, 0), []) == {}

    def test_no_search_paths(self) -> None:
        assert find_downloaded_files(["foo"], (1, 0, 0), []) is None

    def test_no_matching_file_name(self, tmp_path: Path) -> None:
        for name in [
            "foo",  # No version
            "v1.0.0-bar",  # Wrong name
            "v1.0.0-foo.nope",  # Wrong extension
        ]:
            (tmp_path / name).touch()

        assert find_downloaded_files(["foo"], (1, 0, 0), [tmp_path]) is None

    def test_no_suitable_version(self, tmp_path: Path) -> None:
        for name in [
            "v0.0.1-foo",  # Too old
            "v1.5.0-foo",  # Too new
        ]:
            (tmp_path / name).touch()

        assert find_downloaded_files(["foo"], (1, 0, 0), [tmp_path]) is None

    def test_no_complete_set_with_same_version(self, tmp_path: Path) -> None:
        for name in [
            "v1.0.0-foo",
            "v1.0.1-bar",
            "v1.0.1-baz",
        ]:
            (tmp_path / name).touch()

        assert (
            find_downloaded_files(["foo", "bar", "baz"], (1, 1, 0), [tmp_path]) is None
        )

    def test_single_file_single_candidate(self, tmp_path: Path) -> None:
        for name in [
            "v1.0.0-foo",
            "v1.0.0-bar",  # Unrelated file
        ]:
            (tmp_path / name).touch()

        assert find_downloaded_files(["foo"], (1, 1, 0), [tmp_path]) == {
            "foo": tmp_path / "v1.0.0-foo",
        }

    def test_selects_newest_version(self, tmp_path: Path) -> None:
        for name in [
            "v1.0.0-foo",  # OK but old
            "v1.1.0-foo",
            "v1.5.0-foo",  # Too new
        ]:
            (tmp_path / name).touch()

        assert find_downloaded_files(["foo"], (1, 2, 0), [tmp_path]) == {
            "foo": tmp_path / "v1.1.0-foo",
        }

    def test_selects_highest_priority_copy(self, tmp_path: Path) -> None:
        for name in [
            "low/v1.0.0-foo",
            "high/v1.0.0-foo",
        ]:
            (tmp_path / name).parent.mkdir(exist_ok=True)
            (tmp_path / name).touch()

        assert find_downloaded_files(
            ["foo"],
            (1, 2, 0),
            [tmp_path / "high", tmp_path / "low"],
        ) == {
            "foo": tmp_path / "high/v1.0.0-foo",
        }

    def test_multiple_files_simple_case(self, tmp_path: Path) -> None:
        for name in [
            "v1.0.0-foo",
            "v1.0.0-bar",
        ]:
            (tmp_path / name).touch()

        assert find_downloaded_files(["foo", "bar"], (1, 2, 0), [tmp_path]) == {
            "foo": tmp_path / "v1.0.0-foo",
            "bar": tmp_path / "v1.0.0-bar",
        }

    def test_multiple_files_across_search_paths(self, tmp_path: Path) -> None:
        for name in [
            "a/v1.0.0-foo",
            "b/v1.0.0-bar",
        ]:
            (tmp_path / name).parent.mkdir(exist_ok=True)
            (tmp_path / name).touch()

        assert find_downloaded_files(
            ["foo", "bar"],
            (1, 2, 0),
            [tmp_path / "a", tmp_path / "b"],
        ) == {
            "foo": tmp_path / "a/v1.0.0-foo",
            "bar": tmp_path / "b/v1.0.0-bar",
        }


class TestPrintStatus:
    def test_single_file(self, capsys: Any) -> None:
        print_status(["foo"], "foo", 50 * 1024 * 1024, 100 * 1024 * 1024)
        print_status(["foo"], "foo", 100 * 1024 * 1024, 100 * 1024 * 1024)

        out, err = capsys.readouterr()

        assert err == (
            "Downloading foo:  50% ( 50.0 / 100.0 MB)\n"
            "Downloading foo: 100% (100.0 / 100.0 MB)\n"
        )

    def test_unknown_length(self, capsys: Any) -> None:
        print_status(["foo"], "foo", 50 * 1024 * 1024, None)
        print_status(["foo"], "foo", 100 * 1024 * 1024, None)

        out, err = capsys.readouterr()

        assert err == (
            "Downloading foo: 50.0 MB (unknown size)\n"
            "Downloading foo: 100.0 MB (unknown size)\n"
        )

    def test_multiple_files(self, capsys: Any) -> None:
        files = ["first", "second"] + (["_"] * 10)

        print_status(files, "first", 50 * 1024 * 1024, 100 * 1024 * 1024)
        print_status(files, "first", 100 * 1024 * 1024, 100 * 1024 * 1024)
        print_status(files, "second", 50 * 1024 * 1024, 100 * 1024 * 1024)
        print_status(files, "second", 100 * 1024 * 1024, 100 * 1024 * 1024)

        out, err = capsys.readouterr()

        assert err == (
            "Downloading first  ( 1 of 12):  50% ( 50.0 / 100.0 MB)\n"
            "Downloading first  ( 1 of 12): 100% (100.0 / 100.0 MB)\n"
            "Downloading second ( 2 of 12):  50% ( 50.0 / 100.0 MB)\n"
            "Downloading second ( 2 of 12): 100% (100.0 / 100.0 MB)\n"
        )


class TestDownloadWeights:
    @pytest.fixture(autouse=True)
    def mock_download(self, monkeypatch: Any) -> Mock:
        """
        This fixture mocks the download_github_asset function such that the
        progress callback is called three times (with 0%, 50% and 100% progress
        reports).
        """

        def mock_download_github_asset(
            url: str,
            file: BinaryIO,
            progress_callback: Callable[[int, int | None], None] | None = None,
            min_callback_interval: float = 0.5,
        ) -> None:
            if progress_callback is not None:
                progress_callback(0 * 1024 * 1024, 100 * 1024 * 1024)
                progress_callback(50 * 1024 * 1024, 100 * 1024 * 1024)
                progress_callback(100 * 1024 * 1024, 100 * 1024 * 1024)
            file.write(b"Downloaded from: " + url.encode("utf-8"))

        mock = Mock(side_effect=mock_download_github_asset)

        import weightie.downloader

        monkeypatch.setattr(weightie.downloader, "download_github_asset", mock)

        return mock

    @pytest.fixture(autouse=True)
    def mock_releases(self, monkeypatch: Any, mock_download: dict[str, int]) -> Mock:
        """
        This fixture mocks the enumerate_github_releases function so that it
        always returns the releases added to the returned list.

        Also implicitly enables the mock_download fixture.
        """
        mock = Mock(return_value=[])

        import weightie.downloader

        monkeypatch.setattr(weightie.downloader, "enumerate_github_releases", mock)

        return mock

    def test_no_assets_no_forced_update(self) -> None:
        # Special case: No assets requested. Should always succeed.
        assert download("foo/bar", [], "v1.0.0") == {}

    @pytest.mark.parametrize(
        "update, redownload",
        [
            (True, False),
            (True, True),
            (False, True),
        ],
    )
    def test_no_assets_forced_stil_fails_without_release(
        self,
        update: bool,
        redownload: bool,
    ) -> None:
        with pytest.raises(NoSuitableReleaseError):
            download(
                "foo/bar",
                [],
                "v1.0.0",
                update=update,
                redownload=redownload,
            )

    @pytest.mark.parametrize(
        "update, redownload",
        [
            (True, False),
            (True, True),
            (False, True),
        ],
    )
    def test_update_or_redownload_no_assets_but_ok_with_any_valid_release(
        self,
        mock_releases: Mock,
        mock_download: Mock,
        update: bool,
        redownload: bool,
    ) -> None:
        mock_releases.return_value = [
            Release(name="v1.0.0", draft=False, prerelease=False, assets={}),
        ]

        assert (
            download(
                "foo/bar",
                [],
                "v1.5.0",
                update=update,
                redownload=redownload,
            )
            == {}
        )

        # Nothing downloaded
        mock_download.assert_not_called()

    def test_use_available_files_by_default(
        self, tmp_path: Path, mock_releases: Mock
    ) -> None:
        for name in [
            "v1.0.0-foo",
            "v1.0.0-bar",
        ]:
            (tmp_path / name).touch()

        assert download(
            "foo/bar",
            ["foo", "bar"],
            "v1.5.0",
            search_paths=[tmp_path],
        ) == {
            "foo": tmp_path / "v1.0.0-foo",
            "bar": tmp_path / "v1.0.0-bar",
        }

        # Didn't check for updates since valid file available
        mock_releases.assert_not_called()

    def test_download_newer_version_on_update(
        self,
        tmp_path: Path,
        mock_releases: Mock,
        mock_download: Mock,
    ) -> None:
        mock_releases.return_value = [
            Release(
                name="v1.5.0",
                draft=False,
                prerelease=False,
                assets={
                    "foo": "http://example.com/v1.5.0/foo.bin",
                    "bar": "http://example.com/v1.5.0/bar.bin",
                },
            ),
        ]

        for name in [
            "v1.0.0-foo",
            "v1.0.0-bar",
        ]:
            (tmp_path / name).touch()

        assert download(
            "foo/bar",
            ["foo", "bar"],
            "v1.5.0",
            search_paths=[tmp_path],
            update=True,
        ) == {
            "foo": tmp_path / "v1.5.0-foo",
            "bar": tmp_path / "v1.5.0-bar",
        }

        # Should have obtained new files
        assert (tmp_path / "v1.5.0-foo").read_text() == (
            "Downloaded from: http://example.com/v1.5.0/foo.bin"
        )
        assert (tmp_path / "v1.5.0-bar").read_text() == (
            "Downloaded from: http://example.com/v1.5.0/bar.bin"
        )

    def test_dont_redownload_on_update_by_default(
        self,
        tmp_path: Path,
        mock_releases: Mock,
        mock_download: Mock,
    ) -> None:
        mock_releases.return_value = [
            Release(
                name="v1.0.0",
                draft=False,
                prerelease=False,
                assets={
                    "foo": "http://example.com/v1.0.0/foo.bin",
                    "bar": "http://example.com/v1.0.0/bar.bin",
                },
            ),
        ]

        for name in [
            "v1.0.0-foo",
            "v1.0.0-bar",
        ]:
            (tmp_path / name).touch()

        assert download(
            "foo/bar",
            ["foo", "bar"],
            "v1.5.0",
            search_paths=[tmp_path],
            update=True,
        ) == {
            "foo": tmp_path / "v1.0.0-foo",
            "bar": tmp_path / "v1.0.0-bar",
        }

        # Should have checked for updates but not redownloaded anything
        mock_releases.assert_called_once()
        mock_download.assert_not_called()

    def test_download_only_missing_files_by_default(
        self,
        tmp_path: Path,
        mock_releases: Mock,
        mock_download: Mock,
    ) -> None:
        mock_releases.return_value = [
            Release(
                name="v1.0.0",
                draft=False,
                prerelease=False,
                assets={
                    "foo": "http://example.com/v1.0.0/foo.bin",
                    "bar": "http://example.com/v1.0.0/bar.bin",
                    "baz": "http://example.com/v1.0.0/baz.bin",
                },
            ),
        ]

        for name in [
            "v1.0.0-foo",
            "v1.0.0-bar",
        ]:
            (tmp_path / name).touch()

        assert download(
            "foo/bar",
            ["foo", "bar", "baz"],
            "v1.5.0",
            search_paths=[tmp_path],
        ) == {
            "foo": tmp_path / "v1.0.0-foo",
            "bar": tmp_path / "v1.0.0-bar",
            "baz": tmp_path / "v1.0.0-baz",
        }

        # Check just 'baz' was downloaded
        mock_download.assert_called_once()
        assert (
            mock_download.mock_calls[0].args[0] == "http://example.com/v1.0.0/baz.bin"
        )

        # Check file really was downloaded
        assert (tmp_path / "v1.0.0-baz").read_text() == (
            "Downloaded from: http://example.com/v1.0.0/baz.bin"
        )

    def test_force_redownload(
        self,
        tmp_path: Path,
        mock_releases: Mock,
        mock_download: Mock,
    ) -> None:
        mock_releases.return_value = [
            Release(
                name="v1.0.0",
                draft=False,
                prerelease=False,
                assets={
                    "foo": "http://example.com/v1.0.0/foo.bin",
                },
            ),
        ]

        for name in [
            "v1.0.0-foo",
        ]:
            (tmp_path / name).touch()

        assert download(
            "foo/bar",
            ["foo"],
            "v1.5.0",
            search_paths=[tmp_path],
            redownload=True,
        ) == {
            "foo": tmp_path / "v1.0.0-foo",
        }

        # Check 'foo' was actually downloaded
        mock_download.assert_called_once()
        assert (
            mock_download.mock_calls[0].args[0] == "http://example.com/v1.0.0/foo.bin"
        )

        # Check file was overwritten
        assert (tmp_path / "v1.0.0-foo").read_text() == (
            "Downloaded from: http://example.com/v1.0.0/foo.bin"
        )

    def test_downloads_to_first_search_path_entry(
        self,
        tmp_path: Path,
        mock_releases: Mock,
        mock_download: Mock,
    ) -> None:
        mock_releases.return_value = [
            Release(
                name="v1.0.0",
                draft=False,
                prerelease=False,
                assets={
                    "foo": "http://example.com/v1.0.0/foo.bin",
                },
            ),
        ]

        for name in "abc":
            (tmp_path / name).mkdir()

        assert download(
            "foo/bar",
            ["foo"],
            "v1.5.0",
            search_paths=[tmp_path / letter for letter in "abc"],
        ) == {
            "foo": tmp_path / "a" / "v1.0.0-foo",
        }

        # Check 'foo' was actually downloaded
        mock_download.assert_called_once()
        assert (
            mock_download.mock_calls[0].args[0] == "http://example.com/v1.0.0/foo.bin"
        )

        # Check file was written
        assert (tmp_path / "a" / "v1.0.0-foo").read_text() == (
            "Downloaded from: http://example.com/v1.0.0/foo.bin"
        )

    def test_progress_callback(
        self,
        tmp_path: Path,
        mock_releases: Mock,
    ) -> None:
        mock_releases.return_value = [
            Release(
                name="v1.0.0",
                draft=False,
                prerelease=False,
                assets={
                    "foo": "http://example.com/v1.0.0/foo.bin",
                    "bar": "http://example.com/v1.0.0/bar.bin",
                },
            ),
        ]

        progress_callback = Mock()

        assert download(
            "foo/bar",
            ["foo", "bar"],
            "v1.5.0",
            search_paths=[tmp_path],
            progress_callback=progress_callback,
        ) == {
            "foo": tmp_path / "v1.0.0-foo",
            "bar": tmp_path / "v1.0.0-bar",
        }

        assert progress_callback.mock_calls == [
            call(["foo", "bar"], "foo", 0 * 1024 * 1024, 100 * 1024 * 1024),
            call(["foo", "bar"], "foo", 50 * 1024 * 1024, 100 * 1024 * 1024),
            call(["foo", "bar"], "foo", 100 * 1024 * 1024, 100 * 1024 * 1024),
            call(["foo", "bar"], "bar", 0 * 1024 * 1024, 100 * 1024 * 1024),
            call(["foo", "bar"], "bar", 50 * 1024 * 1024, 100 * 1024 * 1024),
            call(["foo", "bar"], "bar", 100 * 1024 * 1024, 100 * 1024 * 1024),
        ]
