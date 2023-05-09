import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-github", action="store_false", help="Skip tests which hit the GitHub API"
    )


@pytest.fixture(scope="session")
def uses_github_api(request):
    if not request.config.getoption("no_github"):
        pytest.skip()
