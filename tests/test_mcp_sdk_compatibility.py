"""Guards against resolving an mcp SDK this server cannot run against.

`hooks_mcp.server` is built on the low-level `Server` decorators and imports
`mcp.server.streamable_http_manager` at module scope. Both ends of the SDK range
break that: 2.x removed the decorators, and the streamable HTTP manager does not
exist before 1.8.0. An unconstrained `mcp` dependency therefore makes
`uvx hooks-mcp` fail at import, which is what these tests exist to catch.

The rest of the suite replaces `Server` with a `MagicMock`, so this module is the
only place the real SDK is exercised end to end.
"""

import asyncio
from importlib.metadata import requires, version
from pathlib import Path

import pytest
from mcp.server import Server
from packaging.requirements import Requirement
from packaging.version import Version

from hooks_mcp.config import Action, HooksMCPConfig
from hooks_mcp.config import Prompt as ConfigPrompt
from hooks_mcp.executor import CommandExecutor
from hooks_mcp.server import _create_server

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 predates tomllib
    import tomli as tomllib  # type: ignore[no-redef]

PYPROJECT_PATH = Path(__file__).parent.parent / "pyproject.toml"

# First release shipping `mcp.server.streamable_http_manager`.
MINIMUM_SUPPORTED_MCP = Version("1.8.0")
# Last release without it, used to prove the declared floor holds.
LAST_UNSUPPORTED_OLD_MCP = Version("1.7.1")
# First release without the low-level `Server` decorators.
FIRST_UNSUPPORTED_MCP = Version("2.0.0")

# Decorators `hooks_mcp.server._create_server` registers handlers with.
REQUIRED_SERVER_DECORATORS = (
    "list_tools",
    "call_tool",
    "list_prompts",
    "get_prompt",
)


def _only_mcp_requirement(declarations: list[str]) -> Requirement:
    requirements = [
        requirement
        for requirement in map(Requirement, declarations)
        if requirement.name == "mcp"
    ]
    assert len(requirements) == 1, f"Expected one mcp requirement, got {requirements}"
    return requirements[0]


def mcp_requirement_in_pyproject() -> Requirement:
    """The `mcp` dependency as declared in pyproject.toml — the source of truth."""
    with PYPROJECT_PATH.open("rb") as f:
        pyproject = tomllib.load(f)

    return _only_mcp_requirement(pyproject["project"]["dependencies"])


def mcp_requirement_in_metadata() -> Requirement:
    """The `mcp` dependency as built into this package's `Requires-Dist`.

    This is what an installer resolves against, so it is what a PyPI user gets.
    It can lag pyproject.toml until the package is rebuilt, which is why both
    sources are checked.
    """
    return _only_mcp_requirement(list(requires("hooks-mcp") or []))


REQUIREMENT_SOURCES = {
    "pyproject.toml": mcp_requirement_in_pyproject,
    "package metadata": mcp_requirement_in_metadata,
}


def echo_config() -> HooksMCPConfig:
    """A minimal config whose action runs without touching the project."""
    return HooksMCPConfig(
        server_name="SdkCompatibilityServer",
        server_description="Server used to exercise the real mcp SDK",
        actions=[
            Action(
                name="echo_hello",
                description="Echo a fixed string",
                command="echo hello_from_hooks_mcp",
            )
        ],
        prompts=[
            ConfigPrompt(
                name="greeting",
                description="A fixed greeting",
                prompt_text="hello from a prompt",
            )
        ],
    )


class TestMcpSdkVersion:
    """Test that the mcp SDK in use is one this server supports."""

    def test_installed_mcp_is_supported(self):
        """The installed SDK must fall inside the supported range."""
        installed = Version(version("mcp"))

        assert MINIMUM_SUPPORTED_MCP <= installed < FIRST_UNSUPPORTED_MCP, (
            f"mcp {installed} is installed, but hooks-mcp supports "
            f">={MINIMUM_SUPPORTED_MCP},<{FIRST_UNSUPPORTED_MCP}. "
            "Check the `mcp` constraint in pyproject.toml and uv.lock."
        )

    @pytest.mark.parametrize("decorator_name", REQUIRED_SERVER_DECORATORS)
    def test_server_exposes_required_decorators(self, decorator_name):
        """The low-level Server must still expose the decorators we register with."""
        assert hasattr(Server, decorator_name), (
            f"mcp {version('mcp')} has no Server.{decorator_name}; "
            "hooks_mcp.server cannot register its handlers against this SDK."
        )

    @pytest.mark.parametrize("source_name", REQUIREMENT_SOURCES)
    def test_declared_requirement_excludes_unsupported_versions(self, source_name):
        """The declared dependency must not allow a version that fails at import."""
        requirement = REQUIREMENT_SOURCES[source_name]()
        specifier = requirement.specifier

        assert not specifier.contains(str(FIRST_UNSUPPORTED_MCP), prereleases=True), (
            f"mcp requirement `{requirement}` in {source_name} allows "
            f"{FIRST_UNSUPPORTED_MCP}, which removed the low-level Server decorators."
        )
        assert not specifier.contains(str(LAST_UNSUPPORTED_OLD_MCP)), (
            f"mcp requirement `{requirement}` in {source_name} allows releases below "
            f"{MINIMUM_SUPPORTED_MCP}, which have no streamable_http_manager module."
        )
        # Proves the range admits a real release without forbidding a hard pin.
        assert specifier.contains(version("mcp")), (
            f"mcp requirement `{requirement}` in {source_name} does not admit the "
            f"installed mcp {version('mcp')}; the declared range and the resolved "
            "environment disagree, so one of the two is wrong."
        )


class TestRealSdkSession:
    """Drive a real MCP session against the real SDK, with no mocks."""

    def test_handshake_lists_and_calls_configured_tools(self, tmp_path):
        """A full initialize/list/call round trip must work against the real SDK.

        This is the check the mocked tests in test_server.py cannot make: it runs
        the real `Server` over a real client session, so an SDK that no longer
        supports our handler registration fails here instead of at a user's
        `uvx hooks-mcp`.
        """
        # `mcp.shared.memory` is an SDK-internal test utility and was removed in
        # 2.x. Importing it here keeps a missing helper from failing collection,
        # which would suppress the version guards above.
        from mcp.shared.memory import create_connected_server_and_client_session

        config = echo_config()
        server = _create_server(config, tmp_path / "hooks_mcp.yaml", CommandExecutor())

        async def run_session():
            async with create_connected_server_and_client_session(
                server, raise_exceptions=True
            ) as client:
                # The helper runs the server and performs the initialize handshake.
                tools = await client.list_tools()
                prompts = await client.list_prompts()
                call_result = await client.call_tool("echo_hello", {})
                prompt_result = await client.get_prompt("greeting", {})
                return tools, prompts, call_result, prompt_result

        tools, prompts, call_result, prompt_result = asyncio.run(run_session())

        assert "echo_hello" in {tool.name for tool in tools.tools}
        assert "greeting" in {prompt.name for prompt in prompts.prompts}

        assert call_result.isError is False
        call_output = "\n".join(
            block.text for block in call_result.content if block.type == "text"
        )
        assert "hello_from_hooks_mcp" in call_output
        assert "Exit code: 0" in call_output

        prompt_message = prompt_result.messages[0].content
        assert prompt_message.type == "text"
        assert prompt_message.text == "hello from a prompt"
