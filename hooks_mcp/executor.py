import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict

from .config import Action, ParameterType
from .utils import process_terminal_output, resolve_path, validate_project_path


class ExecutionError(Exception):
    """Exception raised for errors during command execution."""

    pass


class CommandExecutor:
    """Handles secure execution of commands defined in HooksMCP configuration."""

    def __init__(self):
        self.project_root = Path(os.getcwd())

    def execute_action(
        self, action: Action, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an action with the given parameters.

        Args:
            action: The action to execute
            parameters: Parameters provided by the MCP client

        Returns:
            Dictionary containing stdout, stderr, and status code
        """
        # Validate and prepare parameters
        env_vars = self._prepare_parameters(action, parameters)

        # Build command args and substitute parameters in each arg
        command_args = self._build_command_args(action, env_vars)

        # Determine execution directory
        execution_dir = self.project_root
        if action.run_path:
            execution_dir = self.project_root / action.run_path
            # Validate run_path is within project boundaries
            if not validate_project_path(action.run_path, self.project_root):
                raise ExecutionError(
                    f"HooksMCP Error: Invalid run_path '{action.run_path}' for action '{action.name}'. "
                    f"Path must be within project boundaries and not contain directory traversal sequences."
                )

        # Execute command
        try:
            # Use subprocess.run with shell=False for security
            # Pass environment variables separately to avoid shell injection
            result = subprocess.run(
                command_args,
                cwd=execution_dir,
                env={**os.environ, **env_vars},
                capture_output=True,
                text=True,
                timeout=action.timeout,
                shell=False,
            )

            return {
                "stdout": process_terminal_output(result.stdout),
                "stderr": process_terminal_output(result.stderr),
                "status_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            raise ExecutionError(
                f"HooksMCP Error: Command for action '{action.name}' timed out after {action.timeout} seconds"
            )
        except Exception as e:
            raise ExecutionError(
                f"HooksMCP Error: Failed to execute command for action '{action.name}': {str(e)}"
            )

    def _build_command_args(self, action: Action, env_vars: Dict[str, str]) -> list:
        """
        Build command argv and substitute parameter variables with their values.

        If action.arguments is provided, argv is constructed as:
            [action.command] + action.arguments
        without shlex parsing.

        Otherwise, action.command is parsed with shlex for backwards compatibility.
        """
        if action.arguments is not None:
            return self._substitute_parameters_in_args(
                [action.command, *action.arguments], env_vars
            )

        try:
            command_args = shlex.split(action.command)
        except ValueError as e:
            raise ExecutionError(f"HooksMCP Error: Invalid command syntax: {e}")

        return self._substitute_parameters_in_args(command_args, env_vars)

    def _substitute_parameters_in_args(
        self, command_args: list[str], env_vars: Dict[str, str]
    ) -> list[str]:
        """
        Substitute parameter variables into pre-split argv elements.

        Args:
            command_args: Pre-split argv elements containing $VARIABLE_NAME placeholders
            env_vars: Dictionary of variable names to values

        Returns:
            List of command arguments with variables substituted
        """
        # Sort parameter names by length (descending) to handle collisions
        # This ensures $PREFIX_SUFFIX is processed before $PREFIX
        sorted_params = sorted(env_vars.keys(), key=len, reverse=True)

        # Substitute parameters in each argument
        substituted_args = []
        for arg in command_args:
            substituted_arg = arg
            for param_name in sorted_params:
                param_value = env_vars[param_name]
                # Convert parameter value to string for substitution
                # Note: quoting is not necessary since we're in the context of a single arg, not a string command
                param_str = str(param_value)
                # Replace all occurrences of $PARAM_NAME with the parameter value
                substituted_arg = substituted_arg.replace(f"${param_name}", param_str)
            substituted_args.append(substituted_arg)

        return substituted_args

    def _prepare_parameters(
        self, action: Action, provided_parameters: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Validate and prepare environment variables for command execution.

        Args:
            action: The action being executed
            provided_parameters: Parameters provided by the MCP client

        Returns:
            Dictionary of environment variables to inject
        """
        env_vars = {}

        # Process each parameter defined in the action
        for param in action.parameters:
            # Handle required and optional environment variables
            if param.type in [
                ParameterType.REQUIRED_ENV_VAR,
                ParameterType.OPTIONAL_ENV_VAR,
            ]:
                # These parameters are not provided by the client but should be in the environment
                env_value = os.environ.get(param.name)
                if env_value is not None:
                    env_vars[param.name] = env_value
                elif param.type == ParameterType.REQUIRED_ENV_VAR:
                    raise ExecutionError(
                        f"HooksMCP Error: Required environment variable '{param.name}' not set for action '{action.name}'"
                    )
                continue

            # Handle project file paths
            elif param.type == ParameterType.PROJECT_FILE_PATH:
                # Get the value from provided parameters or use default
                value = provided_parameters.get(param.name, param.default)

                # If no value and no default, it's required
                if value is None:
                    raise ExecutionError(
                        f"HooksMCP Error: Required parameter '{param.name}' not provided for action '{action.name}'"
                    )

                # Validate the path
                if not validate_project_path(value, self.project_root):
                    raise ExecutionError(
                        f"HooksMCP Error: Invalid path '{value}' for parameter '{param.name}' in action '{action.name}'. "
                        f"Path must be within project boundaries and not contain directory traversal sequences."
                    )

                # Check if file exists for project_file_path parameters
                full_path = resolve_path(value, self.project_root)
                if not full_path.exists():
                    raise ExecutionError(
                        f"HooksMCP Error: Path '{value}' for parameter '{param.name}' in action '{action.name}' does not exist"
                    )

                env_vars[param.name] = value

            # Handle insecure strings
            elif param.type == ParameterType.INSECURE_STRING:
                # Get the value from provided parameters or use default
                value = provided_parameters.get(param.name, param.default)

                # If no value and no default, it's required
                if value is None:
                    raise ExecutionError(
                        f"HooksMCP Error: Required parameter '{param.name}' not provided for action '{action.name}'"
                    )

                # Convert to string if needed
                env_vars[param.name] = str(value)

        return env_vars
