#coding=utf8
"""Action Module - action definitions and parsers for the data analysis agent.

This module defines the executable action interface used by the agent:
- Represents Bash, Python, SQL, file, and termination actions
- Parses model responses into structured action objects
- Provides action-space descriptions for prompt construction

References:
- https://github.com/yiyihum/da-code/tree/main/da_agent/agent/action.py
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Any, Union, List, Dict
from abc import ABC

def _extract_balanced_arg(text: str, start: int) -> Optional[str]:
    """Extract content from start position until the matching closing paren,
    respecting quoted strings (single, double, backtick) and their escapes.
    Returns the extracted substring or None if unbalanced."""
    i = start
    n = len(text)
    depth = 1  # we're already inside one paren (the opening `(`)
    while i < n and depth > 0:
        ch = text[i]
        if ch in ('"', "'", '`'):
            quote = ch
            i += 1  # skip opening quote
            while i < n:
                if text[i] == '\\' and i + 1 < n:
                    i += 2  # skip escaped char
                elif text[i] == quote:
                    i += 1  # skip closing quote
                    break
                else:
                    i += 1
        elif ch == '(':
            depth += 1
            i += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return text[start:i]
            i += 1
        else:
            i += 1
    return None


def _split_args(text: str) -> list:
    """Split comma-separated arguments, respecting quoted strings."""
    parts = []
    current = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ('"', "'", '`'):
            quote = ch
            current.append(ch)
            i += 1
            while i < n:
                if text[i] == '\\' and i + 1 < n:
                    current.append(text[i])
                    current.append(text[i + 1])
                    i += 2
                elif text[i] == quote:
                    current.append(text[i])
                    i += 1
                    break
                else:
                    current.append(text[i])
                    i += 1
        elif ch == ',':
            parts.append(''.join(current))
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    if current:
        parts.append(''.join(current))
    return parts


def remove_quote(text: str) -> str:
    """ 
    If the text is wrapped by a pair of quote symbols, remove them.
    In the middle of the text, the same quote symbol should remove the '/' escape character.
    """
    for quote in ['"', "'", "`"]:
        if text.startswith(quote) and text.endswith(quote):
            text = text[1:-1]
            text = text.replace(f"\\{quote}", quote)
            break
    return text.strip()


@dataclass
class Action(ABC):
    
    action_type: str = field(
        repr=False,
        metadata={"help": 'type of action, e.g. "exec_code", "create_file", "terminate"'}
    )


    @classmethod
    def get_action_description(cls) -> str:
        return """
Action: action format
Description: detailed definition of this action type.
Usage: example cases
Observation: the observation space of this action type.
"""

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Any]:
        raise NotImplementedError

@dataclass
class Bash(Action):

    action_type: str = field(
        default="exec_code",
        init=False,
        repr=False,
        metadata={"help": 'type of action, c.f., "exec_code"'}
    )

    code: str = field(
        metadata={"help": 'command to execute'}
    )

    @classmethod
    def get_action_description(cls) -> str:
        return """
## Bash Action
* Signature: Bash(code="shell_command")
* Description: This action string will execute a valid shell command in the `code` field. Only non-interactive commands are supported. Commands like "vim" and viewing images directly (e.g., using "display") are not allowed.
* Example: Bash(code="ls -l")
"""

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        # Find Bash(code=...), handling quoted strings with nested parens
        m = re.search(r'Bash\(code=', text)
        if not m:
            return None
        start = m.end()
        code = _extract_balanced_arg(text, start)
        if code is not None:
            return cls(code=remove_quote(code))
        return None
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(code="{self.code}")'

@dataclass
class Python(Action):

    action_type: str = field(
        default="python_snippet",
        init=False,
        repr=False,
        metadata={"help": 'type of action, c.f., "python_snippet"'}
    )

    code: str = field(
        metadata={"help": 'executable python commands or snippets'}
    )

    filepath: Optional[str] = field(
        default=None,
        metadata={"help": 'name of file to create'}
    )

    @classmethod
    def get_action_description(cls) -> str:
        return """
## Python Action
* Signature: Python(file_path="path/to/python_file"):
```python
executable_python_code
```
* Description: This action will create a python file in the field `file_path` with the content wrapped by paired ``` symbols. If the file already exists, it will be overwritten. After creating the file, the python file will be executed. 
* Example: Python(file_path="./hello_world.py"):
```python
print("Hello, world!")
```
"""

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        pattern=[r'Python\(file_path=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Python\(file_path=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',
                 r'Python\(filepath=(.*?)\).*?```python[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```',r'Python\(filepath=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```']
        for p in pattern:
            matches = re.findall(p, text, flags=re.DOTALL)
            if matches:
                filepath = matches[-1][0].strip()
                code = matches[-1][2].strip()
                return cls(code=code, filepath=remove_quote(filepath))
        return None
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(file_path="{self.filepath}"):\n```python\n{self.code}\n```'

@dataclass
class SQL(Action):

    action_type: str = field(
        default="sql_command",
        init=False,
        repr=False,
        metadata={"help": 'type of action, c.f., "sql_command"'}
    )

    code: str = field(
        metadata={"help": 'SQL command to execute'}
    )

    file_path: str = field(
        default=None,
        metadata={"help": 'path to the database file'}
    )

    output: str = field(
        default=None,
        metadata={"help": 'output file path or "direct"'}
    )

    @classmethod
    def get_action_description(cls) -> str:
        return """
## SQL Action
* Signature: SQL(file_path="path/to/database_file", command="sql_command", output="path/to/output_file.csv" or "direct")
* Description: Executes an SQL command on the specified database file. If `output` is set to a file path, the results are saved to this CSV file; if set to 'direct', results are displayed directly.
* Constraints:
  - The database file must be accessible and in a format compatible with SQLite (e.g., .sqlite, .db).
  - SQL commands must be valid and safely formatted to prevent security issues such as SQL injection.
* Examples:
  - Example1: SQL(file_path="data.sqlite", command="SELECT name FROM sqlite_master WHERE type='table'", output="directly")
  - Example2: SQL(file_path="data.db", command="SELECT * FROM users", output="users_output.csv")
"""

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        m = re.search(r'SQL\(', text)
        if not m:
            return None
        # Extract the full balanced content inside SQL(...)
        content = _extract_balanced_arg(text, m.end())
        if content is None:
            return None
        # Split by commas, respecting quoted strings
        parts = _split_args(content)
        if len(parts) >= 3:
            file_path = parts[0].strip()
            command = ','.join(parts[1:-1]).strip()  # command may contain commas
            output = parts[-1].strip()
            # Remove parameter names if present
            for prefix in ('file_path=', 'filepath='):
                if file_path.startswith(prefix):
                    file_path = file_path[len(prefix):]
            for prefix in ('command=', 'code='):
                if command.startswith(prefix):
                    command = command[len(prefix):]
            for prefix in ('output=',):
                if output.startswith(prefix):
                    output = output[len(prefix):]
            return cls(file_path=remove_quote(file_path), code=remove_quote(command), output=remove_quote(output))
        return None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(file_path="{self.file_path}", command="{self.code}", output="{self.output}")'


@dataclass
class Terminate(Action):

    action_type: str = field(
        default="terminate",
        init=False,
        repr=False,
        metadata={"help": "terminate action representing the task is finished, or you think it is impossible for you to complete the task"}
    )

    output: Optional[str] = field(
        default=None,
        metadata={"help": "answer to the task or output file path or 'FAIL', if exists"}
    )

    code : str = field(
        default=''
    )

    @classmethod
    def get_action_description(cls) -> str:
        return """
## Terminate Action
* Signature: Terminate(output="literal_answer_or_output_path")
* Description: This action denotes the completion of the entire task and returns the final answer or the output file/folder path. Make sure the output file is located in the initial workspace directory.
* Output rule: Put only the shortest exact final answer in `output` (number, boolean, string, or JSON-style list as requested). Do not include explanation, markdown, derivation, file paths, or extra text unless the task explicitly asks for them.
* Examples:
  - Example1: Terminate(output="New York")
  - Example2: Terminate(output="result.csv")
  - Example3: Terminate(output="FAIL")
"""

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(output="{self.output}")'

    @classmethod
    def parse_action_from_text(cls, text: str) -> Optional[Action]:
        m = re.search(r'Terminate\(output=', text)
        if not m:
            return None
        content = _extract_balanced_arg(text, m.end())
        if content is not None:
            return cls(output=remove_quote(content))
        return None
    


# @dataclass
# class ExecutePythonSnippet(Action):

#     action_type: str = field(
#         default="python_snippet",
#         init=False,
#         repr=False,
#         metadata={"help": 'type of action, c.f., "python_snippet"'}
#     )

#     code: str = field(
#         metadata={"help": 'executable python commands or snippets'}
#     )

#     def __repr__(self) -> str:
#         return f"ExecutePythonSnippet:\n```\n{self.code.strip()}\n```"

#     @classmethod
#     def get_action_description(cls) -> str:
#         return """
# ## ExecutePythonSnippet
# Signature: `ExecutePythonSnippet`
# ```
# executable_python_code
# ```

# Description: This action executes a very simple Python snippet command or snippet wrapped in paired ``` symbols. It is intended for short, exploratory code snippets. For longer or more complex code, please use CreateFile to write the code to a file and then execute it.

# Example: ExecutePythonSnippet:
# ```
# print("Hello, world!")
# ```
# """

#     @classmethod
#     def parse_action_from_text(cls, text: str) -> Optional[Action]:
#         matches = re.findall(r'ExecutePythonSnippet.*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```', text, flags=re.DOTALL)
#         if matches:
#             code = matches[-1][1]
#             return cls(code=code.strip())
#         return None
    
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}\n '''\n{self.code}\n'''"


# @dataclass
# class CreateFile(Action):

#     action_type: str = field(
#         default="create_file",
#         init=False,
#         repr=False,
#         metadata={"help": 'type of action, c.f., "create_file"'}
#     )

#     code: str = field(
#         metadata={"help": 'code to write into file'}
#     )

#     filepath: Optional[str] = field(
#         default=None,
#         metadata={"help": 'name of file to create'}
#     )

#     def __repr__(self) -> str:
#         return f"CreateFile(filepath=\"{self.filepath}\"):\n```\n{self.code.strip()}\n```"

#     @classmethod
#     def get_action_description(cls) -> str:
#         return """
# ## CreateFile
# Signature: CreateFile(filepath="path/to/file"):
# ```
# file_content
# ```
# Description: This action will create a file in the field `filepath` with the content wrapped by paired ``` symbols. Make sure the file content is complete and correct. If the file already exists, you can only use EditFile to modify it.
# Example: CreateFile(filepath="hello_world.py"):
# ```
# print("Hello, world!")
# ```
# """

#     @classmethod
#     def parse_action_from_text(cls, text: str) -> Optional[Action]:
#         matches = re.findall(r'CreateFile\(filepath=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```', text, flags=re.DOTALL)
#         if matches:
#             filepath = matches[-1][0].strip()
#             code = matches[-1][2].strip()
#             return cls(code=code, filepath=remove_quote(filepath))
#         return None
    
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}(filepath='{self.filepath}':\n'''\n{self.code}\n''')"
       
# @dataclass
# class EditFile(Action):
#     action_type: str = field(
#         default="edit_file",
#         init=False,
#         repr=False,
#         metadata={"help": 'type of action, c.f., "edit_file"'}
#     )

#     code: str = field(
#         metadata={"help": 'code to write into file'}
#     )

#     filepath: Optional[str] = field(
#         default=None,
#         metadata={"help": 'name of file to edit'}
#     )

#     def __repr__(self) -> str:
#         return f"EditFile(filepath=\"{self.filepath}\"):\n```\n{self.code.strip()}\n```"

#     @classmethod
#     def get_action_description(cls) -> str:
#         return """
# ## EditFile
# Signature: EditFile(filepath="path/to/file"):
# ```
# file_content
# ```
# Description: This action will overwrite the file specified in the filepath field with the content wrapped in paired ``` symbols. Normally, you need to read the file before deciding to use EditFile to modify it.
# Example: EditFile(filepath="hello_world.py"):
# ```
# print("Hello, world!")
# ```
# """

#     @classmethod
#     def parse_action_from_text(cls, text: str) -> Optional[Action]:
#         matches = re.findall(r'EditFile\(filepath=(.*?)\).*?```[ \t]*(\w+)?[ \t]*\r?\n(.*)[\r\n \t]*```', text, flags=re.DOTALL)
#         if matches:
#             filepath = matches[-1][0].strip()
#             code = matches[-1][2].strip()
#             return cls(code=code, filepath=remove_quote(filepath))
#         return None
  

# @dataclass
# class WebSearch(Action):

#     action_type: str = field(
#         default="web_search",
#         init=False,
#         repr=False,
#         metadata={"help": 'type of action, c.f., "web_search"'}
#     )

#     link: Optional[str] = field(
#         default=None,
#         metadata={"help": 'link to the website'}
#     )

#     @classmethod
#     def get_action_description(cls):
#         return """
# Action: WebSearch(link=\"web_url\")
# Description: This action will fetch content from the web when you need extra information to make decisions. You need to provide the `link` field which is an accessible website. The fetched content is a preprocessed HTML page by removing function tags (e.g., script, style) and extracting useful tags (e.g., p, li, div, span). The extracted content is then concatenated into a single string. If the preprocessed string is too long, it will be truncated.
# Usage: WebSearch(link="https://en.wikipedia.org/wiki/Main_Page")
# Observation: The observation space is the preprocessed HTML page content.
# """

#     @classmethod
#     def parse_action_from_text(cls, text: str) -> Optional[Action]:
#         matches = re.findall(r'WebSearch\(link=(.*?)\)', text, flags=re.DOTALL)
#         if matches:
#             link = matches[-1]
#             return cls(link=remove_quote(link))
#         return None



# @dataclass
# class PackageInstall(Action):

#     action_type: str = field(
#         default="package_install",
#         init=False,
#         repr=False,
#         metadata={"help": 'type of action, c.f., "package_install"'}
#     )

#     package: str = field(
#         metadata={"help": 'package to install'}
#     )

#     @classmethod
#     def get_action_description(cls) -> str:
#         return """
# Action: PackageInstall(package=\"package_name\")
# Description: This action string will install a python package in the `package` field for the environment. Notice that, you can only install a package that is available via the pip command. You can also specify the version of the package to install, such as package="numpy==1.19.5". If you want to install multiple packages, use whitespaces to separate them, such as package="numpy pandas".
# Usage: PackageInstall(package="sklearn")
# Observation: The observation space is a text message indicating whether this action is executed successfully or not.
# """

#     @classmethod
#     def parse_action_from_text(cls, text: str) -> Optional[Action]:
#         matches = re.findall(r'PackageInstall\(package=(.*?)\)', text, flags=re.DOTALL)
#         if matches:
#             package = matches[-1]
#             return cls(package=remove_quote(package))
#         return None
