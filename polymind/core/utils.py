import json
import re
from typing import Any, Dict, List, get_args, get_origin

from polymind.core.tool import BaseTool, Param


def json_text_to_tool_param(json_text: str, tool: BaseTool) -> Dict[str, Any]:
    """Convert the JSON text that contains the params to call the tool to the tool parameter dictionary.

    Args:
        json_text (str): The JSON text that contains the params to call the tool.
        tool_input_spec (List[Param]): The tool input specification.
    """
    # Extract and parse the JSON text to the dictionary
    if "```" in json_text:
        groups = re.findall(r"```json(.*?)```", json_text, re.DOTALL)
        if groups:
            tool_param = groups[0]
        else:
            raise ValueError(f"The JSON text is not in the correct format. {json_text}")
    else:
        tool_param = json_text
    tool_param_dict = json.loads(tool_param)
    input_spec: List[Param] = tool.input_spec()

    # Convert the string to the correct type according to the input_spec
    tool_param_dict_typed = {}
    for param in input_spec:
        param_name = param.name
        if param_name in tool_param_dict:
            param_value = tool_param_dict[param_name]
            expected_param_type = eval(param.type)
            try:
                if get_origin(expected_param_type) is list:
                    if isinstance(param_value, str) or not isinstance(param_value, list):
                        raise ValueError(f"The field '{param_name}' must be a list, but got '{param_value}'.")
                    element_type = get_args(expected_param_type)[0]
                    tool_param_dict_typed[param_name] = [convert_value(item, element_type) for item in param_value]
                elif get_origin(expected_param_type) is dict:
                    key_type, value_type = get_args(expected_param_type)
                    if not isinstance(param_value, dict):
                        raise ValueError(f"The field '{param_name}' must be a dictionary, but got '{param_value}'.")
                    tool_param_dict_typed[param_name] = {
                        convert_value(k, key_type): convert_value(v, value_type) for k, v in param_value.items()
                    }
                else:
                    tool_param_dict_typed[param_name] = convert_value(param_value, expected_param_type)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"{tool.tool_name}: The field '{param_name}' must be of type '{param.type}', "
                    f"but failed to convert the value '{param_value}': {e}"
                )
        elif param.required:
            raise ValueError(f"The required parameter {param_name} is not provided.")

    return tool_param_dict_typed


def convert_value(value: Any, target_type: type) -> Any:
    """Convert the value to the target type."""
    if target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == bool:
        return bool(value)
    elif target_type == str:
        return str(value)
    else:
        return value
