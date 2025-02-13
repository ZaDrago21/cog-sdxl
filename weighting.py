import re

def parse_generic_weights(text: str) -> str:
    pattern = r'\(([^:]+):\s*([^)]+?)\s*\)'
    matches = re.finditer(pattern, text)
    results = []
    for match in matches:
        original_match = match.group(0)
        content = match.group(1).strip()
        number_str = match.group(2).strip()
        try:
            number = float(number_str)
        except ValueError:
            number = None
        results.append((original_match, content, number))
    for original, content, number in results:
        text = text.replace(original, f"({content}){number}")
    return text

def parse_nai_weights(text: str) -> str:
    pattern = r'(\{+)(\w+)(\}+)|(\[+)([^\]]+)(\]+)'
    matches = re.finditer(pattern, text)
    results = {}
    for match in matches:
        original_match = match.group(0)
        if match.group(1):
            num_open = len(match.group(1))
            num_close = len(match.group(3))
            num_braces = min(num_open, num_close)
            name = match.group(2)
            results[name] = (original_match, name, num_braces)
        elif match.group(4):
            num_open = len(match.group(4))
            num_close = len(match.group(6))
            num_braces = min(num_open, num_close)
            name = match.group(5)
            results[name] = (original_match, name, num_braces)
    for original_match, name, num_braces in results.values():
        if original_match[0] == "[":
            calculated_value = round(1.0 / (1.05 ** num_braces), 4)
        else:
            calculated_value = round(1.05 ** num_braces, 4)
        text = text.replace(original_match, f"({name}){calculated_value}")
    return text

def parse_webui_weights(text: str) -> str:
    pattern = r'(?<!_|\\)(\(+)(\w+)(\)+)(?!_|\\|\d+)|(\[+)([^\]]+)(\]+)(?!_|\\|\d+)'
    matches = re.finditer(pattern, text)
    results = {}
    for match in matches:
        original_match = match.group(0)
        if match.group(1):
            num_open = len(match.group(1))
            num_close = len(match.group(3))
            num_braces = min(num_open, num_close)
            name = match.group(2)
            results[name] = (original_match, name, num_braces)
        elif match.group(4):
            num_open = len(match.group(4))
            num_close = len(match.group(6))
            num_braces = min(num_open, num_close)
            name = match.group(5)
            results[name] = (original_match, name, num_braces)
    for original_match, name, num_braces in results.values():
        if original_match[0] == "[":
            calculated_value = round(1.0 / (1.1 ** num_braces), 4)
        else:
            calculated_value = round(1.1 ** num_braces, 4)
        text = text.replace(original_match, f"({name}){calculated_value}")
    return text

def parse_weights(text: str) -> str:
    if "{" in text:
        text = parse_nai_weights(text)
        text = parse_webui_weights(text)
        text = parse_generic_weights(text)
    else:
        text = parse_webui_weights(text)
        text = parse_nai_weights(text)
        text = parse_generic_weights(text)
    return text 