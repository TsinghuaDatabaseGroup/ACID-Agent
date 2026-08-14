import re

def parse_token_counts(trace: str):
    """
    Given the full LLM trace string, extract input/output token counts
    from the final '[Step ...]' line.

    Returns:
        (input_tokens, output_tokens) as integers, or (0, 0) on failure.
    """
    if not isinstance(trace, str):
        print(f"Trace is not a string (type: {type(trace)}) Cannot parse token counts. Returning 0")
        return 0, 0

    # Extract the last line
    last_line = trace.strip().splitlines()[-1]

    # Regex to extract token counts, allowing commas in numbers
    pattern = r"Input tokens:\s*([\d,]+)\s*\|\s*Output tokens:\s*([\d,]+)"

    m = re.search(pattern, last_line)
    if not m:
        print(f"Cannot parse token counts with regex pattern. Returning 0")
        return 0, 0

    # Remove commas and convert to int
    input_tokens = int(m.group(1).replace(",", ""))
    output_tokens = int(m.group(2).replace(",", ""))

    return input_tokens, output_tokens
