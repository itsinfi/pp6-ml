from typing import Pattern

def read_meta_tag_value(meta_data: str, re: Pattern[str]):
    meta_val = re.search(meta_data)
    
    if meta_val:
        return [part.strip() for part in meta_val.group(1).strip().split(',') if part.strip()] or []

    return []