from typing import Pattern

def read_meta_tag_value(meta_data: str, re: Pattern[str]):
    meta_val = re.search(meta_data)
    
    if meta_val:
        return meta_val.group(1).strip("'")

    return ''