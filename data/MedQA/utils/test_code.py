

import json
from pathlib import Path

data_path = Path(__file__).resolve().parents[1] / "data_clean/questions_with_knowledge/US/test/retrive_use_question/stella-base-en-v2/test.json"
with open(data_path,'r',encoding='utf-8') as f :
    raw_json_data = json.load(f)

print(raw_json_data[0])
