# /home/sushant/D1/HEDGE/redo_experimnts_2026/llava-med-v1.5-mistral-7b-hfkvasir_vqa_x1_test_answers_hallucination_label_added.parquet.json
import pandas as pd

core_experiment_outputs = [
        {"model": "llava-med-v1.5-mistral-7b", "dataset": "vqa_rad",
        "file": "../results/redo_2026/llava-med-v1.5-mistral-7b-hfvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "llava-med-v1.5-mistral-7b", "dataset": "kvasir_vqa_x1",
        "file": "../results/redo_2026/llava-med-v1.5-mistral-7b-hfkvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        {"model": "medgemma-4b-it", "dataset": "vqa_rad",
        "file": "../results/redo_2026/medgemma-4b-itvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "medgemma-4b-it", "dataset": "kvasir_vqa_x1",
        "file": "../results/redo_2026/medgemma-4b-itkvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},

        {"model": "Qwen2.5-VL-7B-Instruct", "dataset": "vqa_rad",
        "file": "../results/redo_2026/Qwen2.5-VL-7B-Instructvqa_rad_test_answers_hallucination_label_added.parquet"},

        {"model": "Qwen2.5-VL-7B-Instruct", "dataset": "kvasir_vqa_x1",
        "file": "../results/redo_2026/Qwen2.5-VL-7B-Instructvqa_kvasir_vqa_x1_test_answers_hallucination_label_added.parquet"},
]
import json
answer_only = {}
answer_question_passed = {}
for output in core_experiment_outputs:
    for append_question_label in [False, True ]:
        dataset_id = output['dataset']
        filename = output['file']
        answers = pd.read_parquet(filename)
        if append_question_label==False:
            finalfilename = f"{filename}"
        else:
            finalfilename = f"{filename}_append_question.parquet"
        print(f"Model: {output['model']}, Dataset: {output['dataset']}, File: {output['file']}")
        
        path = f"../results/redo_2026/{finalfilename}.json"
        data = json.load(open(path, 'r'))
        # make all values to 3 decimal places recursively
        def round_values(obj):
            if isinstance(obj, dict):
                return {k: round_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [round_values(elem) for elem in obj]
            elif isinstance(obj, float):
                return round(obj, 3)
            else:
                return obj
        data = round_values(data)
        if append_question_label == False:
            answer_only[(output['model'], output['dataset'])] = data
        else:
            answer_question_passed[(output['model'], output['dataset'])] = data
        breakpoint()

print(answer_only  )
print(answer_question_passed)
breakpoint()