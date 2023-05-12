from typing import Dict
import argparse
import json

from tqdm import tqdm


def generate_prompt(data_point: Dict) -> str:
    """
    Generate a prompt for GPT-2.
    :param data_point: A data point from the dataset.
    :return: A prompt for GPT-2.
    """
    if data_point["input"]:
        return f"""### Інструкція:
{data_point["instruction"].strip()}
### Вхідні дані:
{data_point["input"].strip()}
### Відповідь:
{data_point["output"].strip()}"""
    else:
        return f"""### Інструкція:
{data_point["instruction"].strip()}
### Відповідь:
{data_point["output"].strip()}"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate prompts for GPT-2.")
    parser.add_argument("input", type=str, help="Path to the input file.")
    parser.add_argument("output", type=str, help="Path to the output file.")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fp_in, open(
        args.output, "w", encoding="utf-8"
    ) as fp_out:
        for line in tqdm(fp_in):
            data_point = json.loads(line)
            fp_out.write(generate_prompt(data_point) + "\n\n")
