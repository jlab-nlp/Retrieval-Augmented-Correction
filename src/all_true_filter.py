import os
from task_utils import get_outputs
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs", type=str, required=True, help="model_outputs path")
    parser.add_argument("--corrected_outputs", type=str, required=True, help="corrected_outputs path")
    parser.add_argument("--verification_results", type=str, required=True, help="verification_results path")
    args = parser.parse_args()
    verification_results = get_outputs(args.verification_results)
    model_outputs = get_outputs(args.model_outputs)
    corrected_outputs = get_outputs(args.corrected_outputs)
    print(len(model_outputs))
    assert len(model_outputs) == len(verification_results)
    #exit(0)
    all_trues = []
    for verification_result in verification_results:
        all_true_t = True
        results = verification_result.strip("\n").split("\n")
        for result in results:
            if "false" in result.lower():
                all_true_t = False
        all_trues.append(all_true_t)
    new_model_outs = []
    for model_output, corrected_output, all_true in zip(model_outputs, corrected_outputs, all_trues):
        if all_true:
            new_model_outs.append(model_output)
        else:
            new_model_outs.append(corrected_output)
    final_output_name = args.corrected_outputs.split("/")[1].split(".")[0]
    out_file_path = os.path.join("outputs", final_output_name+"-atnc.out")
    with open(out_file_path, "w") as f:
        for model_out in new_model_outs:
            model_outt = model_out.strip("\n")
            f.write(model_outt + "\n---\n")
