from datasets.generated_set import run_generated_set
from datasets.nyc import run_nyc_set
from baseline_models import run_kcpe_wrapper, run_rulsif_wrapper
from atrcspd import run_atr_wrapper
from argparse import ArgumentParser


def run(model_name, dataset):
    if model_name == 'atrcspd':
        if dataset == 'generated':
            atr_func = run_atr_wrapper(window_size=288, penalty_coefficient=.00005, weight_decay=1e-6)
            run_generated_set(atr_func)
        elif dataset == 'nyc':
            atr_func = run_atr_wrapper(window_size=336, penalty_coefficient=.002, distance_threshold=.1)
            run_nyc_set(atr_func)
    
    elif model_name == 'rulsif':
        if dataset == 'generated':
            rulsif_func = run_rulsif_wrapper(window_size=288)
            run_generated_set(rulsif_func)
        elif dataset == 'nyc':
            rulsif_func = run_rulsif_wrapper(window_size=336, threshold=.02)
            run_nyc_set(rulsif_func)
    
    elif model_name == 'kcpe':
        if dataset == 'generated':
            kcpe_func = run_kcpe_wrapper(pen=8, window_size=288*2, jump=8)
            run_generated_set(kcpe_func)
        elif dataset == 'nyc':
            kcpe_func = run_kcpe_wrapper(window_size=2*48, pen=1, min_size=48*7, jump=48)
            run_nyc_set(kcpe_func)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='atrcspd', help="Model to run can be either: atrcspd, rulsif (RDR in the paper) or kcpe.")
    parser.add_argument('--dataset', type=str, default='generated', help="Dataset to run on, can be either: generated or nyc")
    args = parser.parse_args()

    run(args.model, args.dataset)
