
import datetime
import os
import pickle as pk


def store_model_dict(path, runconfig):
    meta_data = os.path.join(path, 'run_info.txt')
    meta_data_file = os.path.join(path, 'run_info.pk')
    if not os.path.exists(meta_data):
        with open(meta_data, 'w') as file:
            file.write('Model Param...' + "\n\n")
            for key, val in runconfig.__dict__.items():
                line = str(key) + ":" + str(val) + "\n"
                file.write(line)
        with open(meta_data_file, 'wb') as file:
            pk.dump(runconfig, file)


def prepare_checkpoint(checkpoint_path, name_prefix):
    if checkpoint_path is None:
        current_date = datetime.datetime.now()
        base_path = "checkpoints/%s/%02d_%02d_%02d__%02d_%02d/" % (
            (name_prefix) if len(name_prefix) > 0 else "",
            current_date.day, current_date.month, current_date.year,
            current_date.hour, current_date.minute)
        figure_path = os.path.join(base_path, 'figure')
        checkpoint_path = os.path.join(base_path, 'cktp')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            os.makedirs(figure_path)
    else:
        base_path = checkpoint_path.split('/cktp')[0]
        figure_path = os.path.join(base_path, 'figure')

    return checkpoint_path, figure_path


def check_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)

    print('Total number of trainable parameters', pytorch_total_params)
    print()


# Function to export the current results to a txt file
def export_result_txt(best_save_iter, best_save_dict, checkpoint_path):
    if best_save_iter is not None:
        with open(os.path.join(checkpoint_path, "results.txt"), "w") as f:
            f.write("Best validation performance: %s\n" %
                    (str(best_save_dict["metric"])))
            f.write("Best iteration: %i\n" %
                    int(str(best_save_iter).split("_")[-1].split(".")[0]))
            f.write("Best checkpoint: %s\n" % str(best_save_iter))
            f.write("Detailed metrics\n")
            for metric_name, metric_val in best_save_dict[
                    "detailed_metrics"].items():
                f.write("-> %s: %s\n" % (metric_name, str(metric_val)))
            if "test" in best_save_dict and best_save_dict["test"] is not None:
                f.write("Test - Detailed metrics\n")
                for metric_name, metric_val in best_save_dict["test"].items():
                    f.write("[TEST] -> %s: %s\n" %
                            (metric_name, str(metric_val)))
            f.write("\n")

    # Function for saving model. Add here in the dictionary necessary parameters that should be saved


def save_train_model_fun(no_model_checkpoints,
                         best_save_dict,
                         evaluation_dict,
                         iteration,
                         save_model,
                         only_weights=True):
    if no_model_checkpoints:
        return
    checkpoint_dict = {
        "iteration": iteration,
        "best_save_dict": best_save_dict,
        "evaluation_dict": evaluation_dict
    }
    save_model(iteration, checkpoint_dict, save_optimizer=not only_weights)


def print_detailed_scores_and_sampling(detailed_scores, sample_eval):

    for key, val in detailed_scores.items():
        if val is not None:
            print(key+" : {:.4f}".format(val), end=" ")
    for key, val in sample_eval.get_printable_metrics_dict().items():
        print(key+" : {:.4f}".format(val), end=" ")
    print()


def check_if_best_than_saved(last_save, eval_NLL, detailed_scores,
                             best_save_dict, index_iter,
                             get_checkpoint_filename, checkpoint_path, evaluation_dict, save_train_model):
    # If model performed better on validation than any other iteration so far => save it and eventually replace old model
    if eval_NLL < best_save_dict["metric"]:
        best_save_iter = get_checkpoint_filename(index_iter + 1)
        best_save_dict["metric"] = eval_NLL
        best_save_dict["detailed_metrics"] = detailed_scores
        if not os.path.isfile(best_save_iter):
            print("Saving model at iteration " + str(index_iter + 1))
            if best_save_dict["file"] is not None and os.path.isfile(
                    best_save_dict["file"]):
                print("Removing checkpoint %s..." % best_save_dict["file"])
                os.remove(best_save_dict["file"])
            if last_save is not None and os.path.isfile(last_save):
                print("Removing checkpoint %s..." % last_save)
                os.remove(last_save)
            best_save_dict["file"] = best_save_iter
            last_save = best_save_iter
            save_train_model(index_iter + 1)
        export_result_txt(best_save_iter, best_save_dict, checkpoint_path)
    evaluation_dict[index_iter + 1] = best_save_dict["metric"]





def get_sampling_stat_if_not_too_long(time_for_sampling, train_time_avg, loss_freq, evaluate_sample, NUM_SAMPLES, overide=False):

    if overide:  # just do it regardless
        sample_eval = evaluate_sample(
            num_samples=NUM_SAMPLES)
    else:
        time_for_print = train_time_avg*loss_freq
        if time_for_sampling < time_for_print/20 or time_for_sampling < 2 or time_for_print < 2:
            sample_eval = evaluate_sample(
                num_samples=NUM_SAMPLES)

        elif time_for_sampling < time_for_print/10:
            sample_eval = evaluate_sample(
                num_samples=int(NUM_SAMPLES/2))
        else:  # just take too long, forget it
            sample_eval = evaluate_sample(
                num_samples=0, return_empty=True)

    return sample_eval
