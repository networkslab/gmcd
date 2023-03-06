from src.synthetic_run_config import SyntheticRunConfig
from src.train import start_training
from src.train import TrainSyntheticModeling


S =6 
dataset = 'sort'


def train(set_size, dataset):

    run_config = SyntheticRunConfig(model='FCDM', dataset=dataset, S=set_size)
    start_training(run_config, TrainSyntheticModeling)
       


#train(set_size=S, max_iterations=5000, encoding_dim=5,  dataset=dataset)
# main_path = 'checkpoints/'+dataset+'/S_' + \
#     str(S)+'_K_'+str(S)
# if S == 6:
#     print_table = print_table_6
# else:
#     def print_table(result, title): return print_other_table(
#         result, title, S, dataset)
# eval(main_path, run_eval, lambda list_sample_eval, big_sample_eval, detailed_metrics_test: compile_result(list_sample_eval, big_sample_eval, detailed_metrics_test,
#                                                                                                           S, S), print_table,best=True)
