from genericpath import exists
import pickle as pk
import os

class SamplesEvaluation():
    def __init__(self, samples=None, metrics_dict={}, all_dict={}):
        self.samples = samples
        if self.samples is not None:
            self.num_samples = samples.shape[0]
        if metrics_dict:
            self.metrics_dict = metrics_dict
            self.all_dict = all_dict
            self.all_dict.update(metrics_dict)
        else:
            self.metrics_dict = {}
            self.all_dict = all_dict
            for key, val in all_dict.items():
                if isinstance(val, float) or isinstance(val, int) :
                    self.metrics_dict[key] = val


    def train_string(self):
        if self.metrics_dict:
            #base_string = self.base_sample_eval.train_string()
            return self.metrics_dict
        else:
            return ""

    def get_printable_metrics_dict(self):
        return self.metrics_dict

    def get_all_metrics_dict(self):
        return self.all_dict

    def store(self, path, index=None):
        if index is None:
            file_name = 'sample_result.pk'
        else:
            file_name = str(index)+'sample_result.pk'
        file_path = os.path.join(path, file_name)

        with open(file_path, 'wb') as f:
            pk.dump(self.get_all_metrics_dict(), f)
        if index is None:
            file_name = 'samples.pk'
        else:
            file_name = str(index)+'sample.pk'
        file_path = os.path.join(path, file_name)

        with open(file_path, 'wb') as f:
            pk.dump(self.samples, f)

    def is_stored(self, path, index):
        file_name = str(index)+'sample_result.pk'
        file_path = os.path.join(path, file_name)
        return exists(file_path)

    def get(self, path, index):
        file_name = str(index)+'sample_result.pk'
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            get_all_metrics_dict = pk.load(f)

        file_name = str(index)+'sample.pk'
        file_path = os.path.join(path, file_name)
        with open(file_path, 'rb') as f:
            samples = pk.load(f)

        sample = SamplesEvaluation(samples=samples, all_dict=get_all_metrics_dict)
        return sample

        
