

class SamplesMetrics():
    def __init__(self, samples_np, metrics, histogram=None):
        self.samples_np = samples_np  # numpy array of samples
        self.metrics = metrics  # computed metrics on the samples
        self.histogram = histogram  # optional histogram view of the samples.

    def add_new_metrics(self, metrics):
        self.metrics.update(metrics)

    def get_printable_metrics_dict(self):
        printable_metrics_dict = {}
        for key, val in self.metrics.items(): # we only print string/int or float values. (Avoid any list or dict).
            if isinstance(val, str) or  isinstance(val, int) or isinstance(val, float):
                printable_metrics_dict[key] = val
        return printable_metrics_dict