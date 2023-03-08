
class ParameterScheduler:

	def __init__(self, param_name=None):
		self.param_name = param_name

	def get(self, iteration):
		raise NotImplementedError

	def info(self):
		return self._scheduler_description() + \
			   " for parameter %s" % str(self.param_name) if self.param_name is not None else ""

	def _scheduler_description(self):
		raise NotImplementedError

class SlopeScheduler(ParameterScheduler):

	def __init__(self, start_val, end_val, stepsize, logit_factor=0, delay=0, param_name=None):
		super().__init__(param_name=param_name)
		self.start_val = start_val
		self.end_val = end_val
		self.logit_factor = logit_factor
		self.stepsize = stepsize
		self.delay = delay
		assert self.stepsize > 0


	def get(self, iteration):
		if iteration < self.delay:
			return self.start_val
		else:
			iteration = iteration - self.delay
			return self.get_val(iteration)


	def get_val(self, iteration):
		raise NotImplementedError

class ExponentialScheduler(SlopeScheduler):

	def __init__(self, start_val, end_val, logit_factor, stepsize, delay=0, param_name=None):
		super().__init__(start_val=start_val, 
						 end_val=end_val, 
						 logit_factor=logit_factor, 
						 stepsize=stepsize, 
						 delay=delay, 
						 param_name=param_name)


	def get_val(self, iteration):
		return self.start_val + (self.end_val - self.start_val) * (1 - self.logit_factor ** (-iteration*1.0/self.stepsize))


	def _scheduler_description(self):
		return "Exponential Scheduler from %s to %s with logit %s and stepsize %s" % \
				(str(self.start_val), str(self.end_val), str(self.logit_factor), str(self.stepsize))