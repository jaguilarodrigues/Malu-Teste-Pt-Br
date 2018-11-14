from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.train import online
from rasa_core.utils import EndpointConfig
from rasa_core.run import serve_application

logger = logging.getLogger(__name__)

def train_dialogue(domain_file = 'malu_domain.yml',
					model_path = './models/dialogue',
					training_data_file = './data/stories.md'):

	fallback = FallbackPolicy(fallback_action_name="action_default_fallback",
                              core_threshold=0.01,
                              nlu_threshold=0.01)
	agent = Agent(domain_file, policies = [MemoizationPolicy(max_history=2), KerasPolicy(), fallback])
	data = agent.load_data(training_data_file)
	agent.train(
				data,
				epochs = 300,
				batch_size = 50,
				validation_split = 0.2)

	agent.persist(model_path)
	return agent

def run_malu_bot(serve_forever=True):
	interpreter = RasaNLUInterpreter('./models/nlu/default/malu') #carrega o modelo de nlu
	action_endpoint = EndpointConfig(url="http://localhost:5055/webhook")
	agent = Agent.load('./models/dialogue', interpreter=interpreter, action_endpoint=action_endpoint) #carregar um agente
	serve_application(agent ,channel='cmdline')

	return agent

if __name__ == '__main__':
	train_dialogue()
	run_malu_bot()
