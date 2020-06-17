#Experiment
from comet_ml import Experiment
from DeepTreeAttention.main import AttentionModel

experiment = Experiment(project_name="deeptreeattention", workspace="bw4sz")

#Create a class and run
model = AttentionModel()
model.create()
model.read_data()

#How big is the data
counter = 0 
for data, label in model.testing_set:
    print(data.shape)
    counter+=1
    
    
#Log config
experiment.log_parameters(model.config)
model.train()
results = model.evaluate()
experiment.log_metrics(results)



