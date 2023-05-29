import os
from datetime import datetime


# save bertopic model
def save_bertopic_model(model, filename = 'bertopic_', model_dir = "/kaggle/working/model"):
    # retrieve time
    now = datetime.now()
    # create the directory if it doesn't exist
    try:
        os.makedirs(model_dir)
    except:
        pass
    model.save(model_dir + '/' + filename + now.strftime("%d%m%Y_%H%M%S"))