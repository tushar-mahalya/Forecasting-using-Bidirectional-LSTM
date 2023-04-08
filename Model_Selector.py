import os
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import OrderedDict
from termcolor import colored as color
from Models import models
from keras.models import save_model
from Data_Initializer import train_test_data


def test_models(data_dict, epochs: int, models: dict, window_sizes: list):
    
    #test result data
    best_params_dict = OrderedDict()
    parameter_dict = OrderedDict()
    sequences = ['Simple', 'Multi']
    
    for ticker in data_dict.keys():
        
        lowest_train_error = 1.0
        lowest_test_error = 1.0
        best_model = str()
        best_model_param_count = int()
        best_window_size = int()
        best_seq = str()
        
        seq_type = {}
        print(color(f'\n================================ NSE : {ticker} ================================', 'red', attrs = ["bold"]))
        for seq in sequences:
            print(color('\n--------------------------------------------------------------- ', 'green') + color(f'{seq} Sequence' , 'green'))
            
            win_size = {}
            for window_size in window_sizes:

                model_name = {}
                print(color(f"\nWindow size : {window_size}", 'blue'))
                print(color('----------------', 'yellow'))
                for model_type in models.keys():

                    data = pd.read_csv(data_dict[ticker])
                    X_train,y_train,X_test,y_test = train_test_data(data, split_size = 0.2, window_size = window_size, seq = seq)
                    model = models[model_type](X_train,y_train,0.001, model_name = f'{ticker.replace("&", "and")}_Model')

                    # fitting model
                    model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)

                    # printing training and testing errors
                    training_error = model.evaluate(X_train, y_train, verbose=0)
                    testing_error = model.evaluate(X_test, y_test, verbose=0)
                    msg = color("  >",'green') + " Model: {0:<15} Param count: {1:} \tTraining error: {2:.4f}\tTesting error: {3:.4f}"
                    print(msg.format(model_type,model.count_params(),training_error,testing_error))

                    model_name[model_type] = {
                                              'Param Count' : model.count_params(),
                                              'Training Error' : float("{0:.4f}".format(training_error)),
                                              'Testing Error' : float("{0:.4f}".format( testing_error))
                                             }
                    
                    if testing_error < lowest_test_error:
                        lowest_train_error = training_error
                        lowest_test_error = testing_error
                        best_model = model_type
                        best_model_param_count = model.count_params()
                        best_window_size = window_size
                        best_seq = seq
                        
                win_size[f'Window_Size_{window_size}'] = model_name
                
            seq_type[f'{seq} Sequence'] = win_size
        
        parameter_dict[ticker] = seq_type
        best_params_dict[ticker] = {
                               'Best Model' : best_model,
                               'Window Size' : best_window_size,
                               'Sequence' : best_seq,
                               'Training Error' : lowest_train_error,
                               'Testing Error' : lowest_test_error,
                               'Learning Rate' : 0.001,
                               'Parameters Count' : best_model_param_count
                              }
        
    # Exporting performance of different models and best parameters dict in .json format
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, 'Model Performance')):
        print("Required data/directory is already present in current working directory.")
    
    else:
        os.mkdir('Model Performance')
        with open("Model Performance/Performance_Params.json", "w") as f:
            json.dump(parameter_dict, f)
            f.close()
        with open("Model Performance/Best_Model_Params.json", "w") as f:
            json.dump(best_params_dict, f)
            f.close()
   
            
    print(color('\nSuccesfully Evaluated different models and saved all logs and best model performance metrics in \'Model Performance\' directory.', 'magenta'))
        
    return parameter_dict, best_params_dict



def save_models(data_dict: dict, epochs: int, models: dict, best_params: dict):
    
    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, 'Models')):
        print("Required data/directory is already present in current working directory.")
    
    else:
        os.mkdir('Models')
    
    track = 1
    
    for ticker in data_dict.keys():
        
        best_model = best_params[ticker]['Best Model']
        win_size = best_params[ticker]['Window Size']
        seq_type = best_params[ticker]['Sequence']
        lr = best_params[ticker]['Learning Rate']
        
        data = pd.read_csv(data_dict[ticker])
        X_train,y_train,X_test,y_test = train_test_data(data, split_size = 0.2, window_size = win_size, seq = seq_type)
        model = models[best_model](X_train,y_train, lr, model_name = f'{ticker.replace("&", "and")}_Model')
        
        print(color(f'\n({track}) NSE:{ticker}' , 'red', attrs = ['bold']))
        print(color('-----------------------', 'yellow'))
        print(color(' >> Model : ', 'blue') + f'{best_model}' + color('\t >> Window Size : ', 'blue') + f'{win_size}' + color('\n >> Seq. Type : ', 'blue') + f'{seq_type} Sequence' + color('\t >> Epochs : ', 'blue') + f'{epochs}')
        
        # fitting model
        model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)
        
        testing_error = model.evaluate(X_test, y_test, verbose=0)
        training_error = model.evaluate(X_train, y_train, verbose=0)
        
        print(color(' >> Training Error : ', 'blue') + '{0:.4f}'.format(training_error) + color('\t >> Testing Error : ', 'blue') + '{0:.4f}'.format(testing_error))
        
        # Save the trained model
        model.save(f'Models/{ticker}.h5')
        
        track += 1
        
        print(color('\nSuccesfully saved best performing models in \'Models\' directory.', 'magenta'))