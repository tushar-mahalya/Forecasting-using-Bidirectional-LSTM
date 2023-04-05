import pandas as pd
from collections import OrderedDict
from termcolor import colored as color
from Data_Initializer import train_test_data

def test_models(data_dict, epochs: int, models: dict, window_sizes: list):
    
    #test result data
    parameter_dict = OrderedDict()
    sequences = ['Simple', 'Multi']
    for ticker in data_dict.keys():
        
        seq_type = {}
        print(color(f'\n================================ NSE : {ticker} ================================', 'red', attrs = ["bold"]))
        for seq in sequences:
            print(color('\n--------------------------------------------------------------- ', 'green') + color(f'{seq} Sequence' , 'green'))
            
            win_size = {}
            for window_size in window_sizes:

                model_name = {}
                print(color(f"\nWindow size: {window_size}", 'blue'))
                print(color('----------------', 'yellow'))
                for model_item in models:

                    data = pd.read_csv(data_dict[ticker])
                    X_train,y_train,X_test,y_test = train_test_data(data, split_size = 0.2, window_size = window_size, seq = seq)
                    model = model_item[1](X_train,y_train,0.001)

                    # fitting model
                    model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)

                    # printing training and testing errors
                    training_error = model.evaluate(X_train, y_train, verbose=0)
                    testing_error = model.evaluate(X_test, y_test, verbose=0)
                    msg = color("  >",'green') + " Model: {0:<15} Param count: {1:} \tTraining error: {2:.4f}\tTesting error: {3:.4f}"
                    print(msg.format(model_item[0],model.count_params(),training_error,testing_error))

                    model_name[model_item[0]] = {
                                                 'Param Count' : model.count_params(),
                                                 'Training Error' : float("{0:.4f}".format(training_error)),
                                                 'Testing Error' : float("{0:.4f}".format( testing_error))
                                                }

                win_size[f'Window_Size_{window_size}'] = model_name
                
            seq_type[f'{seq} Sequence'] = win_size
        
        parameter_dict[ticker] = seq_type
        
    # Exporting performance of different models in .json format
    with open("Performance_Parameters.json", "w") as f:
        json.dump(parameter_dict, f)
        f.close()
   
            
    print(color('\nSuccesfully Evaluated different models and saved the performance metrics in \'Performance Parameters\' directory.', 'magenta'))
        
    return parameter_dict
