def get_dataset_name(fid='data_util/dataset_name.txt'):
    '''
    get the dataset name from wherever she happens to be
    '''
    with open( fid ,'r') as f:
        return f.readline().strip()
