## -- global params -- ##
target_sr = 16000

length_sec = 30 # 8 for bilstm, 30
limit_sec = 8 # 可以进行evaluate的最小长度
hop_length = 160 # 160

train_length = target_sr * length_sec
eval_length = train_length
limit_length = target_sr * limit_sec
target_frame = round(target_sr/hop_length*length_sec)+1
limit_frame = round(target_sr/hop_length*limit_sec)+1

SCALAR_WRITER = "./vis_scalar"
SCALAR_WRITER_PP = "./vis_scalar_pp"
MODEL_PATH = './model'
MODEL_PATH_PP = './model_pp'

## -- data path -- ##
root_path = '/home/fd-lamt-04/2T/beat_chord_hdf5'

## -- training and validing files list -- ##
train_sets = [   
    'hjdb',       # beat/tempo
    'hainsworth', # beat/tempo
    'rwc_popular',# beat/downbeat/tempo/chord
    "ballroom",
    "beatles2",
    "smc",
    # "gtzan",
]

## -- testing file list -- ##
test_sets = [
    'gtzan',      # beat/tempo
    # 'ballroom',   # beat/tempo  
    # 'hainsworth', 
    # 'hjdb',
    # 'smc',        # beat/tempo
    # "beatles2",
    # 'rwc_popular',
]

