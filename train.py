import argparse
from trainer import Trainer

from param_file.params import *
from simple_test import test_func

def train(args):
    # train or test
    if args.mode == 'train':
        ## -- create log file -- ##
        log_file_name = 'log/' + args.network + '_' + args.mark + '.log'
        args.log_file = open(log_file_name, 'ab') ##
        # arg_settings_file = 

        trainer = Trainer(args)
        trainer.train()
    elif args.mode == 'test':
        tester = test_func(args)
        tester.test()
    elif args.mode == "real_time":
        tester = test_func(args)
        tester.calculate_rtf()
    return 0

if __name__ == '__main__':
    # avoid fault in dataloader
    # torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    
    # training level args input
    parser.add_argument('-n', "--network", default='lc42_dt_1')
    parser.add_argument('-m', "--mark", default='round4_device1') # 需要标注是否是完全数据集，是否使用八倍交叉验证,是否是multi-task
    
    parser.add_argument('-p', "--pp_network", default='cnn1')
    parser.add_argument("--mark_pp", default='round2_device0') 
    
    parser.add_argument('-md', "--mode", default='test') # real_time, train, test， train_pp
    
    parser.add_argument('--eval_mode', default = "online") # online, offline
    parser.add_argument('--inference_model', default = "simple") # simple, DBN， dl, PF
    parser.add_argument('--peak_type', default = "librosa") # librosa, simple, thresh
    parser.add_argument('--peak_latency', type=int, default=5)
    
    # testing level args settings
    parser.add_argument("--check_num", default = "135") # 0 refer to the latest file
    parser.add_argument("--check_num_pp", default = "79") # 0 refer to the latest file
    parser.add_argument("--test_all", default = True)
    parser.add_argument("--folder", type=int, default = 0, help="-1,0,1,2,3,4,5,6,7") # eight fold validation num

    # training/testing mode
    parser.add_argument("--only_beat", default = False)
    parser.add_argument("--only_downbeat", default = False)##
    
    # specific training settings    
    parser.add_argument("-me","--max_epochs", type=int, default=250)   
    parser.add_argument("--patience", type=int, default=50)

    parser.add_argument('--train_subset', type=str, default='train')
    parser.add_argument('--val_subset', type=str, default='val')
    
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001) 
    parser.add_argument('--lr_pp', type=float, default=0.001) 
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--device_index', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4) # 4
    
    # loss settings 
    parser.add_argument('--loss', type=str, default="softbce_n") 
    parser.add_argument('--loss_param', type=float, default=10) 
    parser.add_argument("--clip_grad", type=float, default=0.5) # no use
    
    parser.add_argument('--checkname', type=int, default=210)
    parser.add_argument('--ifcheckpoint', default=True)

    # 测试代码模式是否开启
    parser.add_argument("--epoch_batch", type=int, default=20)
    parser.add_argument('--iftestmode', default=False)
    
       
    args = parser.parse_args()   
    
    args.length_sec = length_sec
    args.train_length = train_length
    args.eval_length = eval_length
    args.target_length = target_frame
    args.limit_length = limit_length

    args.train_sets = train_sets
    args.test_sets = test_sets
    args.model_path = MODEL_PATH
    args.pp_model_path = MODEL_PATH_PP
    args.scalar_writer = SCALAR_WRITER
    args.scalar_writer_pp = SCALAR_WRITER_PP
        
    args.sr = target_sr
    args.hop_length = hop_length
    args.target_factor = hop_length
    
    args.root_path = root_path    

    train(args)


