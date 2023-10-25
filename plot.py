import os
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def make_table(songs, sort_key="Beat F-measure"):
    # first sort by ascending f-measure on beats
    songs = sorted(songs, key=lambda k: k[sort_key])

    table = ""
    table += "| File     | dataset | Time Sig.| Beat F-measure |  Downbeat F-measure |\n"
    table += "|:---------|-------|----------|---------------:|--------------------:|\n"

    for song in songs:
        table += f"""| {os.path.basename(song["Filename"])} |"""
        table += f"""  {song["dataset"]} |"""
        table += f"""  {song["Time signature"]} |"""
        table += f"""{song["Beat F-measure"]:0.3f} | """
        table += f"""{song["Downbeat F-measure"]:0.3f} |\n"""

    return table

def plt_netout(pred, target):
    # only numpy can be the input array
    plt.figure(figsize=(12,3), dpi=300)
    beat_output = pred[0,:]
    beat_target = target[0,:]
    
    x_axis = np.arange(pred.shape[-1])
    
    plt.subplot(211)
    plt.plot()
    plt.plot(x_axis, beat_output, linewidth=1)
    plt.subplot(212)
    plt.plot()
    plt.plot(x_axis, beat_target, linewidth=1)
    plt.show()
    plt.savefig("test.png")
    
    
def plt_netout(pred, target, ifbeat=True):
    # only numpy can be the input array
    plt.figure(figsize=(12,3), dpi=300)
    if ifbeat:
        beat_output = pred[0,:]
        beat_target = target[0,:]
    else:
        beat_output = pred[1,:]
        beat_target = target[1,:]       
    
    x_axis = np.arange(pred.shape[-1])
    
    plt.subplot(211)
    plt.plot()
    plt.plot(x_axis, beat_output, linewidth=1)
    plt.subplot(212)
    plt.plot()
    plt.plot(x_axis, beat_target, linewidth=1)
    plt.show()
    plt.savefig("test.png")
    
def plt_eval(t, p, est_beats, length = 300, target_sample_rate=16000):
    # p_rbeats = np.zeros((t.shape[-1]))
    p_ebeats = np.zeros((t.shape[-1]))
    if length:
        x_axis = np.arange(length)
    else:
        length = p_ebeats.shape[-1]
        x_axis = np.arange(p_ebeats.shape[-1])
    # for item in int(est_beats):
    #     p_ebeats[item] = 1
    p_ebeats[est_beats.astype(int)] = 1
    
    plt.subplot(311)
    plt.plot()
    plt.plot(x_axis, t[:length], linewidth=1)
    plt.subplot(312)
    plt.plot()
    plt.plot(x_axis, p[:length], linewidth=1)
    plt.subplot(313)
    plt.plot()
    plt.plot(x_axis, p_ebeats[:length], linewidth=1)
    plt.show()
    plt.savefig("eval_vis.png")
    