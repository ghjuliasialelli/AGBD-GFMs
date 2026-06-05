for encoder in ['croma_optical', 'dofa', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'spectralgpt', 'ssl4eo_moco', 'terramind_optical_tiny', 'prithvi2_100m'] :
    command = f"python throughput.py task=regression dataset=agbdlite encoder={encoder} decoder=reg_upernet preprocessing=reg_resize criterion=mse batch_size=32 --warmup 20 --iterations 100"
    print(command)