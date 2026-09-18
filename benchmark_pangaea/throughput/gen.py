OPTICAL_ENCODERS = ['croma_optical', 'dofa_optical', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'spectralgpt', 'ssl4eo_moco', 'terramind_optical_tiny', 'prithvi2_100m']
# Encoders that also take the AGBD SAR modality; `dofa_joint` is the multimodal DOFA
# (`dofa_optical` above is its optical-only counterpart).
SAR_ENCODERS = ['croma_joint', 'terramind_tiny', 'dofa_joint']

for encoder in OPTICAL_ENCODERS + SAR_ENCODERS :
    command = f"python throughput.py task=regression dataset=agbdlite encoder={encoder} decoder=reg_upernet preprocessing=reg_resize criterion=mse batch_size=32 --warmup 20 --iterations 100"
    print(command)