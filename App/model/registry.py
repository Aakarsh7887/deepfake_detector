import os

BASE_DIR = os.path.dirname(__file__)

MODEL_REGISTRY = {
    10: ["model_84_acc_10_frames_final_data.pt"],
    20: ["model_87_acc_20_frames_final_data.pt", "model_90_acc_20_frames_FF_data.pt"],
    40: ["model_89_acc_40_frames_final_data.pt", "model_95_acc_40_frames_FF_data.pt"],
    60: ["model_90_acc_60_frames_final_data.pt", "model_97_acc_60_frames_FF_data.pt"],
    80: ["model_97_acc_80_frames_FF_data.pt"],
    100: ["model_93_acc_100_frames_celeb_FF_data.pt", "model_97_acc_100_frames_FF_data.pt"]
}