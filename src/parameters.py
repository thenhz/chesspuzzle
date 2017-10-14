from src.Utils import now

app_path = '/home/thenhz/TensorFlow/ChessPuzzle/'+now()
chess_size = 16
max_episode_length = 300
gamma = .99 # discount rate for advantage estimation and reward discounting
s_size = chess_size #7056 # Observations are greyscale frames of 84 * 84 * 1
a_size = chess_size*chess_size #3 # Agent can move Left, Right, or Fire
load_model = False
model_path = app_path+'/model'
img_path = app_path+'/img'
tensorboard_path = app_path+'/tensorboard'

