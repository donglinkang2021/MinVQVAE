# Define the session name
model="ffnae"
devices="4,5,6,7"
# devices="0,1,2,3"
SESSION_NAME="exp-${model}-0"
CONDA_NAME="linkdom"

# Create a new tmux session with the specified name
tmux new-session -d -s $SESSION_NAME

# Create a new window for training the model
tmux new-window -t ${SESSION_NAME}:1 -n 'train'
tmux send-keys -t ${SESSION_NAME}:1 "conda activate ${CONDA_NAME}" C-m
tmux send-keys -t ${SESSION_NAME}:1 "export CUDA_VISIBLE_DEVICES=${devices}" C-m
tmux send-keys -t ${SESSION_NAME}:1 "python train.py logger.name=${SESSION_NAME} model=${model}" C-m

# Create a new window for running tensorboard
# tmux new-window -t ${SESSION_NAME}:2 -n 'tensorboard'
# tmux send-keys -t ${SESSION_NAME}:2 "conda activate ${CONDA_NAME}" C-m
# tmux send-keys -t ${SESSION_NAME}:2 "tensorboard --logdir=/data1/linkdom/output/logs --bind_all" C-m

# Select the first window by default
tmux select-window -t ${SESSION_NAME}:1

# Attach to the tmux session
tmux attach-session -t $SESSION_NAME
