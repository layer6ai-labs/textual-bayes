sudo apt install git-lfs

# arc, commensense_qa, openbookqa, boolq, and mmlu will be directly loaded from huggingface

# git clone https://huggingface.co/datasets/allenai/ai2_arc
# git clone https://huggingface.co/datasets/tau/commonsense_qa
# git clone https://huggingface.co/datasets/allenai/openbookqa
# git clone https://huggingface.co/datasets/google/boolq
git clone https://github.com/wilburOne/cosmosqa.git

# wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
# mkdir -p mmlu
# tar xvf data.tar -C mmlu --strip-components=1
# rm data.tar

wget https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip
unzip mutlirc-v2.zip
mv splitv2 multirc
rm mutlirc-v2.zip
