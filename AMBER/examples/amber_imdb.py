import numpy as np
import tensorflow_datasets as tfds
from amber import Amber
from amber.architect import ModelSpace, Operation
from keras.utils.np_utils import to_categorical   


class Tokenizer(object):
    def __init__(self, 
                 chars='abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} ',
                 unk_token=True):
        self.chars = chars
        self.unk_token = 69 if unk_token else None

        self.build()

    def build(self):
        """Build up char2idx.
        """
        self.idx = 1    # idx 0 reserved for zero padding
        self.char2idx = {}
        self.idx2char = {}

        for char in self.chars:
            self.char2idx[char] = self.idx
            self.idx2char[self.idx] = char
            self.idx += 1

    def char_to_idx(self, 
                    c):
        """Return the integer character index of a character token.
        """
        if not c in self.char2idx:
            if self.unk_token is None:
                return None   # Return None if no unknown word's defined
            else:
                return self.unk_token

        return self.char2idx[c]

    def idx_to_char(self, 
                    idx):
        """Return the character string of an integer word index.
        """
        # Unknown token
        if idx > len(self.idx2char):
            if self.unk_token is None:
                return ''
            else:
                return '<UNK>'

        # Return nothing for zero padding
        elif idx == 0:
            return ''
        
        return self.idx2char[idx]

    def __len__(self):
        """Return the length of the vocabulary.
        """
        return len(self.char2idx)

    def text_to_sequence(self, 
                         text,
                         maxlen=1014):
        text = text.lower() # Forced lower casing, as specified in VDCNN paper

        data = np.zeros(maxlen, ).astype(int)
        for i in range(len(text)):
            if i >= maxlen:
                return data
            if text[i] in self.char2idx:
                data[i] = self.char_to_idx(text[i])
        return data

def utf8_to_sequence(text, maxlen=1014):
    text = text.decode('utf-8')
    text = text.lower()
    data = np.zeros((maxlen, 1)).astype(int)
    for i in range(len(text)):
        if i>= maxlen:
            return data
        data[i] = ord(text[i])
    return data 


def get_model_space(out_filters=64, num_layers=9):
    model_space = ModelSpace()
    num_pool = 4
    expand_layers = [num_layers//4-1, num_layers//4*2-1, num_layers//4*3-1]
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu', dilation=10),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu', dilation=10),
            # max/avg pool has underlying 1x1 conv
            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space


# First, define the components we need to use
type_dict = {
    'controller_type': 'GeneralController',
    'modeler_type': 'EnasCnnModelBuilder',
    'knowledge_fn_type': 'zero',
    'reward_fn_type': 'LossAucReward',
    'manager_type': 'EnasManager',
    'env_type': 'EnasTrainEnv'
}

all_data = tfds.load(
        name="imdb_reviews",
        split=('train'),
        as_supervised=True)

validation_gen = tfds.as_numpy(all_data.take(5000))
train_gen = tfds.as_numpy(all_data.skip(5000))
train_data = [(x,y) for x,y in train_gen]
validation_data = [(x,y) for x,y in validation_gen]

tokenizer = Tokenizer()

#train_x = np.array([tokenizer.text_to_sequence(x.decode('utf-8')) for x,y in train_data])
train_x = np.array([utf8_to_sequence(x) for x,y in train_data])
#train_x = np.expand_dims(train_x, axis=2)
train_y = np.array([y for x,y in train_data])

#valid_x = np.array([tokenizer.text_to_sequence(x.decode('utf-8')) for x,y in validation_data])
#valid_x = np.expand_dims(valid_x, axis=2)
valid_x = np.array([utf8_to_sequence(x) for x,y in validation_data])
valid_y = np.array([y for x,y in validation_data])

CHAR_MAX_LEN = 1014

print(train_x.shape)
print(train_y.shape)

wd = "./outputs/AmberIMDB/"
input_node = Operation('input', shape=(CHAR_MAX_LEN, 1), name="input")
#embedding_node = Operation('embedding', input_dim=69, output_dim=16, input_length=CHAR_MAX_LEN, name='embed')
output_node = Operation('dense', units=1, activation='sigmoid')
model_compile_dict = {
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
}
model_space = get_model_space(out_filters=32, num_layers=12)

specs = {
    'model_space': model_space,
    
    'controller': {
            'share_embedding': {i:0 for i in range(1, len(model_space))},
            'with_skip_connection': True,
            'num_input_blocks': 1,
            'skip_connection_unique_connection': False,
            'skip_weight': 1.0,
            'skip_target': 0.4,
            'lstm_size': 64,
            'lstm_num_layers': 1,
            'kl_threshold': 0.01,
            'train_pi_iter': 10,
            'optim_algo': 'adam',
            'temperature': 2.,
            'lr_init': 0.001,
            'tanh_constant': 1.5,
            'buffer_size': 1,  
            'batch_size': 20
    },

    'model_builder': {
        'dag_func': 'EnasConv1dDAG',
        'batch_size': 500,
        'inputs_op': [input_node],
        'outputs_op': [output_node],
        'model_compile_dict': model_compile_dict,
         'dag_kwargs': {
            'stem_config': {
                'flatten_op': 'flatten',
                'fc_units': 925
            }
        }
    },

    'knowledge_fn': {'data': None, 'params': {}},

    'reward_fn': {'method': 'auc'},

    'manager': {
        'data': {
            'train_data': (train_x, train_y),
            'validation_data': (valid_x, valid_y),
        },
        'params': {
            'epochs': 1,
            'child_batchsize': 64,
            'store_fn': 'minimal',
            'working_dir': wd,
            'verbose': 1
        }
    },

    'train_env': {
        'max_episode': 300,
        'max_step_per_ep': 100,
        'working_dir': wd,
        'time_budget': "24:00:00",
        'with_input_blocks': False,
        'with_skip_connection': True,
        'child_train_steps': 500,
        'child_warm_up_epochs': 1
    }
}


# finally, run program
amb = Amber(types=type_dict, specs=specs)
amb.run()
