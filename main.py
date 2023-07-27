import pickle
import numpy as np
from tensorflow import keras

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

################################################################################################################

latent_dim = 256
num_encoder_tokens = 62
num_decoder_tokens = 148
max_encoder_seq_length = 24
max_decoder_seq_length = 26


################################################################################################################


def data_load():
    file_path = "input_token_index.pkl"
    with open(file_path, "rb") as file:
        input_token_index = pickle.load(file)

    file_path = "target_token_index.pkl"
    with open(file_path, "rb") as file:
        target_token_index = pickle.load(file)

    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model("rutu_v2")

    return model, input_token_index, target_token_index


model, input_token_index, target_token_index = data_load()

################################################################################################################

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,), name="wow")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]

decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)

decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.

reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


################################################################################################################

def rutu(roman_urdu):
    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq, verbose=0)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # print(target_seq.shape)

        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index[" "]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).

        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value, verbose=0)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == " " or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence

    def for_pred(texts):
        text_vec = np.zeros((1, 24, 62), dtype="float32")
        for i, text in enumerate(texts):
            for t, char in enumerate(text):
                text_vec[i, t, input_token_index[char]] = 1.0
            text_vec[i, t + 1:, input_token_index[" "]] = 1.0
            # text_vec[i, t, input_token_index[char]]
            return text_vec

    def sentence_pred(text):
        words = text.lower().split(" ")
        res = ""
        for w in words:
            res += decode_sequence(for_pred([w]))
        return res.replace("\n", "")

    urdu = sentence_pred(roman_urdu)

    return urdu


################################################################################################################

class InputData(BaseModel):
    text: str


@app.post("/process")
def process_text(data: InputData):
    # Access the text from the request
    input_text = data.text

    # Process the input_text using your model
    output_text = rutu(input_text)

    return {"Urdu": output_text}

################################################################################################################
