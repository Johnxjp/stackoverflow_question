from keras.models import Model
from keras.layers import LSTM, Dense, Input, Lambda
from keras import backend as K
import numpy as np

class BenchmarkModel(object):

    def __init__(self,
                 latent_dims=1,
                 classifier_model=None,
                 word2id=None,
                 start_character="<go>",
                 stop_character="<eos>",
                 dropout=0,
                 max_timestep=None,
                 max_timestep_classifier=None,
                 train_classifier=False):

        # Length of the inputs is len(vocab) + 1, the +1 is for the style label
        self.latent_dims = latent_dims
        self.input_dims = len(word2id)
        self.word2id = word2id
        self.start_character = start_character
        self.stop_character = stop_character
        self.max_timestep = max_timestep

        # Encoder
        print("Initialising Encoder Forward...", end="", flush=True)
        # Input shape = (time-steps, input_dim). (None, input_dim) means the number of time-steps is variable (can't do this for batches)
        self.encoder_inputs = Input(shape=(max_timestep, self.input_dims), name="encoder_inputs")
        self.encoder = LSTM(self.latent_dims, return_state=True, dropout=dropout, name="encoder")
        encoder_output, state_h, state_c = self.encoder(self.encoder_inputs)

        # Discard the output and only keep the state information to pass to the decoder
        self.encoder_states = [state_h, state_c]
        print("done")

        # Decoder
        # Enforce same parameters on decoder as encoder for now
        # No teacher forcing - decoder only processes one input at a time
        print("Initialising Decoder Forward...", end="", flush=True)
        self.decoder_inputs_forward = Input(shape=(1, self.input_dims), name="decoder_inputs_forward")  # How to feed output back into input? Should I set Max?
        self.decoder = LSTM(self.latent_dims, return_sequences=True, return_state=True, dropout=dropout, name="decoder")
        self.decoder_dense_1 = Dense(self.input_dims, activation="softmax")

        # Forward Transfer
        softmax_output_forward = self._decoder_serial_input(self.decoder_inputs_forward,
                                                            self.encoder_states,
                                                            max_timestep)
        print("done")

        print("Decoder_output_forward: ", softmax_output_forward)

        # Classifier layer forwards
        print("Initialising Classifier Forward...", end="", flush=True)
        classifier = classifier_model
        classifier_output_forward = classifier(softmax_output_forward)
        print("done")

        # Backward Transfer
        print("Initialising Encoder Backward...", end="", flush=True)
        encoder_output, state_h, state_c = self.encoder(softmax_output_forward)
        self.encoder_states = [state_h, state_c]
        print("done")

        # I need to pass whole sentence in here, the input is not just one at a time
        print("Initialising Decoder Backward...", end="", flush=True)
        self.decoder_inputs_backwards = Input(shape=(max_timestep, self.input_dims), name="decoder_inputs_backward")  # Same as encoder
        decoder_output_backward, _, _ = self.decoder(inputs=self.decoder_inputs_backwards, initial_state=self.encoder_states)
        softmax_output_backward = self.decoder_dense_1(decoder_output_backward)
        print("done")

        # Classifier layer backwards
        print("Initialising Classifier Backward...", end="", flush=True)
        classifier_output_backward = classifier(softmax_output_backward)  # Get a loss from this!

        self.model = Model(inputs=[self.encoder_inputs, self.decoder_inputs_forward, self.decoder_inputs_backwards],
                           outputs=[classifier_output_forward, classifier_output_backward, softmax_output_backward])
        print("done")

        # Set training mode of classifier
        if not train_classifier:
            print("Setting classifier parameters to untrainable")
            for layer in classifier.layers:
                layer.trainable = False

        print("\n\n")
        print("Note: Before compiling, some parameters may still appear trainable even when set to untrained.")
        print(self.model.summary())

    def _decoder_serial_input(self, decoder_input, initial_state, max_timestep):
        """
        Compute one-by-one input to decoder, taking output from previous time-step as input
        :param decoder_input: Input layer to decoder
        :param initial_state: starting state for decoder LSTM
        :return: ?
        """

        # Should we also stop when eos generated?
        all_outputs = []
        inputs = decoder_input
        states = initial_state
        for _ in range(max_timestep):
            decoder_output_forward, state_h, state_c = \
                self.decoder(inputs, initial_state=states)

            outputs = self.decoder_dense_1(decoder_output_forward)
            all_outputs.append(outputs)

            inputs = outputs
            states = [state_h, state_c]

        # Concatenate all predictions into a single sequence (Possible source of error) - what is the shape?
        return Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    def train(self, inputs, validation, labels, losses, batch_size, epochs, loss_weights, optimizer="adadelta", callbacks=None):
            pass

    def inference(self, input_sequence, id2word, max_sequence_length):
        """
        New model for inference which uses the same setup as the training model but different outputs

        :param input_sequence: sequence to input to model
        :param id2word: dictionary with key, val = id, word
        :param max_sequence_length: maximum length of input sequences. Should be same as initialised before.
        :return: decoded sequence as a string
        """

        # Define encoder model
        # This is a copy of the model that is defined in the constructor.
        """
        ********Problem statement**********
        """
        encoder_inference = Model(self.encoder_inputs, self.encoder_states)

        # Decoder with output at time t feeding back into input at t+1
        decoder_state_input_h = Input(shape=(self.latent_dims,))  # This comma means one
        decoder_state_input_c = Input(shape=(self.latent_dims,))
        decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_outputs, state_h, state_c = self.decoder(self.decoder_inputs_forward, initial_state=decoder_state_inputs)

        decoder_states = [state_h, state_c]

        decoder_outputs = self.decoder_dense_1(decoder_outputs)

        # Decoder Inference Model:
        # Inputs -- Input layer + state information. If time-step is variable, then can be just one
        # Outputs -- outputs and state information, to be fed into the model again
        decoder_inference = Model([self.decoder_inputs_forward] + decoder_state_inputs,
                                   [decoder_outputs] + decoder_states)

        return self._decode_sequence(encoder_inference, decoder_inference, input_sequence, id2word, max_sequence_length)

    def _decode_sequence(self, encoder, decoder, input_sequence, id2word, max_sequence_length):
        # Encode the input as state vectors
        states_value = encoder.predict(input_sequence)

        """
        Might not need the below if the input sequences are already in one-hot state
        
        Unsure why this is not the same as the above?
        """
        # Generate empty target sequence of length 1
        target_seq = np.zeros((1, 1, self.input_dims))

        # Populate the first character of target sequence with the start character
        target_seq[0, 0, self.word2id[self.start_character]] = 1

        # Sampling loop for a batch of sequences (in this example, simplified so batch is 1) #TODO
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = id2word[sampled_token_index] # will this always return something legit? [defo because softmax is in vocab range]
            decoded_sentence += (sampled_word + " ")

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_word == self.stop_character or len(decoded_sentence) > max_sequence_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.input_dims))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
